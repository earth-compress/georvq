"""Executables of the AlphaEarth release codec.

Everything here encodes AE-RELEASE POLICY, not general georvq behavior:
the pixel prep mirrors the AEF int8/-128 conventions, the 8* sidecar and
quality ladder are decisions validated for this source, and the tile
container (georvq.ae.cog) mirrors the AEF source layout. Other sources
should design their own release layer against georvq.inference.

Three console entry points (see pyproject):

  georvq-ae-export-onnx  checkpoint/bundle -> ONNX export dir (+ parity)
  georvq-ae-patch        DEV TOOL: pre-cut (B, 64, W, W) windows <-> npz
  georvq-ae-tile         the release codec: source tile <-> COG asset
                         directory (encode/decode/info)

Both codecs share `--backend {torch,onnx}` and `--device {cpu,cuda}`:
torch runs the checkpoint (CPU or CUDA), onnx runs an exported dir through
onnxruntime with no torch dependency at all. All backends read and write
the same container format (`.gvq`, an npz), so a file encoded on GPU
decodes on a CPU-only box and vice versa.

Container: npz with a JSON header plus per-tier code-index arrays
(uint16/uint32, deflate-compressed by the npz layer). Entropy coding was
measured and deliberately dropped: order-0 rANS saved only ~0.6% on real
AlphaEarth tiles (the fine tier runs at ~95% of codebook capacity and
deflate recovers the rest) while dominating decode time. Indices are
stored full-depth; any monotone depth tuple can be chosen at decode time
(`--tuple 6,4,4`).

Nodata: the source's validity mask (band 0 == -128, the convention the
model was trained with — inputs zeroed, loss masked) rides along
bit-packed in the container whenever any pixel is invalid. Fully-nodata
windows are never encoded (decode rebuilds them from the mask), and
decode restores the source's exact fill value on int8 output (zero
vectors on float output). `check_fill` warns at encode time if masked
pixels are not pure fill, since then byte-exact restoration is impossible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

NODATA = -128
FORMAT = "georvq-ae/2"


# --------------------------------------------------------------------------
# AlphaEarth pixel prep (numpy mirror of sources.ae.AESource.prep)
# --------------------------------------------------------------------------

def prep_ae_np(s):
    """int8 (B, C, W, W) -> (float32 unit-norm x, valid mask)."""
    s = np.asarray(s)
    valid = s[:, 0] != NODATA
    x = s.astype(np.float32) / 127.5
    x = np.square(x) * np.sign(x)
    x *= valid[:, None]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-6), valid


def requant_ae_np(x, valid=None, fill=NODATA):
    """float32 field -> int8 signed-sqrt quantization (inverse of dequant);
    invalid pixels -> `fill` (the source's own nodata value)."""
    s = np.rint(np.sign(x) * np.sqrt(np.abs(x)) * 127.5)
    s = np.clip(s, -127, 127).astype(np.int8)
    if valid is not None:
        s[np.broadcast_to(~valid[:, None], s.shape)] = fill
    return s


def check_fill(a, valid, fill=NODATA):
    """Verify masked pixels carry exactly `fill` in every band, so decode
    can promise byte-exact nodata restoration. Returns True when it can."""
    if valid.all():
        return True
    bad = a[np.broadcast_to(~valid[:, None], a.shape)] != fill
    if bad.any():
        print(f"warning: {int(bad.sum())} nodata-masked samples are not "
              f"uniformly {fill}; decode will normalize them to {fill}",
              file=sys.stderr)
        return False
    return True


# --------------------------------------------------------------------------
# 8* coarse-residual sidecar (2-bit, per-window 2-sigma scale, 66 B/window)
# --------------------------------------------------------------------------

STAR_BITS, STAR_SIGMA = 2, 2.0


def pack_star(res):
    """(N, cells, dim) float residuals -> (codes uint8 packed, step f16).
    Symmetric mid-rise uniform quantizer, 4 levels at (+-0.5, +-1.5)*step,
    step = STAR_SIGMA*std/1.5 per window. Retains 98% of the float-star
    gain incl. the OOD tail (per-window scale adapts to outliers)."""
    r = res.reshape(res.shape[0], -1).astype(np.float32)
    step = (STAR_SIGMA * r.std(axis=1, keepdims=True) / 1.5)         .clip(min=1e-12)
    c = np.clip(np.round(r / step - 0.5) + 0.5, -1.5, 1.5) + 1.5  # {0..3}
    packed = np.packbits(
        ((c.astype(np.uint8)[:, :, None] >> np.arange(1, -1, -1)) & 1)
        .reshape(c.shape[0], -1), axis=1)
    return packed, step[:, 0].astype(np.float16)


def unpack_star(packed, step, cells, dim):
    """Inverse of pack_star -> (N, cells, dim) float32."""
    bits = np.unpackbits(packed, axis=1, count=cells * dim * 2)         .reshape(packed.shape[0], -1, 2)
    c = (bits[:, :, 0] << 1 | bits[:, :, 1]).astype(np.float32)
    r = (c - 1.5) * step.astype(np.float32)[:, None]
    return r.reshape(-1, cells, dim)


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

class TorchBackend:
    """numpy-in / numpy-out adapter over the torch Codec.

    Accepts raw int8 AE windows directly (prep runs on the device — the
    numpy dequant+normalize was ~90 s/tile of CPU time) and runs its
    forwards in bf16 on CUDA unless fp32=True (dcos impact: mean 1.8e-5,
    two orders below the p50 reconstruction error)."""

    def __init__(self, model_path, device="cpu", fp32=False):
        import torch
        self.torch = torch
        self.bf16 = device.startswith("cuda") and not fp32
        if os.path.isdir(model_path):
            from georvq.inference.release import load_release
            self.codec = load_release(model_path, device)
        else:
            from georvq.inference.api import load_codec
            self.codec = load_codec(model_path, device)
        self.window = self.codec.window
        self.bands = self.codec.model.bands
        self.specs_sig = [(sp.cells, sp.depth, sp.codes)
                          for sp in self.codec.specs]
        self.device = device
        sp0 = self.codec.specs[0]
        self.star_cost_bits = sp0.cells * sp0.dim * STAR_BITS + 16
        from georvq.inference.release import bitstream_id, decoder_id
        self.bitstream_id = bitstream_id(self.codec.model)
        self.decoder_id = decoder_id(self.codec.model)

    def _ac(self):
        import contextlib
        return self.torch.autocast("cuda", dtype=self.torch.bfloat16) \
            if self.bf16 else contextlib.nullcontext()

    def _prep(self, arr):
        """int8 (or prepped float) numpy -> (x, valid) device tensors."""
        t = self.torch
        a = t.from_numpy(np.ascontiguousarray(arr)).to(self.device)
        if a.dtype == t.int8:
            valid = a[:, 0] != NODATA
            x = a.float().div_(127.5)
            x = x.square().mul_(t.sign(x)) * valid.unsqueeze(1)
            import torch.nn.functional as F
            x = F.normalize(x, dim=1, eps=1e-6)
        else:
            x = a.float()
            valid = x.norm(dim=1) > 0.5
        return x, valid

    def encode(self, x, coarse_residual=False):
        # encoder + assignment stay fp32: bf16 latents flip ~24% of
        # near-tie RVQ indices (quality-neutral but bitstream-unstable);
        # bf16 buys its speed on the decoder forwards instead
        xt, _ = self._prep(x)
        return self.codec.encode(xt, coarse_residual=coarse_residual)

    def decode(self, packets, depths=None, int8=False):
        with self._ac():
            out = self.codec.decode(packets, depths)
        out = out.float()
        if int8:      # requant on device: 4x less transfer, no CPU sqrt
            t = self.torch
            out = t.clamp(t.round(t.sign(out) * t.sqrt(out.abs()) * 127.5),
                          -127, 127).to(t.int8)
        return out.cpu().numpy()

    def encode_and_stats(self, x, valid, ladder, coarse_residual=False):
        """Fused encode + ladder stats: one encoder pass, cumulative
        codebook sums built alongside assignment, then one decoder
        forward per ladder point with dcos reduced on-device. Avoids the
        8x re-encode + float round-trips of the generic path."""
        from georvq.inference.api import Packet
        t = self.torch
        m = self.codec.model
        with t.no_grad():
            xt, vt = self._prep(x)
            zs = m.encode_levels(xt)      # fp32: indices stay bit-stable
            cums, idxs, res0 = [], [], None
            for ti, (z, rvq) in enumerate(zip(zs, m.rvqs)):
                residual = z.float()
                acc = t.zeros_like(residual)
                levels, per = [acc], []
                for lev in rvq.levels:
                    q, idx, _ = lev(residual)
                    residual = residual - q
                    acc = acc + q
                    levels.append(acc)
                    per.append(idx)
                cums.append(t.stack(levels, 2))
                idxs.append(t.stack(per, -1).cpu().numpy())
                if ti == 0 and coarse_residual:
                    res0 = residual
            star_q = None
            if res0 is not None:      # measure what decode will apply:
                rf = res0.reshape(res0.shape[0], -1)   # the 2-bit sidecar
                step = (STAR_SIGMA / 1.5
                        * rf.std(dim=1, unbiased=False, keepdim=True)) \
                    .clamp_min(1e-12)
                star_q = (t.clamp(t.round(rf / step - 0.5) + 0.5,
                                  -1.5, 1.5) * step).reshape(res0.shape)
            d0 = m.specs[0].depth
            dcos = {}
            for pt in ladder:
                zqs = [cums[i][:, :, L] for i, L in enumerate(pt)]
                # decode applies the sidecar whenever coarse is at full
                # depth — true at every ladder point; measure the same
                if pt[0] == d0 and star_q is not None:
                    zqs[0] = zqs[0] + star_q
                Ls = t.tensor(list(pt), device=self.device)[None] \
                    .expand(xt.shape[0], -1)
                with self._ac():
                    out = m.decode(zqs, Ls)
                d = 1.0 - (out.float() * xt).sum(1)
                wm = (d * vt).sum((1, 2)) / vt.sum((1, 2)).clamp_min(1)
                dcos[pt] = (d[vt].clamp_min(1e-9).cpu().numpy(),
                            wm.cpu().numpy())
        res_np = res0.cpu().numpy() if res0 is not None else None
        packets = [Packet(
            indices=[a[b] for a in idxs],
            coarse_residual=res_np[b] if res_np is not None else None)
            for b in range(x.shape[0])]
        return packets, dcos


class OnnxBackend:
    def __init__(self, export_dir, device="cpu", threads=None):
        from georvq.inference.onnx_backend import OnnxCodec
        self.codec = OnnxCodec(export_dir, device=device, threads=threads)
        self.window = self.codec.window
        self.bands = self.codec.bands
        self.specs_sig = [(t["grid"] ** 2, t["depth"], t["codes"])
                          for t in self.codec.tiers]
        self.bitstream_id = self.codec.meta.get("bitstream_id")
        self.decoder_id = self.codec.meta.get("decoder_id")
        t0 = self.codec.tiers[0]
        self.star_cost_bits = t0["grid"] ** 2 * t0["dim"] * STAR_BITS + 16

    def encode(self, x, coarse_residual=False):
        return self.codec.encode(x, coarse_residual=coarse_residual)

    def decode(self, packets, depths=None, int8=False):
        out = self.codec.decode(packets, depths)
        return requant_ae_np(out) if int8 else out


def load_backend(args):
    kind = args.backend
    if kind == "auto":
        kind = "onnx" if (os.path.isdir(args.model) and os.path.exists(
            os.path.join(args.model, "encoder.onnx"))) else "torch"
    if kind == "onnx":
        return OnnxBackend(args.model, device=args.device,
                           threads=getattr(args, "threads", None))
    return TorchBackend(args.model, device=args.device,
                        fp32=getattr(args, "fp32", False))


# --------------------------------------------------------------------------
# quality ladder + encode-time error stats
# --------------------------------------------------------------------------

def quality_ladder(specs_sig):
    """The rate ladder stats are recorded at: max coarse, mid {1,2,4,..,D},
    then fine {1,2,4,..,D} at full mid — the measured Pareto family."""
    def steps(d):
        out, v = [], 1
        while v < d:
            out.append(v); v *= 2
        return out + [d]
    d = [t[1] for t in specs_sig]
    pts = [tuple([d[0], L] + [0] * (len(d) - 2)) for L in steps(d[1])]
    for t in range(2, len(d)):
        base = [d[i] for i in range(t)]
        pts += [tuple(base + [L] + [0] * (len(d) - t - 1))
                for L in steps(d[t])]
    return pts


class QualityStats:
    """Streaming per-ladder-point stats of per-pixel dcos (1 - cos):
    exact mean/max, percentiles via a fine log histogram."""
    EDGES = np.logspace(-8, 0.31, 385)

    def __init__(self, ladder):
        self.ladder = ladder
        self.h = {t: np.zeros(len(self.EDGES) - 1, np.int64) for t in ladder}
        self.acc = {t: [0.0, 0, 0.0] for t in ladder}   # sum, n, max
        self.win = {t: [] for t in ladder}   # per-window mean dcos chunks

    def add(self, t, d, win_mean=None):
        d = np.clip(d, 1e-8, 2.0)
        self.h[t] += np.histogram(d, bins=self.EDGES)[0]
        a = self.acc[t]
        a[0] += float(d.sum()); a[1] += d.size; a[2] = max(a[2], float(d.max()))
        if win_mean is not None:
            self.win[t].append(win_mean)

    def window_means(self):
        """per ladder point: (n_windows,) mean dcos, encode order."""
        return {t: np.concatenate(v) for t, v in self.win.items() if v}

    def _pct(self, t, q):
        c = np.cumsum(self.h[t])
        if c[-1] == 0:
            return 0.0
        i = int(np.searchsorted(c, q / 100.0 * c[-1]))
        return float(self.EDGES[min(i + 1, len(self.EDGES) - 1)])

    def summary(self, specs_sig, star_cost=0):
        """star_cost: bits of the sidecar (0 if absent); it is applied —
        and therefore measured — only at the full-depth point."""
        out = {}
        for t in self.ladder:
            s, n, mx = self.acc[t]
            name = "-".join(map(str, t))
            starred = t[0] == specs_sig[0][1]     # coarse at full depth
            out[name] = {
                "bits": nominal_bits(specs_sig, list(t))
                + (star_cost if starred else 0),
                "star": bool(star_cost and starred),
                "mean": s / max(n, 1), "p50": self._pct(t, 50),
                "p90": self._pct(t, 90), "p99": self._pct(t, 99),
                "max": mx,
            }
        return out


def pick_quality(stats, metric, bound):
    """Cheapest recorded ladder point whose `metric` <= bound, or None."""
    ok = [(v["bits"], k) for k, v in stats.items() if v[metric] <= bound]
    return min(ok)[1] if ok else None


# --------------------------------------------------------------------------
# container (.gvq = npz)
# --------------------------------------------------------------------------

def _index_dtype(codes):
    return np.uint16 if codes <= 2 ** 16 else np.uint32


def write_container(path, packets, specs_sig, header, valid=None):
    """Write N packets (+ header dict) as one npz, laid out progressively.

    One zip member per (tier, level), written in the rate ladder's order —
    all coarse levels, then mid, then fine (the empirical Pareto order:
    within a fine band you always saturate coarse, then mid). A decode at
    depths (L0, L1, L2) therefore touches only the members it needs, and
    ladder tuples read a sequential prefix of the file: a wide-area
    low-rate decode of a full tile reads a few hundred KiB of a ~60 MB
    blob. Indices are the smallest sufficient uint dtype; the npz deflate
    layer compresses each member (within ~0.6% of entropy-coded).

    `valid`: optional bool ndarray of any shape (per-pixel nodata mask,
    True = data). Stored bit-packed only when some pixels are invalid;
    decode uses it to restore the source's exact fill values."""
    payload = {}
    if valid is not None and not valid.all():
        payload["mask"] = np.packbits(valid)
        header = {**header, "mask_shape": list(valid.shape)}
    for t, (cells, depth, codes) in enumerate(specs_sig):
        arr = np.stack([np.asarray(p.indices[t]) for p in packets]) \
            .astype(_index_dtype(codes))                # (N, cells, depth)
        for j in range(depth):
            payload[f"t{t}l{j}"] = np.ascontiguousarray(arr[:, :, j])
        if t == 1 and packets and packets[0].coarse_residual is not None:
            # 8* sidecar sits after the mid levels in the rate ladder
            res = np.stack([p.coarse_residual for p in packets])
            payload["t0star"], payload["t0star_step"] = pack_star(res)
            header = {**header,
                      "t0star": {"bits": STAR_BITS, "sigma": STAR_SIGMA}}
    header = {**header, "format": FORMAT, "n_windows": len(packets),
              "specs_sig": [list(s) for s in specs_sig]}
    np.savez_compressed(path, header=json.dumps(header), **payload)
    return header


def read_container(path, depths=None):
    """-> (header dict, list[Packet], valid mask ndarray or None).

    `depths`: optional per-tier level counts — only those members are
    read and inflated (progressive layout), and the returned packets
    carry exactly that many levels per tier."""
    from georvq.inference.api import Packet
    z = np.load(path, allow_pickle=False)
    header = json.loads(str(z["header"]))
    if header.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} container (v2, "
                         "progressive layout) — re-encode pre-release files")
    sig = [tuple(s) for s in header["specs_sig"]]
    n = header["n_windows"]
    depths = depths or [d for _, d, _ in sig]
    tiers = []
    for t, ((cells, depth, _), L) in enumerate(zip(sig, depths)):
        if L == 0:
            tiers.append(np.zeros((n, cells, 0), np.int64))
            continue
        tiers.append(np.stack([z[f"t{t}l{j}"].astype(np.int64)
                               for j in range(L)], axis=-1))
    star = None
    if "t0star" in header and depths[0] == sig[0][1]:
        dim = z["t0star"].shape[1] * 8 // (2 * sig[0][0])
        star = unpack_star(z["t0star"][:], z["t0star_step"][:],
                           sig[0][0], dim)
    packets = [Packet(indices=[a[i] for a in tiers],
                      coarse_residual=star[i] if star is not None else None)
               for i in range(n)]
    valid = None
    if "mask" in z:
        shape = header["mask_shape"]
        valid = np.unpackbits(z["mask"], count=int(np.prod(shape))) \
            .reshape(shape).astype(bool)
    return header, packets, valid


def container_read_bytes(path, sig, depths=None):
    """(bytes touched for a depths-decode, total bytes) of a container."""
    import zipfile
    depths = depths or [d for _, d, _ in sig]
    names = {"header.npy", "mask.npy"} | \
        {f"t{t}l{j}.npy" for t in range(len(sig)) for j in range(depths[t])}
    if depths[0] == sig[0][1]:
        names |= {"t0star.npy", "t0star_step.npy"}
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
    return (sum(i.compress_size for i in infos if i.filename in names),
            sum(i.compress_size for i in infos))


def nominal_bits(specs_sig, depths=None):
    depths = depths or [d for _, d, _ in specs_sig]
    return float(sum(c * L * np.log2(k)
                     for (c, d, k), L in zip(specs_sig, depths)))


# --------------------------------------------------------------------------
# shared CLI plumbing
# --------------------------------------------------------------------------

def _common_args(ap):
    ap.add_argument("--model", required=True,
                    help="torch checkpoint .pt, release bundle dir, or "
                         "ONNX export dir")
    ap.add_argument("--backend", choices=["auto", "torch", "onnx"],
                    default="auto",
                    help="auto: onnx if --model is an export dir")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default) or cuda[:N]")
    ap.add_argument("--threads", type=int, default=None,
                    help="onnxruntime intra-op threads (onnx backend)")
    ap.add_argument("--fp32", action="store_true",
                    help="run torch forwards in fp32 (default: bf16 on "
                         "CUDA; changes dcos by ~2e-5, far below the "
                         "reconstruction error)")


def _check_bitstream(header, be, path):
    """Same-architecture models with different codebooks decode each
    other's blobs into silent garbage — refuse on id mismatch."""
    want = header.get("bitstream_id")
    if want and be.bitstream_id and want != be.bitstream_id:
        raise SystemExit(
            f"{path}: encoded against bitstream {want}, but --model has "
            f"{be.bitstream_id} — same layout, different codebooks would "
            "decode to garbage. Use the release bundle it was encoded with.")


def _parse_tuple(s, specs_sig):
    if not s:
        return None
    ls = [int(v) for v in s.split(",")]
    if len(ls) != len(specs_sig):
        raise SystemExit(f"--tuple needs {len(specs_sig)} values")
    for (_, d, _), L in zip(specs_sig, ls):
        if not 0 <= L <= d:
            raise SystemExit(f"--tuple {s}: level {L} outside [0, {d}]")
    return ls


def encode_all(be, arr, valid, batch, star, want_stats):
    """-> (packets, QualityStats|None). `arr` may be raw int8 windows
    (preferred: torch preps on-device) or prepped float32. Uses the
    backend's fused encode+stats pass when available, else encode then
    generic stats."""
    if hasattr(be, "encode_and_stats"):
        if not want_stats:
            return _batched(
                lambda ab: be.encode(ab, coarse_residual=star),
                arr, batch), None
        ladder = quality_ladder(be.specs_sig)
        qs, packets = QualityStats(ladder), []
        for i in range(0, len(arr), batch):
            pk, dcos = be.encode_and_stats(arr[i:i + batch],
                                           valid[i:i + batch], ladder,
                                           coarse_residual=star)
            packets += pk
            for pt, (d, wm) in dcos.items():
                qs.add(pt, d, win_mean=wm)
        return packets, qs
    # generic backend: needs prepped float input
    def _prep_b(ab):
        return prep_ae_np(ab)[0] if ab.dtype == np.int8 else ab
    packets = _batched(
        lambda ab: be.encode(_prep_b(ab), coarse_residual=star), arr, batch)
    if not want_stats:
        return packets, None
    x = _prep_b(arr) if arr.dtype == np.int8 else arr
    return packets, collect_stats(be, x, valid, packets, batch)


def collect_stats(be, x, valid, packets, batch):
    """Decode every ladder point and accumulate true per-pixel dcos —
    the encoder is the one party that ever holds the original, so error
    metadata must be minted here."""
    qs = QualityStats(quality_ladder(be.specs_sig))
    for i in range(0, len(x), batch):
        xb, vb = x[i:i + batch], valid[i:i + batch]
        pk = packets[i:i + batch]
        for t in qs.ladder:
            out = be.decode(pk, list(t))
            d = 1.0 - (out * xb).sum(1)
            wm = (d * vb).sum((1, 2)) / np.maximum(vb.sum((1, 2)), 1)
            qs.add(t, d[vb], win_mean=wm)
    return qs


def _read_header(path):
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["header"]))


def _quality_arg(sub):
    sub.add_argument("--quality", default=None, metavar="STAT<=X",
                     help="quality-control decode: cheapest recorded ladder "
                          "point with STAT (mean|p50|p90|p99|max of pixel "
                          "dcos) <= X, e.g. --quality p90<=0.006; needs "
                          "encode-time stats in the blob")


def _resolve_quality(header, spec_arg, specs_sig):
    import re
    m = re.fullmatch(r"(mean|p50|p90|p99|max)<=([0-9.eE+-]+)", spec_arg)
    if not m:
        raise SystemExit(f"--quality {spec_arg}: expected STAT<=X with "
                         "STAT in mean|p50|p90|p99|max")
    stats = header.get("quality_stats")
    if not stats:
        raise SystemExit("this blob carries no quality stats; re-encode "
                         "with stats or use --tuple")
    name = pick_quality(stats, m.group(1), float(m.group(2)))
    if name is None:
        best = min(stats.items(), key=lambda kv: kv[1][m.group(1)])
        raise SystemExit(
            f"no ladder point reaches {spec_arg}; best available is "
            f"{best[0]} with {m.group(1)}={best[1][m.group(1)]:.4g}")
    depths = [int(v) for v in name.split("-")]
    print(f"quality {spec_arg} -> {name} "
          f"({stats[name]['bits']:.0f} bits/window, "
          f"{m.group(1)}={stats[name][m.group(1)]:.4g})")
    return depths


def _batched(fn, arr, batch):
    outs = [fn(arr[i:i + batch]) for i in range(0, len(arr), batch)]
    if isinstance(outs[0], list):
        return [p for o in outs for p in o]
    return np.concatenate(outs)


def _report(tag, header, n_windows, window, file_bytes, t0, depths=None):
    sig = [tuple(s) for s in header["specs_sig"]]
    nom = nominal_bits(sig, depths)
    line = (f"{tag}: {n_windows} windows of {window}px | "
            f"{nom:.0f} nominal bits/window ({nom / window ** 2:.2f} bpp)")
    if file_bytes:  # encode side: the realized file
        fb = 8 * file_bytes / n_windows
        line += (f" | file {fb:.0f} bits/window "
                 f"({fb / window ** 2:.2f} bpp)")
    print(f"{line} | {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------
# georvq-export-onnx
# --------------------------------------------------------------------------

def export_onnx_main(argv=None):
    ap = argparse.ArgumentParser(
        description="Export a georvq checkpoint to an ONNX dir usable by "
                    "the onnx backend of the codec executables.")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint .pt or release bundle dir")
    ap.add_argument("--out", required=True, help="export directory")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    from georvq.inference.onnx_backend import export_onnx
    tb = TorchBackend(args.ckpt, "cpu")
    meta = export_onnx(tb.codec.model, args.out, opset=args.opset)
    print(f"exported {args.out}: window={meta['window']} "
          f"tiers={[t['depth'] for t in meta['tiers']]} "
          f"bitstream_id={meta['bitstream_id']}")

    if not args.skip_verify:
        ob = OnnxBackend(args.out, "cpu")
        rng = np.random.default_rng(0)
        s = rng.integers(-127, 128, (4, meta["bands"], meta["window"],
                                     meta["window"]), dtype=np.int8)
        x, _ = prep_ae_np(s)
        pt, po = tb.encode(x), ob.encode(x)
        agree = np.mean([np.mean(np.asarray(a.indices[t])
                                 == np.asarray(b.indices[t]))
                         for a, b in zip(pt, po)
                         for t in range(len(tb.specs_sig))])
        d_cross = np.abs(tb.decode(po) - ob.decode(po)).max()
        print(f"verify: index agreement {agree:.4%} (fp32-vs-torch "
              f"assignment), decoder max|torch-onnx| {d_cross:.2e} "
              f"on identical packets")
        if d_cross > 1e-3:
            raise SystemExit("verify FAILED: decoder outputs diverge")


# --------------------------------------------------------------------------
# georvq-codec-patch
# --------------------------------------------------------------------------

def _load_windows(path, window, bands):
    a = np.load(path)
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4 or a.shape[1] != bands:
        raise SystemExit(f"{path}: expected ({bands}, W, W) or "
                         f"(B, {bands}, W, W), got {a.shape}")
    if a.shape[2] != window or a.shape[3] != window:
        raise SystemExit(f"{path}: windows are {a.shape[2]}x{a.shape[3]}, "
                         f"model window is {window}")
    return a


def codec_patch_main(argv=None):
    ap = argparse.ArgumentParser(
        description="Patch-wise AlphaEarth codec: encode/decode pre-cut "
                    "(B, 64, W, W) windows to/from .gvq containers.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="windows .npy (int8 or float32) -> .gvq")
    _common_args(e)
    e.add_argument("input", help=".npy windows: int8 raw AE or float32 "
                                 "already unit-norm")
    e.add_argument("output", help="output .gvq container")
    e.add_argument("--no-coarse-residual", action="store_true",
                   help="omit the 8* sidecar (2-bit coarse residual, "
                        "66 B/window; recovers the coarse quantizer gap "
                        "and most of the OOD error tail)")
    e.add_argument("--no-stats", action="store_true",
                   help="skip encode-time quality stats (per-ladder-point "
                        "dcos summaries in the header + .stats.json)")
    e.add_argument("--batch", type=int, default=64)

    d = sub.add_parser("decode", help=".gvq -> reconstructed windows .npy")
    _common_args(d)
    d.add_argument("input", help=".gvq container")
    d.add_argument("output", help="output .npy")
    d.add_argument("--tuple", default=None,
                   help="rate-control decode: depth tuple, e.g. 8,8,2 "
                        "(default: full)")
    _quality_arg(d)
    d.add_argument("--int8", action="store_true",
                   help="re-quantize output to AE int8 (signed-sqrt)")
    d.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)

    be = load_backend(args)
    t0 = time.time()
    if args.cmd == "encode":
        a = _load_windows(args.input, be.window, be.bands)
        if a.dtype == np.int8:
            valid = a[:, 0] != NODATA      # prep itself runs on-device
            check_fill(a, valid)
            arr = a
        else:
            arr = a.astype(np.float32)
            # float input: invalid pixels are the (near-)zero vectors the
            # prep convention produces for nodata
            valid = np.linalg.norm(arr, axis=1) > 0.5
        keep = valid.any((1, 2))     # fully-nodata windows are not encoded
        star = not args.no_coarse_residual
        packets, qs = encode_all(be, arr[keep], valid[keep], args.batch,
                                 star, not args.no_stats)
        hdr = {"kind": "patch", "window": be.window, "n_total": len(arr),
               "bitstream_id": be.bitstream_id, "nodata_fill": NODATA}
        if qs is not None:
            hdr["quality_stats"] = qs.summary(
                be.specs_sig, be.star_cost_bits if star else 0)
            with open(args.output + ".stats.json", "w") as f:
                json.dump(hdr["quality_stats"], f, indent=1)
        header = write_container(args.output, packets, be.specs_sig, hdr,
                                 valid=valid)
        skipped = int((~keep).sum())
        _report(f"encoded{f' ({skipped} empty windows skipped)' if skipped
                          else ''}",
                header, max(len(packets), 1), be.window,
                os.path.getsize(args.output), t0)
    else:
        if args.quality and args.tuple:
            raise SystemExit("--tuple (rate-control) and --quality "
                             "(quality-control) are mutually exclusive")
        if args.quality:
            depths = _resolve_quality(_read_header(args.input),
                                      args.quality, be.specs_sig)
        else:
            depths = _parse_tuple(args.tuple, be.specs_sig)
        header, packets, valid = read_container(args.input, depths=depths)
        _check_bitstream(header, be, args.input)
        depths = depths or [d for _, d, _ in be.specs_sig]
        n_total = header.get("n_total", header["n_windows"])
        dec = np.concatenate(
            [be.decode(packets[i:i + args.batch], depths)
             for i in range(0, len(packets), args.batch)]) if packets             else np.zeros((0, be.bands, be.window, be.window), np.float32)
        out = np.zeros((n_total, *dec.shape[1:]), np.float32)
        keep = valid.any((1, 2)) if valid is not None             else np.ones(n_total, bool)
        out[keep] = dec
        fill = header.get("nodata_fill", NODATA)
        if args.int8:
            out = requant_ae_np(out, valid, fill)
        elif valid is not None:      # model-space convention: zero vectors
            out = out * valid[:, None]
        np.save(args.output, out)
        rd, tot = container_read_bytes(args.input, be.specs_sig, depths)
        _report(f"decoded (tuple={depths}, read {rd/2**20:.2f} of "
                f"{tot/2**20:.2f} MiB)", header, len(packets),
                be.window, None, t0, depths)


# --------------------------------------------------------------------------
# georvq-codec-tile
# --------------------------------------------------------------------------

_PROFILE_KEYS = ("driver", "nodata", "blockxsize", "blockysize", "tiled",
                 "compress", "interleave")


def _read_tile(path):
    """-> (int8 (C, H, W) NORTH-UP, geo dict or None).

    GeoTIFFs keep their full creation profile so decode can reproduce the
    source file type exactly (the AEF source.coop tiles: GTiff, 1024^2
    blocks, zstd, band-interleaved, bottom-up). Bottom-up rasters
    (transform.e > 0) are flipped to north-up here — the orientation the
    model was trained on — and flipped back on write."""
    if path.endswith((".tif", ".tiff")):
        try:
            import rasterio
        except ImportError as e:
            raise SystemExit("GeoTIFF input needs rasterio: "
                             "pip install 'georvq[eval]'") from e
        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), \
                rasterio.open(path) as ds:
            a = ds.read()      # threaded zstd/deflate: ~10x faster inflate
            prof = {k: v for k, v in ds.profile.items()
                    if k in _PROFILE_KEYS}
            geo = {"transform": list(ds.transform)[:6],
                   "crs": str(ds.crs) if ds.crs else None,
                   "profile": prof,
                   "flipped": bool(ds.transform.e > 0)}
        a = a.astype(np.int8)
        if geo["flipped"]:
            a = a[:, ::-1, :]
        return np.ascontiguousarray(a), geo
    return np.load(path).astype(np.int8), None


def _write_tile(path, a, geo, int8, fill=NODATA):
    """Write north-up array back out; GeoTIFF output restores the source's
    exact profile (orientation, blocking, compression, CRS, nodata tag)."""
    if geo and geo.get("flipped"):
        a = np.ascontiguousarray(a[:, ::-1, :])
    if path.endswith((".tif", ".tiff")):
        try:
            import rasterio
            from rasterio.transform import Affine
        except ImportError as e:
            raise SystemExit("GeoTIFF output needs rasterio") from e
        prof = dict((geo or {}).get("profile")
                    or {"driver": "GTiff", "compress": "deflate",
                        "tiled": True})
        prof.update(height=a.shape[1], width=a.shape[2],
                    count=a.shape[0], dtype=str(a.dtype))
        if int8:
            prof["nodata"] = fill
        else:
            prof.pop("nodata", None)     # float output: no int fill value
        if geo:
            prof["transform"] = Affine(*geo["transform"])
            if geo["crs"]:
                prof["crs"] = geo["crs"]
        opts = {"num_threads": "all_cpus"}
        if prof.get("compress") == "zstd":
            opts["zstd_level"] = 1       # same format, ~5x faster writes
        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), \
                rasterio.open(path, "w", **prof, **opts) as ds:
            ds.write(a)
    else:
        np.save(path, a)


def codec_tile_main(argv=None):
    ap = argparse.ArgumentParser(
        description="Tile-wise AlphaEarth codec: a full raster (GeoTIFF or "
                    ".npy, (64, H, W) int8) is encoded into a directory of "
                    "georeferenced COG assets (per-tier index rasters, "
                    "1024px-aligned blocks) and stitched back on decode.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="raster -> COG asset directory")
    _common_args(e)
    e.add_argument("input", help="GeoTIFF or .npy, (64, H, W) int8")
    e.add_argument("output", help="output asset directory")
    e.add_argument("--no-coarse-residual", action="store_true",
                   help="omit the 8* sidecar (2-bit coarse residual, "
                        "66 B/window; recovers the coarse quantizer gap "
                        "and most of the OOD error tail)")
    e.add_argument("--no-stats", action="store_true",
                   help="skip encode-time quality stats (per-ladder-point "
                        "dcos summaries in the header + .stats.json)")
    e.add_argument("--batch", type=int, default=64)

    d = sub.add_parser("decode", help="COG asset directory -> raster")
    _common_args(d)
    d.add_argument("input", help="asset directory (from encode)")
    d.add_argument("output", help=".npy or .tif")
    d.add_argument("--tuple", default=None,
                   help="rate-control decode: depth tuple (default: full)")
    _quality_arg(d)

    i = sub.add_parser("info", help="print a tile's metadata + quality "
                                    "stats")
    i.add_argument("input", help="asset directory")
    d.add_argument("--float32", action="store_true",
                   help="write float32 unit-norm instead of AE int8")
    d.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)

    if args.cmd == "info":
        from georvq.ae.cog import read_meta
        header = read_meta(args.input)
        stats = header.pop("quality_stats", None)
        print(json.dumps(header, indent=1))
        if stats:
            print(f"{'point':>8s} {'bits/win':>9s} {'mean':>9s} {'p50':>9s} "
                  f"{'p90':>9s} {'p99':>9s} {'max':>9s}")
            for k, v in stats.items():
                print(f"{k + ('*' if v.get('star') else ''):>8s} "
                      f"{v['bits']:9.0f} " + " ".join(
                          f"{v[m]:9.2e}" for m in
                          ("mean", "p50", "p90", "p99", "max")))
        return

    be = load_backend(args)
    W = be.window
    t0 = time.time()

    if args.cmd == "encode":
        a, geo = _read_tile(args.input)
        C, H, Wd = a.shape
        gh, gw = -(-H // W), -(-Wd // W)
        pad = np.full((C, gh * W, gw * W), NODATA, np.int8)
        pad[:, :H, :Wd] = a
        wins = pad.reshape(C, gh, W, gw, W).transpose(1, 3, 0, 2, 4) \
            .reshape(gh * gw, C, W, W)
        check_fill(a[None], (a[0] != NODATA)[None])
        keep = (wins[:, 0] != NODATA).any((1, 2))   # skip fully-nodata wins
        kept = np.nonzero(keep)[0]
        star = not args.no_coarse_residual
        wk = wins[kept]              # int8; prep runs on-device per batch
        packets, qs = encode_all(be, wk, wk[:, 0] != NODATA, args.batch,
                                 star, not args.no_stats)
        if geo is None:              # .npy input: identity georeferencing
            geo = {"transform": [1, 0, 0, 0, 1, 0], "crs": None,
                   "profile": None, "flipped": False}
        geo = {**geo, "src_height": H}
        meta = {"kind": "tile", "window": W, "grid": [gh, gw],
                "shape": [C, H, Wd], "geo": geo,
                "bitstream_id": be.bitstream_id, "nodata_fill": NODATA,
                "specs_sig": [list(x) for x in be.specs_sig],
                "star": {"bits": STAR_BITS, "sigma": STAR_SIGMA}
                if star else None}
        if qs is not None:
            meta["quality_stats"] = qs.summary(
                be.specs_sig, be.star_cost_bits if star else 0)
        star_arrays = None
        if star:
            res = np.stack([p.coarse_residual for p in packets])
            star_arrays = pack_star(res)
        quality = None
        if qs is not None and be.decoder_id:
            quality = (be.decoder_id, qs.ladder, qs.window_means())
        from georvq.ae.cog import write_assets
        sizes = write_assets(args.output, packets, kept, meta,
                             a[0] != NODATA, geo, be.specs_sig,
                             star=star_arrays, quality=quality)
        skipped = int((~keep).sum())
        _report(f"encoded {H}x{Wd} ({gh}x{gw} windows"
                f"{f', {skipped} empty skipped' if skipped else ''})",
                meta, max(len(packets), 1), W, sum(sizes.values()), t0)
    else:
        if args.quality and args.tuple:
            raise SystemExit("--tuple (rate-control) and --quality "
                             "(quality-control) are mutually exclusive")
        from georvq.ae.cog import read_assets, read_meta
        if args.quality:
            depths = _resolve_quality(read_meta(args.input),
                                      args.quality, be.specs_sig)
        else:
            depths = _parse_tuple(args.tuple, be.specs_sig)
        meta, packets, kept, valid = read_assets(
            args.input, depths=depths, expect_bitstream=be.bitstream_id)
        depths = depths or [d for _, d, _ in be.specs_sig]
        C, H, Wd = meta["shape"]
        gh, gw = meta["grid"]
        fill = meta.get("nodata_fill", NODATA)
        # stitch straight into one canvas, re-quantizing per batch, so a
        # full 8192^2 source tile never needs an all-float intermediate
        canvas = (np.zeros((C, gh * W, gw * W), np.float32) if args.float32
                  else np.full((C, gh * W, gw * W), fill, np.int8))
        for i in range(0, len(packets), args.batch):
            dec = be.decode(packets[i:i + args.batch], depths,
                            int8=not args.float32)
            for w, slot in zip(dec, kept[i:i + args.batch]):
                r, c = divmod(int(slot), gw)
                canvas[:, r * W:(r + 1) * W, c * W:(c + 1) * W] = w
        a = canvas[:, :H, :Wd]
        if args.float32:             # restore the source's exact fill
            a = a * valid[None]
        else:
            a[np.broadcast_to(~valid[None], a.shape)] = fill
        _write_tile(args.output, a, meta.get("geo"),
                    int8=not args.float32, fill=fill)
        from georvq.ae.cog import tier_names
        rd = sum(os.path.getsize(os.path.join(args.input, f"{n}.tif"))
                 * depths[i] / be.specs_sig[i][1]
                 for i, n in enumerate(tier_names(len(be.specs_sig)))
                 if depths[i] > 0)
        tot = sum(os.path.getsize(os.path.join(args.input, f))
                  for f in os.listdir(args.input))
        _report(f"decoded {H}x{Wd} (tuple={depths}, ~{rd/2**20:.2f} of "
                f"{tot/2**20:.2f} MiB)", meta,
                len(packets), W, None, t0, depths)


if __name__ == "__main__":   # python -m georvq.ae.cli <tool> ...
    tools = {"export-onnx": export_onnx_main, "patch": codec_patch_main,
             "tile": codec_tile_main}
    if len(sys.argv) < 2 or sys.argv[1] not in tools:
        raise SystemExit(f"usage: python -m georvq.ae.cli "
                         f"{{{','.join(tools)}}} ...")
    tools[sys.argv.pop(1)]()
