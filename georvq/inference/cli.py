"""Deployment executables for the AlphaEarth codec.

Three console entry points (see pyproject):

  georvq-export-onnx   checkpoint/bundle -> ONNX export dir (+ parity check)
  georvq-codec-patch   encode/decode pre-cut (B, 64, W, W) windows
  georvq-codec-tile    encode/decode a full raster: tile -> windows ->
                       one self-contained archive file -> stitch back

Both codecs share `--backend {torch,onnx}` and `--device {cpu,cuda}`:
torch runs the checkpoint (CPU or CUDA), onnx runs an exported dir through
onnxruntime with no torch dependency at all. All backends read and write
the same container format (`.gvq`, an npz), so a file encoded on GPU
decodes on a CPU-only box and vice versa.

Container: npz with a JSON header plus either raw per-tier index arrays or
rANS-coded streams with their embedded CDF tables (`--raw` to skip entropy
coding; coding needs compressai). Indices are stored full-depth; any
monotone depth tuple can be chosen at decode time (`--tuple 6,4,4`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

NODATA = -128
FORMAT = "georvq-ae/1"


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


def requant_ae_np(x, valid=None):
    """float32 field -> int8 signed-sqrt quantization (inverse of dequant);
    invalid pixels -> NODATA."""
    s = np.rint(np.sign(x) * np.sqrt(np.abs(x)) * 127.5)
    s = np.clip(s, -127, 127).astype(np.int8)
    if valid is not None:
        s[np.broadcast_to(~valid[:, None], s.shape)] = NODATA
    return s


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

class TorchBackend:
    """numpy-in / numpy-out adapter over the torch Codec."""

    def __init__(self, model_path, device="cpu"):
        import torch
        self.torch = torch
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

    def encode(self, x):
        xt = self.torch.from_numpy(np.ascontiguousarray(x, np.float32)) \
            .to(self.device)
        return self.codec.encode(xt)

    def decode(self, packets, depths=None):
        return self.codec.decode(packets, depths).float().cpu().numpy()


class OnnxBackend:
    def __init__(self, export_dir, device="cpu", threads=None):
        from georvq.inference.onnx_backend import OnnxCodec
        self.codec = OnnxCodec(export_dir, device=device, threads=threads)
        self.window = self.codec.window
        self.bands = self.codec.bands
        self.specs_sig = [(t["grid"] ** 2, t["depth"], t["codes"])
                          for t in self.codec.tiers]

    def encode(self, x):
        return self.codec.encode(x)

    def decode(self, packets, depths=None):
        return self.codec.decode(packets, depths)


def load_backend(args):
    kind = args.backend
    if kind == "auto":
        kind = "onnx" if (os.path.isdir(args.model) and os.path.exists(
            os.path.join(args.model, "encoder.onnx"))) else "torch"
    if kind == "onnx":
        return OnnxBackend(args.model, device=args.device,
                           threads=getattr(args, "threads", None))
    return TorchBackend(args.model, device=args.device)


# --------------------------------------------------------------------------
# container (.gvq = npz)
# --------------------------------------------------------------------------

def _index_dtype(codes):
    return np.uint16 if codes <= 2 ** 16 else np.uint32


def write_container(path, packets, specs_sig, header, raw=False):
    """Write N packets (+ header dict) as one npz. rANS-coded with embedded
    CDFs unless raw (or compressai is missing, with a warning)."""
    from georvq.inference.api import Packet  # noqa: F401  (type only)
    stacked = [np.stack([np.asarray(p.indices[t]) for p in packets])
               for t in range(len(specs_sig))]
    payload = {}
    if not raw:
        try:
            from georvq.inference.archive import CdfTable, encode_packet
            table = CdfTable.calibrate(stacked, _SpecsView(specs_sig))
            streams = [encode_packet(p, table) for p in packets]
            payload = {
                "cdf_cdfs": np.array(
                    [np.pad(c, (0, max(table.lengths) - len(c)))
                     for c in table.cdfs], dtype=np.int64),
                "cdf_lengths": np.array(table.lengths),
                "cdf_offsets": np.array(table.offsets),
                "n_bytes": np.array([len(s) for s in streams]),
                "payload": np.frombuffer(b"".join(streams), np.uint8),
            }
            header = {**header, "coded": True}
        except ImportError:
            print("warning: compressai not installed - writing raw "
                  "(uncoded) indices; pip install 'georvq[archive]'",
                  file=sys.stderr)
            raw = True
    if raw:
        payload = {f"idx_t{t}": arr.astype(_index_dtype(specs_sig[t][2]))
                   for t, arr in enumerate(stacked)}
        header = {**header, "coded": False}
    header = {**header, "format": FORMAT, "n_windows": len(packets),
              "specs_sig": [list(s) for s in specs_sig]}
    np.savez_compressed(path, header=json.dumps(header), **payload)
    return header


def read_container(path):
    """-> (header dict, list[Packet])."""
    from georvq.inference.api import Packet
    z = np.load(path, allow_pickle=False)
    header = json.loads(str(z["header"]))
    if header.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} container")
    sig = [tuple(s) for s in header["specs_sig"]]
    n = header["n_windows"]
    if header["coded"]:
        from georvq.inference.archive import CdfTable, decode_packet
        lengths = z["cdf_lengths"].tolist()
        table = CdfTable(
            [z["cdf_cdfs"][i][:ln].tolist() for i, ln in enumerate(lengths)],
            lengths, z["cdf_offsets"].tolist(), sig)
        raw = z["payload"].tobytes()
        offs = np.concatenate([[0], np.cumsum(z["n_bytes"])])
        packets = [Packet(indices=decode_packet(raw[offs[i]:offs[i + 1]],
                                                table))
                   for i in range(n)]
    else:
        stacked = [z[f"idx_t{t}"].astype(np.int64) for t in range(len(sig))]
        packets = [Packet(indices=[a[i] for a in stacked]) for i in range(n)]
    return header, packets


class _SpecsView:
    """Adapt (cells, depth, codes) tuples to the spec surface CdfTable uses."""

    def __init__(self, sig):
        self._sig = sig

    def __iter__(self):
        from types import SimpleNamespace
        return iter(SimpleNamespace(cells=c, depth=d, codes=k)
                    for c, d, k in self._sig)


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
    if file_bytes:  # encode side: the realized file, CDF tables included
        fb = 8 * file_bytes / n_windows
        line += (f" | file {fb:.0f} bits/window ({fb / window ** 2:.2f} bpp"
                 f", CDF tables amortized over {n_windows})")
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
    e.add_argument("--raw", action="store_true",
                   help="skip rANS entropy coding (indices stored uncoded)")
    e.add_argument("--batch", type=int, default=64)

    d = sub.add_parser("decode", help=".gvq -> reconstructed windows .npy")
    _common_args(d)
    d.add_argument("input", help=".gvq container")
    d.add_argument("output", help="output .npy")
    d.add_argument("--tuple", default=None,
                   help="decode depth tuple, e.g. 6,4,4 (default: full)")
    d.add_argument("--int8", action="store_true",
                   help="re-quantize output to AE int8 (signed-sqrt)")
    d.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)

    be = load_backend(args)
    t0 = time.time()
    if args.cmd == "encode":
        a = _load_windows(args.input, be.window, be.bands)
        if a.dtype == np.int8:
            x, _ = prep_ae_np(a)
        else:
            x = a.astype(np.float32)
        packets = _batched(be.encode, x, args.batch)
        header = write_container(args.output, packets, be.specs_sig,
                                 {"kind": "patch", "window": be.window},
                                 raw=args.raw)
        _report("encoded", header, len(packets), be.window,
                os.path.getsize(args.output), t0)
    else:
        header, packets = read_container(args.input)
        depths = _parse_tuple(args.tuple, be.specs_sig)
        out = np.concatenate(
            [be.decode(packets[i:i + args.batch], depths)
             for i in range(0, len(packets), args.batch)])
        if args.int8:
            out = requant_ae_np(out)
        np.save(args.output, out)
        _report(f"decoded (tuple={depths or 'full'})", header, len(packets),
                be.window, None, t0)


# --------------------------------------------------------------------------
# georvq-codec-tile
# --------------------------------------------------------------------------

def _read_tile(path):
    """-> (int8 (C, H, W), geo dict or None)."""
    if path.endswith((".tif", ".tiff")):
        try:
            import rasterio
        except ImportError as e:
            raise SystemExit("GeoTIFF input needs rasterio: "
                             "pip install 'georvq[eval]'") from e
        with rasterio.open(path) as ds:
            a = ds.read()
            geo = {"transform": list(ds.transform)[:6],
                   "crs": str(ds.crs) if ds.crs else None}
        return a.astype(np.int8), geo
    return np.load(path).astype(np.int8), None


def _write_tile(path, a, geo, int8):
    if path.endswith((".tif", ".tiff")):
        try:
            import rasterio
            from rasterio.transform import Affine
        except ImportError as e:
            raise SystemExit("GeoTIFF output needs rasterio") from e
        prof = {"driver": "GTiff", "height": a.shape[1], "width": a.shape[2],
                "count": a.shape[0], "dtype": str(a.dtype),
                "compress": "deflate", "tiled": True}
        if int8:
            prof["nodata"] = NODATA
        if geo:
            prof["transform"] = Affine(*geo["transform"])
            if geo["crs"]:
                prof["crs"] = geo["crs"]
        with rasterio.open(path, "w", **prof) as ds:
            ds.write(a)
    else:
        np.save(path, a)


def codec_tile_main(argv=None):
    ap = argparse.ArgumentParser(
        description="Tile-wise AlphaEarth codec: a full raster (GeoTIFF or "
                    ".npy, (64, H, W) int8) is tiled into windows, encoded "
                    "into one .gvq archive, and stitched back on decode.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="raster -> .gvq")
    _common_args(e)
    e.add_argument("input", help="GeoTIFF or .npy, (64, H, W) int8")
    e.add_argument("output")
    e.add_argument("--raw", action="store_true")
    e.add_argument("--batch", type=int, default=64)

    d = sub.add_parser("decode", help=".gvq -> raster")
    _common_args(d)
    d.add_argument("input")
    d.add_argument("output", help=".npy or .tif")
    d.add_argument("--tuple", default=None)
    d.add_argument("--float32", action="store_true",
                   help="write float32 unit-norm instead of AE int8")
    d.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)

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
        x, valid = prep_ae_np(wins)
        packets = _batched(be.encode, x, args.batch)
        header = {"kind": "tile", "window": W, "grid": [gh, gw],
                  "shape": [C, H, Wd], "geo": geo}
        if not valid.all():
            full = np.zeros((gh * W, gw * W), bool)
            full[:H, :Wd] = (a[0] != NODATA)
            header["valid_zlib"] = zlib.compress(
                np.packbits(full[:H, :Wd]).tobytes(), 9).hex()
        header = write_container(args.output, packets, be.specs_sig,
                                 header, raw=args.raw)
        _report(f"encoded {H}x{Wd} ({gh}x{gw} windows)", header,
                len(packets), W, os.path.getsize(args.output), t0)
    else:
        header, packets = read_container(args.input)
        if header["kind"] != "tile":
            raise SystemExit(f"{args.input} is a {header['kind']} container;"
                             " use georvq-codec-patch")
        depths = _parse_tuple(args.tuple, be.specs_sig)
        C, H, Wd = header["shape"]
        gh, gw = header["grid"]
        out = np.concatenate(
            [be.decode(packets[i:i + args.batch], depths)
             for i in range(0, len(packets), args.batch)])
        stitched = out.reshape(gh, gw, C, W, W).transpose(2, 0, 3, 1, 4) \
            .reshape(C, gh * W, gw * W)[:, :H, :Wd]
        valid = np.ones((1, H, Wd), bool)
        if "valid_zlib" in header:
            bits = np.frombuffer(
                zlib.decompress(bytes.fromhex(header["valid_zlib"])),
                np.uint8)
            valid = np.unpackbits(bits, count=H * Wd) \
                .reshape(1, H, Wd).astype(bool)
        if args.float32:
            a = (stitched * valid).astype(np.float32)
        else:
            a = requant_ae_np(stitched[None], valid)[0]
        _write_tile(args.output, a, header.get("geo"),
                    int8=not args.float32)
        _report(f"decoded {H}x{Wd} (tuple={depths or 'full'})", header,
                len(packets), W, None, t0, depths)


if __name__ == "__main__":   # python -m georvq.inference.cli <tool> ...
    tools = {"export-onnx": export_onnx_main, "patch": codec_patch_main,
             "tile": codec_tile_main}
    if len(sys.argv) < 2 or sys.argv[1] not in tools:
        raise SystemExit(f"usage: python -m georvq.inference.cli "
                         f"{{{','.join(tools)}}} ...")
    tools[sys.argv.pop(1)]()
