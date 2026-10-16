"""AlphaEarth v2 release layer: annual tiles as georvq GeoParquet files.

An AE v2 file holds one AlphaEarth annual tile (64-band int8, nodata
-128) coded by a `stats_cond` model whose per-window mean travels through
an `RVQPCAMeanChannel`. The layout is the "AE v2 instance" of
docs/GEORVQ_PARQUET.md: one row and one row group per chunk, with a
chunk size per kind, in this file order:

- `base`, one 8192 px chunk (the whole tile): `base`, per window the
  mean channel's fields and every level of the coarsest tier, then
  `mask`, the tile's pixel validity (north-up), empty when fully valid;
- `mid`, 4096 px chunks: `mid_1..mid_8`, per level the chunk's windows'
  codes, so a depth prefix of a chunk is one contiguous byte range;
- `fine`, 1024 px chunks: `fine_1..fine_8`;
- `quality`, optional, chunked as base: per window and per rung, the
  mean and p99 of per-pixel `1 - cos` measured at encode time.

A codec with other than three tiers names them `tier0` and up, the
finest taking the fine chunk size and the others the mid one.

The source's profile, tags and orientation are JSON in the file's
`georvq_aux` metadata. Coded data is north-up. A bottom-up source
(transform.e > 0, as the AEF tiles are stored) is flipped on encode, and
`AEV2Dataset.read` returns rows in the source's own order, so it can
stand in for the rasterio dataset of the source tile.

Both ends condition on the coded mean: the encoder is run against
`channel.round_trip(exact mean)`, and the decoder reads the same vector
back from the base record. Windows are coded independently, so any
window can be decoded from its own records.

Reading can also run without torch: `export_onnx` writes the model's
graphs, codebooks and mean channel to a directory, and
`open_dataset(path, backend="onnx", onnx_dir=...)` decodes through
`AEV2OnnxCodec` with onnxruntime and numpy. This module imports torch
only when the torch codec is used.

The header names the release a file was coded with (`codec.release`),
so `open_dataset(path)` with no codec arguments finds the bundle that
decodes it, in the environment, the local cache, or by downloading it
from a trusted release (georvq.ae.v2_resolve). The bundle is located
when pixels are first read, so reading metadata needs none.

"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np
from rasterio.transform import TransformMethodsMixin
from rasterio.windows import WindowMethodsMixin

from georvq.ae.v2_resolve import (RELEASE_NAME, RELEASE_URI, BundleError,
                                  default_device, release_record,
                                  resolve_codec)
from georvq.inference.georvq_parquet import (Layout, RVQParquetReader,
                                             RVQParquetWriter, output_name,
                                             span, tier_names)
from georvq.inference.georvq_zarr import ZarrReader, open_reader
from georvq.inference.mean_codec import RVQPCAMeanChannel

NODATA = -128
KIND = "aef-annual-tile"
FORMAT = "georvq-ae/v2"
BASE_CHUNK_PX, MID_CHUNK_PX, FINE_CHUNK_PX = 8192, 4096, 1024
STRIP_PX = 1024
QUALITY_STATS = ("mean", "p99")
# quality code = round(Q_SCALE * (log10(e) + Q_OFFSET)), clipped to 0..255
Q_SCALE, Q_OFFSET, Q_FLOOR = 32.0, 6.0, 1e-6

_GDAL_ENV = {"GDAL_NUM_THREADS": "ALL_CPUS"}


@contextlib.contextmanager
def _quiet_build():
    """Silence the UserWarning torch's TransformerEncoder gives for the
    tokenizer's norm_first layers, on every model build."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="enable_nested_tensor",
                                category=UserWarning)
        yield


def _no_grad(fn):
    """torch.no_grad, imported on call: reading with the ONNX backend
    (AEV2OnnxCodec) does not import torch."""
    @functools.wraps(fn)
    def run(*a, **kw):
        import torch
        with torch.no_grad():
            return fn(*a, **kw)
    return run


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

def _index_bits(codes: int) -> int:
    b = int(codes).bit_length() - 1
    if codes != 1 << b:
        raise ValueError(f"codebook size {codes} is not a power of two")
    return b


def mean_fields(channel) -> list:
    """The mean channel's record fields: the RVQ indices, then runs of
    PCA components sharing a bit width, named by their first component."""
    L = int(channel.books.shape[0])
    out = [("mean_rvq", L, channel.index_bits)]
    bits = [int(b) for b in channel.bits]
    k = 0
    while k < len(bits):
        j = k
        while j < len(bits) and bits[j] == bits[k]:
            j += 1
        out.append((f"mean_pca_{k}", j - k, bits[k]))
        k = j
    return out


def chunk_sizes(n_tiers: int) -> dict:
    """Chunk side in pixels per kind of an AE v2 file."""
    names = tier_names(n_tiers)
    out = {"base": BASE_CHUNK_PX}
    out.update({n: MID_CHUNK_PX for n in names[1:-1]})
    if n_tiers > 1:
        out[names[-1]] = FINE_CHUNK_PX
    return out


def layout(specs, window: int, channel, n_rungs: int = 0,
           chunk_px=None) -> Layout:
    """Columns of an AE v2 file. `n_rungs` 0 leaves out the quality
    column. The coarsest tier's levels follow the mean in `base`.
    chunk_px: per-kind chunk sizes, `chunk_sizes` by default."""
    names = tier_names(len(specs))
    sp0 = specs[0]
    base = mean_fields(channel) + [
        ("coarse", sp0.cells * sp0.depth, _index_bits(sp0.codes))]
    tiers = [(names[0], sp0.depth, None)] + [
        (name, sp.depth, [(name, sp.cells, _index_bits(sp.codes))])
        for name, sp in zip(names[1:], specs[1:])]
    quality = [(st, n_rungs, 8) for st in QUALITY_STATS] if n_rungs else None
    return Layout(window, chunk_px or chunk_sizes(len(specs)), base, tiers,
                  quality)


def rung_bits(specs, depths, channel) -> int:
    """Bits per window at a depth tuple, the mean record included."""
    return int(channel.cost_bits()) + sum(
        sp.cells * int(L) * _index_bits(sp.codes) for sp, L in zip(specs, depths))


def quality_code(e) -> np.ndarray:
    """Error e = 1 - cos -> uint8 code; 0 means e <= 1e-6. NaN (a window
    with no valid pixels) codes as 0."""
    e = np.asarray(e, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.rint(Q_SCALE * (np.log10(np.maximum(e, 1e-30)) + Q_OFFSET))
    c = np.clip(np.nan_to_num(c, nan=0.0), 0, 255)
    c[~(e > Q_FLOOR)] = 0
    return c.astype(np.uint8)


def quality_value(code) -> np.ndarray:
    """The error a code stands for (1e-6 for code 0)."""
    return 10.0 ** (np.asarray(code, dtype=np.float64) / Q_SCALE - Q_OFFSET)


def parse_rung(q, depths_max=None) -> tuple:
    """"8-8-2", "8,8,2" or a sequence -> tuple of ints."""
    if isinstance(q, str):
        t = tuple(int(v) for v in re.split(r"[-,]", q.strip()))
    else:
        t = tuple(int(v) for v in q)
    if depths_max is not None and (len(t) != len(depths_max) or any(
            not 0 <= a <= b for a, b in zip(t, depths_max))):
        raise ValueError(f"depth tuple {t} does not fit tiers {depths_max}")
    return t


_BOUND = re.compile(r"^\s*(\w+)\s*<=\s*([0-9.eE+-]+)\s*$")


def _flip_rows(t, height):
    """Affine of the same raster with its rows in the other order."""
    from affine import Affine
    return Affine(t.a, -t.b, t.c + t.b * height, t.d, -t.e, t.f + t.e * height)


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------

class BlockRecords:
    """Records of the windows of an encoded pixel region, in the
    region's own window coordinates.

    rows, cols: (n,) positions of the coded windows; records: {group:
    (levels, n, n_values)} for the groups `base`, each finer tier's name
    and `quality`; valid: the region's pixel validity (unpadded); grid:
    the region's window grid. Windows with no valid pixel are left out
    and their records are zero."""

    def __init__(self, rows, cols, records, valid, grid):
        self.rows, self.cols = rows, cols
        self.records, self.valid, self.grid = records, valid, grid

    def dense(self) -> dict:
        """{group: (rows, cols, n_values)} for base and quality, and
        (rows, cols, levels, n_values) for a tier."""
        gh, gw = self.grid
        out = {}
        for name, arr in self.records.items():
            L, _, nv = arr.shape
            d = np.zeros((gh, gw, L, nv), np.int64)
            d[self.rows, self.cols] = arr.transpose(1, 0, 2)
            out[name] = d[:, :, 0] if name in ("base", "quality") else d
        return out

    def add_to(self, writer, window_row=0, window_col=0):
        """Add the region's windows to an RVQParquetWriter, its top-left
        window at (window_row, window_col) of the product's grid."""
        writer.add_windows(window_row, window_col, self.dense(), self.valid)


class AEV2Codec:
    """A stats_cond AE model with its mean channel, as used by the v2
    files. Encoder passes run in fp32; decoder passes run in bf16 on
    CUDA unless `bf16=False` (on CPU always fp32)."""

    def __init__(self, model, channel, bf16=None, rungs=None):
        from georvq.inference.api import Codec
        from georvq.inference.release import bitstream_id, decoder_id
        from georvq.model.codec import path_tuples
        self.model = model.eval()
        self.channel = channel
        self.device = next(model.parameters()).device
        sc = model.stats_cond
        if sc is None or not sc.subtract_mean or sc.use_std:
            raise ValueError("AE v2 needs a stats_cond model with a "
                             "transmitted mean and no std")
        if channel.bands != model.bands:
            raise ValueError(f"mean channel has {channel.bands} bands, model "
                             f"{model.bands}")
        self.codec = Codec(model)
        self.specs = list(model.specs)
        self.window = int(model.window)
        self.bands = int(model.bands)
        self.full = tuple(sp.depth for sp in self.specs)
        self.rungs = [tuple(r) for r in (rungs or path_tuples(
            self.specs, model.cfg.min_depths))]
        cuda = self.device.type == "cuda"
        self.bf16 = cuda if bf16 is None else (bool(bf16) and cuda)
        self.names = tier_names(len(self.specs))
        self.n_mean = len(channel.field_widths())
        self.bitstream_id = bitstream_id(model, extra=channel.id_state())
        self.decoder_id = decoder_id(model)

    @classmethod
    def load(cls, model, channel, device="cpu", bf16=None):
        """model: a checkpoint path or a model; channel: a .npz path or a
        channel."""
        if isinstance(model, (str, os.PathLike)):
            from georvq.factory import from_checkpoint
            model = from_checkpoint(os.fspath(model), device)
        if isinstance(channel, (str, os.PathLike)):
            channel = RVQPCAMeanChannel.load(os.fspath(channel))
        return cls(model, channel, bf16=bf16)

    def _ac(self):
        import torch
        return (torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16
                else contextlib.nullcontext())

    def layout(self, quality=True, chunk_px=None) -> Layout:
        return layout(self.specs, self.window, self.channel,
                      len(self.rungs) if quality else 0, chunk_px)

    def header_codec(self, release=None) -> dict:
        return {"format": FORMAT, "bitstream_id": self.bitstream_id,
                "decoder_id": self.decoder_id,
                "tiers": [str(sp) for sp in self.specs],
                "window": self.window, "bands": self.bands,
                "min_depths": list(self.model.cfg.min_depths),
                "mean_channel": {"kind": "rvq-pca",
                                 "bits": int(self.channel.cost_bits())},
                "release": release}

    def header_rungs(self) -> list:
        W2 = self.window ** 2
        return [{"tuple": list(r),
                 "bits_window": rung_bits(self.specs, r, self.channel),
                 "bits_px": rung_bits(self.specs, r, self.channel) / W2}
                for r in self.rungs]

    def header_quality(self) -> dict:
        return {"column": "quality", "stats": list(QUALITY_STATS),
                "error": "1 - cos(decoded, source) per valid pixel, "
                         "decoded float field against the prepared source",
                "encoding": "code = round(32 * (log10(e) + 6)) clipped to "
                            "0..255; 0 means e <= 1e-6",
                "scale": Q_SCALE, "offset": Q_OFFSET,
                "order": "per stat, one byte per rung in header rungs order",
                "decode_precision": "bf16" if self.bf16 else "fp32"}

    # ---- encode ----

    @_no_grad
    def encode_windows(self, raw, stats=True):
        """raw: (B, C, W, W) int8 north-up windows. Returns ({group:
        (levels, B, n_values)}, x, valid) with x, valid the prepared
        device input."""
        import torch
        dev = self.device
        x, valid = self.model.source.prep(
            torch.from_numpy(np.ascontiguousarray(raw)), dev)
        mu = self.model.patch_stats(x, valid).float().cpu().numpy()
        fields = self.channel.encode(mu)
        coded = torch.from_numpy(self.channel.decode(fields)).to(dev)
        stacked, _, _ = self.codec.encode_batch(x, valid=valid, mean=coded)
        B = raw.shape[0]
        coarse = stacked[0].transpose(0, 2, 1).reshape(B, -1)
        rec = {"base": np.concatenate([fields, coarse], 1)[None]}
        for name, idx in zip(self.names[1:], stacked[1:]):
            rec[name] = idx.transpose(2, 0, 1)
        if stats:
            rec["quality"] = self._quality(stacked, coded, x, valid)[None]
        return rec, x, valid

    def _quality(self, stacked, mean, x, valid):
        """(B, 2 * n_rungs) codes: per rung, mean and p99 of 1 - cos."""
        import torch
        dev = self.device
        idx = [torch.from_numpy(s).to(dev) for s in stacked]
        cums = [_cumulative(rvq, t, sp.depth)
                for rvq, t, sp in zip(self.model.rvqs, idx, self.specs)]
        B = x.shape[0]
        vf = valid.reshape(B, -1)
        means, p99s = [], []
        for r in self.rungs:
            out = self._forward([c[L] for c, L in zip(cums, r)], r, mean, valid)
            e = (1.0 - (out * x).sum(1)).clamp_min(0.0).reshape(B, -1)
            e = torch.where(vf, e, torch.full_like(e, float("nan")))
            means.append(e.nanmean(1))
            p99s.append(torch.nanquantile(e, 0.99, dim=1))
        m = quality_code(torch.stack(means, 1).cpu().numpy())
        p = quality_code(torch.stack(p99s, 1).cpu().numpy())
        return np.concatenate([m, p], 1).astype(np.int64)

    def _forward(self, zqs, depths, mean, valid):
        import torch
        B = mean.shape[0]
        Ls = torch.tensor(list(depths), device=self.device)[None].expand(B, -1)
        with self._ac():
            out = self.model.decode(zqs, Ls, stats=mean, valid=valid)
        return out.float()

    def encode_block(self, raw, valid=None, stats=True, batch=256) -> BlockRecords:
        """raw: (C, H, W) int8 north-up. `valid` (H, W), when given, marks
        further pixels invalid; otherwise validity is band 0 != -128."""
        raw = np.asarray(raw)
        if raw.dtype != np.int8 or raw.ndim != 3 or raw.shape[0] != self.bands:
            raise ValueError(f"expected int8 ({self.bands}, H, W), got "
                             f"{raw.dtype} {raw.shape}")
        C, H, Wd = raw.shape
        W = self.window
        gh, gw = -(-H // W), -(-Wd // W)
        a = np.full((C, gh * W, gw * W), NODATA, np.int8)
        a[:, :H, :Wd] = raw
        if valid is not None:
            v = np.zeros((gh * W, gw * W), bool)
            v[:H, :Wd] = valid
            a[:, ~v] = NODATA
        pv = a[0] != NODATA
        live = pv.reshape(gh, W, gw, W).any((1, 3))
        rows, cols = np.nonzero(live)
        win = a.reshape(C, gh, W, gw, W)
        parts = []
        for i in range(0, len(rows), batch):
            r, c = rows[i:i + batch], cols[i:i + batch]
            chunk = np.ascontiguousarray(win[:, r, :, c, :])   # (n, C, W, W)
            parts.append(self.encode_windows(chunk, stats=stats)[0])
        if parts:
            recs = {k: np.concatenate([p[k] for p in parts], 1) for k in parts[0]}
        else:
            recs = {}
        return BlockRecords(rows.astype(np.int64), cols.astype(np.int64), recs,
                            pv[:H, :Wd], (gh, gw))

    # ---- decode ----

    @_no_grad
    def decode_records(self, base, tiers, depths, valid, trace=None):
        """base: (n, n_values) base records; tiers: per finer tier (n,
        levels_read, cells) codes, or None at depth 0; depths: the tuple; valid: (n, W, W)
        bool. -> (n, C, W, W) int8 with invalid pixels at -128. `trace`,
        a georvq_parquet.Trace, records the decode stages; on CUDA each
        stage then waits for the device before it ends."""
        out, v = self.decode_field(base, tiers, depths, valid, trace)
        with span(trace, "decode", "requantize", sync=self._sync(trace)):
            return requant(out, v)

    def _sync(self, trace):
        if trace is None or self.device.type != "cuda":
            return None
        import torch
        return torch.cuda.synchronize

    @_no_grad
    def decode_field(self, base, tiers, depths, valid, trace=None):
        """As decode_records, but returns the decoded float field (n, C,
        W, W) on the codec's device before requantization, and the
        validity as a device tensor."""
        import torch
        dev = self.device
        sync = self._sync(trace)
        n = base.shape[0]
        sp0 = self.specs[0]
        with span(trace, "decode", "mean channel", sync=sync):
            mean = torch.from_numpy(
                self.channel.decode(base[:, :self.n_mean])).to(dev)
        coarse = base[:, self.n_mean:].reshape(n, sp0.depth, sp0.cells)
        idx = [coarse.transpose(0, 2, 1)] + [
            None if t is None else t.transpose(0, 2, 1) for t in tiers]
        zqs = []
        with span(trace, "decode", "codebook lookup", sync=sync):
            for rvq, sp, i, L in zip(self.model.rvqs, self.specs, idx, depths):
                if int(L) == 0:
                    zqs.append(torch.zeros(n, sp.cells, sp.dim, device=dev))
                    continue
                i = torch.from_numpy(np.ascontiguousarray(i)).to(dev)
                zqs.append(_cumulative(rvq, i, int(L))[int(L)])
            v = torch.from_numpy(np.ascontiguousarray(valid)).to(dev)
        with span(trace, "decode", "decoder forward", sync=sync):
            return self._forward(zqs, depths, mean, v), v


def requant_ae_np(x, valid=None, fill=NODATA):
    """float32 field -> int8 signed-sqrt quantization (inverse of dequant);
    invalid pixels -> `fill` (the source's own nodata value)."""
    s = np.rint(np.sign(x) * np.sqrt(np.abs(x)) * 127.5)
    s = np.clip(s, -127, 127).astype(np.int8)
    if valid is not None:
        s[np.broadcast_to(~valid[:, None], s.shape)] = fill
    return s


def requant(out, valid):
    """Decoded float field (n, C, W, W) -> int8, the signed-sqrt inverse of
    the AE dequantization, with invalid pixels at -128. On CUDA it runs
    on the device, where sqrt is correctly rounded; on CPU it runs in
    numpy, because torch's vectorized CPU sqrt is not correctly rounded
    and would differ from `requant_ae_np` in rare cases."""
    if out.device.type != "cuda":
        return requant_ae_np(out.cpu().numpy(), valid.cpu().numpy())
    import torch
    q = torch.clamp(torch.round(torch.sign(out) * torch.sqrt(out.abs())
                                * 127.5), -127, 127).to(torch.int8)
    q = torch.where(valid[:, None], q, torch.full_like(q, NODATA))
    return q.cpu().numpy()


def _cumulative(rvq, idx, L):
    """Partial codebook sums at depths 0..L for (B, cells, >=L) indices,
    added level by level in the order Codec.decode adds them."""
    import torch
    B, cells = idx.shape[:2]
    out = [torch.zeros(B, cells, rvq.levels[0].dim, device=idx.device)]
    z = None
    for lev in range(L):
        v = rvq.levels[lev].codebook(idx[..., lev])
        z = v if z is None else z + v
        out.append(z)
    return out


# --------------------------------------------------------------------------
# ONNX
# --------------------------------------------------------------------------

ONNX_CHANNEL_NAME = "mean_channel.npz"


def _channel_sha(channel) -> str:
    import hashlib
    h = hashlib.sha256()
    for a in channel.id_state():
        h.update(np.asarray(a).tobytes())
    return h.hexdigest()[:16]


def export_onnx(codec: AEV2Codec, out_dir, opset=18) -> dict:
    """Export an AEV2Codec for `AEV2OnnxCodec`: the stats_cond graphs and
    codebooks (georvq.inference.onnx_stats) and the mean channel's
    tables, under the codec's bitstream id. Returns the meta dict."""
    from georvq.inference.onnx_stats import export_stats_onnx
    os.makedirs(out_dir, exist_ok=True)
    codec.channel.save(os.path.join(out_dir, ONNX_CHANNEL_NAME))
    extra = {"ae_v2": {"format": FORMAT, "mean_channel": ONNX_CHANNEL_NAME,
                       "mean_channel_sha": _channel_sha(codec.channel),
                       "min_depths": list(codec.model.cfg.min_depths),
                       "rungs": [list(r) for r in codec.rungs]}}
    meta = export_stats_onnx(codec.model, out_dir,
                             extra_id=codec.channel.id_state(),
                             meta_extra=extra, opset=opset)
    if meta["bitstream_id"] != codec.bitstream_id:
        raise RuntimeError(f"export bitstream id {meta['bitstream_id']} != "
                           f"codec {codec.bitstream_id}")
    return meta


class AEV2OnnxCodec:
    """Decode side of AEV2Codec over an `export_onnx` directory, with
    onnxruntime and numpy only. Decoding runs in fp32 on any device, so
    on CPU it agrees with the torch fp32 decode up to float rounding in
    the graphs, which can move a rare int8 value by one step."""

    def __init__(self, onnx_dir, device="cpu", threads=None):
        from georvq.inference.onnx_stats import OnnxStatsCodec
        self.ort = OnnxStatsCodec(onnx_dir, device=device, threads=threads)
        meta = self.ort.meta
        ae = meta.get("ae_v2")
        if ae is None or ae.get("format") != FORMAT:
            raise ValueError(f"{onnx_dir}: not an AE v2 ONNX export")
        self.channel = RVQPCAMeanChannel.load(
            os.path.join(onnx_dir, ae["mean_channel"]))
        if _channel_sha(self.channel) != ae["mean_channel_sha"]:
            raise ValueError(f"{onnx_dir}: mean channel tables do not match "
                             "the export's record of them")
        from georvq.config import TierSpec
        self.specs = [TierSpec(**t) for t in meta["tiers"]]
        self.window = int(meta["window"])
        self.bands = int(meta["bands"])
        self.full = tuple(sp.depth for sp in self.specs)
        self.rungs = [tuple(r) for r in ae["rungs"]]
        self.names = tier_names(len(self.specs))
        self.n_mean = len(self.channel.field_widths())
        self.bitstream_id = meta["bitstream_id"]
        self.decoder_id = meta["decoder_id"]
        self.bf16 = False

    def decode_records(self, base, tiers, depths, valid, trace=None):
        """As AEV2Codec.decode_records. The trace records the mean
        channel, the ONNX graphs (codebook lookup and decoder together)
        and requantization."""
        n = base.shape[0]
        sp0 = self.specs[0]
        with span(trace, "decode", "mean channel"):
            mean = self.channel.decode(base[:, :self.n_mean])
        coarse = base[:, self.n_mean:].reshape(n, sp0.depth, sp0.cells)
        idx = [coarse.transpose(0, 2, 1)] + [
            None if t is None else t.transpose(0, 2, 1) for t in tiers]
        valid = np.ascontiguousarray(valid, dtype=bool)
        with span(trace, "decode", "lookup + decoder (onnx)"):
            out = self.ort.decode(idx, depths, mean, valid)
        with span(trace, "decode", "requantize"):
            return requant_ae_np(out, valid)


def load_codec(model=None, channel=None, device="cpu", bf16=None,
               backend="torch", onnx_dir=None, threads=None):
    """An AEV2Codec (backend "torch": `model` and `channel` as for
    AEV2Codec.load) or an AEV2OnnxCodec (backend "onnx": `onnx_dir`, an
    `export_onnx` directory, which holds the mean channel too)."""
    if backend == "onnx":
        if onnx_dir is None:
            raise ValueError("backend 'onnx' needs onnx_dir")
        return AEV2OnnxCodec(onnx_dir, device=device, threads=threads)
    if backend != "torch":
        raise ValueError(f"unknown backend {backend!r}")
    if model is None or channel is None:
        raise ValueError("backend 'torch' needs a model and its mean channel")
    return AEV2Codec.load(model, channel, device=device, bf16=bf16)


# --------------------------------------------------------------------------
# tile encode
# --------------------------------------------------------------------------

def _source_meta(ds) -> dict:
    """What a reconstruction needs from the source dataset, as JSON."""
    prof = dict(ds.profile)
    prof["crs"] = ds.crs.to_wkt() if ds.crs else None
    prof["transform"] = list(ds.transform)[:6]
    return {"profile": prof,
            "orientation": "bottom-up" if ds.transform.e > 0 else "north-up",
            "tags": ds.tags(),
            "band_tags": [ds.tags(i) for i in range(1, ds.count + 1)],
            "descriptions": list(ds.descriptions),
            "block_shapes": [list(b) for b in ds.block_shapes],
            "overviews": ds.overviews(1),
            "image_structure": ds.tags(ns="IMAGE_STRUCTURE")}


def _projjson(crs):
    if crs is None:
        return None
    try:
        return crs.to_dict(projjson=True)
    except (TypeError, AttributeError):
        from pyproj import CRS
        return CRS.from_wkt(crs.to_wkt()).to_json_dict()


def encode_tile(src_path, out_path, codec: AEV2Codec, stats=True,
                chunk_px=None, strip_px=STRIP_PX, batch=256,
                release=RELEASE_NAME, uri=RELEASE_URI, revision=None,
                source=None, log=None) -> dict:
    """Encode one AE tile (GeoTIFF) to `out_path`, conventionally
    `output_name(src_path)`. Strips of `strip_px` rows are encoded in
    turn. chunk_px: per-kind chunk sizes, `chunk_sizes` by default.
    `release`, `uri` and `revision` are written to the header's
    `codec.release` (see georvq.ae.v2_resolve); `release=None` writes
    none. `source` is recorded as the header's `product.source`, the
    path or URL a reader is pointed to for the source tile (default
    `src_path`). Returns timings."""
    import rasterio
    from rasterio.windows import Window
    W = codec.window
    path = os.fspath(src_path)
    if source is None and os.path.isabs(path) and not path.startswith("/vsi"):
        warnings.warn(f"the header's product.source records the local path "
                      f"{path}; pass source= (--source) with the path or URL "
                      "readers of the file should use", stacklevel=2)
    if strip_px % W:
        raise ValueError(f"strip_px {strip_px} is not a multiple of {W}")
    t0 = time.time()
    with rasterio.Env(**_GDAL_ENV), rasterio.open(src_path) as ds:
        if ds.count != codec.bands or ds.dtypes[0] != "int8":
            raise ValueError(f"{src_path}: {ds.count} x {ds.dtypes[0]}, expected "
                             f"{codec.bands} x int8")
        if ds.nodata is not None and int(ds.nodata) != NODATA:
            raise ValueError(f"{src_path}: nodata {ds.nodata}, expected {NODATA}")
        H, Wd = ds.height, ds.width
        bottom_up = ds.transform.e > 0
        north = _flip_rows(ds.transform, H) if bottom_up else ds.transform
        meta = _source_meta(ds)
        pid = os.path.splitext(os.path.basename(os.fspath(src_path)))[0]
        header = {
            "codec": codec.header_codec(
                release_record(codec, release, uri, revision)),
            "product": {
                "kind": KIND, "id": pid,
                "source": os.fspath(src_path if source is None else source),
                "crs_wkt": ds.crs.to_wkt() if ds.crs else None,
                "epsg": ds.crs.to_epsg() if ds.crs else None,
                "transform": list(north)[:6],
                "width": Wd, "height": H, "count": ds.count,
                "dtype": "int8", "nodata": NODATA,
                "orientation": meta["orientation"]},
            "rungs": codec.header_rungs()}
        if stats:
            header["quality"] = codec.header_quality()
        lay = codec.layout(quality=stats, chunk_px=chunk_px)
        t_read = t_enc = t_write = 0.0
        w = RVQParquetWriter(out_path, header, lay, pid, list(north)[:6], H, Wd,
                             crs=_projjson(ds.crs),
                             aux=json.dumps(meta, default=str))
        for y0 in range(0, H, strip_px):
            y1 = min(H, y0 + strip_px)
            s0, s1 = (H - y1, H - y0) if bottom_up else (y0, y1)
            ta = time.time()
            a = ds.read(window=Window(0, s0, Wd, s1 - s0))
            if bottom_up:
                a = a[:, ::-1]
            tb = time.time()
            rec = codec.encode_block(a, stats=stats, batch=batch)
            tc = time.time()
            rec.add_to(w, window_row=y0 // W)
            t_read += tb - ta
            t_enc += tc - tb
            t_write += time.time() - tc
            if log:
                log(f"rows {y1}/{H}: {len(rec.rows)} windows")
        tc = time.time()
        w.close()
        t_write += time.time() - tc
    return {"read_s": t_read, "encode_s": t_enc, "write_s": t_write,
            "total_s": time.time() - t0}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

class AEV2Dataset(WindowMethodsMixin, TransformMethodsMixin):
    """Read-only view of an AE v2 file that reads like the rasterio
    DatasetReader of the SOURCE tile.

    Georeferencing, shape and the read methods follow the source tile,
    including its bottom-up row order, so code written against the
    source dataset can read this instead: `read`, `read_masks`,
    `dataset_mask`, `bounds`, `res`, `index`, `xy`, `window` and the
    other window helpers give the answers rasterio gives for the source
    tile, with decoded pixels in place of the source's. `read` takes a
    `quality` beyond rasterio's arguments. `driver` is "GeoRVQParquet"
    ("GeoRVQZarr" for a Zarr store);
    `profile` and `meta` carry the source's driver, so either can be
    passed to `rasterio.open(..., "w")` to write a GeoTIFF. The file has
    no overviews, so a decimated read samples the full-resolution
    decode, where rasterio would read the source's overviews.

    `read` needs a codec; the metadata and masks do not. Without a
    codec, `resolver(header)` is called for one the first time `codec`
    is used."""

    driver = "GeoRVQParquet"
    mode = "r"

    def __init__(self, path_or_url, codec=None, resolver=None, **reader_kw):
        self.reader = open_reader(path_or_url, **reader_kw)
        if isinstance(self.reader, ZarrReader):
            self.driver = ZarrReader.driver
        self.name = self.reader.source
        h = self.header = self.reader.header
        p = self.product = h["product"]
        if p.get("kind") != KIND:
            raise ValueError(f"{self.name}: product kind {p.get('kind')!r}, "
                             f"not {KIND}")
        self._codec = None
        self._resolver = resolver
        if codec is not None:
            self.codec = codec
        self.height, self.width = int(p["height"]), int(p["width"])
        self.count = int(p["count"])
        self.dtypes = (p["dtype"],) * self.count
        self.nodata = p["nodata"]
        self.window_px = int(h["codec"]["window"])
        self.bottom_up = p["orientation"] == "bottom-up"
        from affine import Affine
        north = Affine(*p["transform"])
        self.transform = (_flip_rows(north, self.height) if self.bottom_up
                          else north)
        self.rungs = [tuple(r["tuple"]) for r in h["rungs"]]
        self.full = tuple(int(t.split(":")[3]) for t in h["codec"]["tiers"])
        self.names = tier_names(len(self.full))
        self._meta = None
        self._crs = None
        self._closed = False

    @property
    def codec(self):
        if self._codec is None and self._resolver is not None:
            self.codec = self._resolver(self.header)
        return self._codec

    @codec.setter
    def codec(self, codec):
        bid = self.header["codec"]["bitstream_id"]
        if codec is not None and codec.bitstream_id != bid:
            raise ValueError(f"{self.name}: bitstream {bid}, codec holds "
                             f"{codec.bitstream_id}")
        self._codec = codec

    @property
    def crs(self):
        if self._crs is None and self.product.get("crs_wkt"):
            from rasterio.crs import CRS
            self._crs = CRS.from_wkt(self.product["crs_wkt"])
        return self._crs

    @property
    def source_meta(self) -> dict:
        if self._meta is None:
            self._meta = json.loads(self.reader.aux) if self.reader.aux else {}
        return self._meta

    @property
    def profile(self) -> dict:
        prof = dict(self.source_meta.get("profile") or {
            "driver": "GTiff", "tiled": True, "blockxsize": 1024,
            "blockysize": 1024, "compress": "zstd", "interleave": "band"})
        prof.update(dtype=self.dtypes[0], nodata=self.nodata, width=self.width,
                    height=self.height, count=self.count, crs=self.crs,
                    transform=self.transform)
        return prof

    @property
    def descriptions(self):
        return tuple(self.source_meta.get("descriptions") or
                     (None,) * self.count)

    def tags(self, bidx=0):
        m = self.source_meta
        return dict(m.get("tags", {}) if bidx == 0
                    else (m.get("band_tags") or [{}] * self.count)[bidx - 1])

    @property
    def block_shapes(self):
        p = self.profile
        return [(int(p.get("blockysize", 1024)), int(p.get("blockxsize", 1024)))
                ] * self.count

    def block_windows(self, bidx=0):
        from rasterio.windows import Window
        bh, bw = self.block_shapes[0]
        for i, r in enumerate(range(0, self.height, bh)):
            for j, c in enumerate(range(0, self.width, bw)):
                yield (i, j), Window(c, r, min(bw, self.width - c),
                                     min(bh, self.height - r))

    @property
    def shape(self):
        return self.height, self.width

    @property
    def indexes(self):
        return tuple(range(1, self.count + 1))

    @property
    def bounds(self):
        """As rasterio's DatasetBase.bounds; for a bottom-up raster
        `bottom` is above `top`, as rasterio reports it."""
        from rasterio.coords import BoundingBox
        a, b, c, d, e, f = tuple(self.transform)[:6]
        if b == d == 0:
            return BoundingBox(c, f + e * self.height, c + a * self.width, f)
        xs, ys = zip(*(self.transform * p for p in
                       ((0, 0), (0, self.height), (self.width, self.height),
                        (self.width, 0))))
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))

    @property
    def res(self):
        a, b, _, d, e, _ = tuple(self.transform)[:6]
        return math.sqrt(a * a + d * d), math.sqrt(b * b + e * e)

    @property
    def meta(self):
        return {"driver": self.profile.get("driver", "GTiff"),
                "dtype": self.dtypes[0], "nodata": self.nodata,
                "width": self.width, "height": self.height,
                "count": self.count, "crs": self.crs,
                "transform": self.transform}

    def overviews(self, bidx=1):
        """The file holds no overviews."""
        return []

    @property
    def closed(self):
        return self._closed

    def close(self):
        self.reader.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- quality ----

    def _depth_map(self, quality, wy0, wy1, wx0, wx1) -> np.ndarray:
        """(rows, cols) rung index per window. A rung or None (full depth)
        that is not a rung of the file raises ValueError; a bound resolves
        to the last rung, which is full depth, where no rung meets it."""
        shape = (wy1 - wy0, wx1 - wx0)
        if quality is None:
            t = self.full
        else:
            m = _BOUND.match(quality) if isinstance(quality, str) else None
            if m is None:
                try:
                    t = parse_rung(quality, self.full)
                except (TypeError, ValueError) as err:
                    if "does not fit" in str(err):
                        raise
                    raise ValueError(
                        f"quality {quality!r}: expected a rung such as "
                        "'8-8-2', '8,8,2' or (8, 8, 2), a bound such as "
                        f"'p99<=0.02' on one of the stats "
                        f"{list(self.header.get('quality', {}).get('stats', QUALITY_STATS))}"
                        ", or None for full depth") from None
            else:
                return self._resolve_bound(m.group(1), float(m.group(2)),
                                           wy0, wy1, wx0, wx1)
        if t not in self.rungs:
            raise ValueError(f"{t} is not a rung of this file: "
                             f"{['-'.join(map(str, r)) for r in self.rungs]}")
        return np.full(shape, self.rungs.index(t), np.int64)

    def _resolve_bound(self, stat, bound, wy0, wy1, wx0, wx1):
        if self.reader.layout.quality is None:
            raise ValueError(f"{self.name}: no quality column to resolve a "
                             "bound against")
        stats = self.header["quality"]["stats"]
        if stat not in stats:
            raise ValueError(f"quality stat {stat!r} not in {stats}")
        n = len(self.rungs)
        q = self.quality_codes(wy0, wy1, wx0, wx1)
        k = stats.index(stat)
        ok = quality_value(q[..., k * n:(k + 1) * n]) <= bound
        first = np.where(ok.any(-1), ok.argmax(-1), n - 1)
        return first.astype(np.int64)

    def quality_codes(self, wy0, wy1, wx0, wx1) -> np.ndarray:
        """(rows, cols, stats * rungs) quality codes of windows
        [wy0:wy1, wx0:wx1], in the north-up window grid."""
        return self.reader.read(wy0, wy1, wx0, wx1, depths=[0] * len(self.full),
                                base=False, mask=False, quality=True)["quality"]

    def window_rungs(self, quality, window=None) -> np.ndarray:
        """Rung index chosen for each coded window under `window`."""
        ny0, ny1, nx0, nx1 = self._north_px(window)
        W = self.window_px
        return self._depth_map(quality, ny0 // W, -(-ny1 // W), nx0 // W,
                               -(-nx1 // W))

    # ---- read ----

    def _north_px(self, window):
        """Source-pixel window -> north-up pixel bounds (y0, y1, x0, x1),
        offsets and lengths rounded."""
        if window is None:
            r0, r1, c0, c1 = 0, self.height, 0, self.width
        elif hasattr(window, "row_off"):
            r0, c0 = int(round(window.row_off)), int(round(window.col_off))
            r1, c1 = r0 + int(round(window.height)), c0 + int(round(window.width))
        else:
            (r0, r1), (c0, c1) = window
        if not (0 <= r0 < r1 <= self.height and 0 <= c0 < c1 <= self.width):
            raise ValueError(f"window rows {r0}:{r1} cols {c0}:{c1} outside "
                             f"the {self.height}x{self.width} raster")
        if self.bottom_up:
            r0, r1 = self.height - r1, self.height - r0
        return r0, r1, c0, c1

    def _check_open(self):
        if self._closed:
            from rasterio.errors import RasterioIOError
            raise RasterioIOError(f"Dataset is closed: {self.name}")

    def _bands(self, indexes):
        """(1-based band list, whether the result is 2-D), validated as
        rasterio validates them."""
        if indexes is None:
            return list(self.indexes), False
        two_d = isinstance(indexes, (int, np.integer))
        bands = [indexes] if two_d else list(indexes)
        if not bands:
            raise ValueError("No indexes to read")
        for b in bands:
            if b not in self.indexes:
                raise IndexError(f"band index {b} out of range (not in "
                                 f"{self.indexes})")
        return [int(b) for b in bands], two_d

    def _window(self, window, boundless):
        """`window` as a rasterio Window, cropped to the raster unless
        `boundless`, as rasterio's read does."""
        from rasterio.windows import Window
        if window is None:
            return Window(0, 0, self.width, self.height)
        if not isinstance(window, Window):
            window = Window.from_slices(*window, height=self.height,
                                        width=self.width, boundless=boundless)
        if not boundless:
            window = window.crop(self.height, self.width)
        return window

    @staticmethod
    def _out_hw(window, n_bands, out, out_shape, resampling):
        """Output (rows, cols): `out`'s, `out_shape`'s, or the window's
        lengths rounded as rasterio rounds them."""
        from rasterio.enums import Resampling
        from rasterio.errors import DatasetIOShapeError
        if out is not None and out_shape is not None:
            raise ValueError("out and out_shape are exclusive")
        shape = out.shape if out is not None else out_shape
        if shape is None:
            hw = (math.floor(window.height + 0.5), math.floor(window.width + 0.5))
        else:
            shape = tuple(int(v) for v in shape)
            if len(shape) == 3 and shape[0] != n_bands:
                raise DatasetIOShapeError("Dataset indexes and destination "
                                          "buffer are mismatched")
            hw = shape[-2:]
        exact = (hw == (window.height, window.width)
                 and float(window.row_off).is_integer()
                 and float(window.col_off).is_integer())
        if resampling not in (None, Resampling.nearest) and not exact:
            raise NotImplementedError(
                f"resampling {Resampling(resampling).name}: only nearest is "
                "implemented")
        return hw

    def _gather(self, window, oh, ow, fetch, k, fill):
        """(k, oh, ow) samples of `window` at nearest-neighbour positions,
        as GDAL picks them, and the (oh, ow) mask of those inside the
        raster. `fetch(y0, y1, x0, x1)` returns (k, y1 - y0, x1 - x0) for
        a north-up pixel block; it is called on runs of window rows of at
        most STRIP_PX rows that hold a sample. Samples outside the raster
        are `fill`."""
        def sample(off, length, n, size):
            # An output pixel whose footprint overlaps the raster takes the
            # nearest raster pixel, clamped at the edge, as GDAL's VRT
            # does for a boundless read.
            step, j = length / max(n, 1), np.arange(n)
            at = np.floor(off + (j + 0.5) * step + 1e-10).astype(np.int64)
            ins = (off + (j + 1) * step > 1e-10) & (off + j * step < size - 1e-10)
            return np.clip(at, 0, size - 1), ins
        ry, iy = sample(window.row_off, window.height, oh, self.height)
        cx, ix = sample(window.col_off, window.width, ow, self.width)
        inside = iy[:, None] & ix[None, :]
        out = None
        if iy.any() and ix.any():
            W = self.window_px
            ny = self.height - 1 - ry if self.bottom_up else ry
            jo = np.nonzero(ix)[0]
            c0, c1 = int(cx[jo].min()), int(cx[jo].max()) + 1
            cols = cx[jo] - c0
            dense = ((np.diff(cols) == 1).all() and (np.diff(jo) == 1).all()
                     and (np.diff(np.nonzero(iy)[0]) == 1).all()
                     and (np.abs(np.diff(ny[iy])) == 1).all())
            need = np.unique(ny[iy] // W)
            runs = np.split(need, np.nonzero(np.diff(need) > 1)[0] + 1)
            step = max(1, STRIP_PX // W)
            for run in runs:
                for i in range(0, len(run), step):
                    y0 = int(run[i]) * W
                    y1 = min(self.height, (int(run[min(i + step, len(run)) - 1]) + 1) * W)
                    blk = fetch(y0, y1, c0, c1)
                    if out is None:
                        out = np.full((k, oh, ow), fill, blk.dtype)
                    io = np.nonzero(iy & (ny >= y0) & (ny < y1))[0]
                    rows = ny[io] - y0
                    if dense:
                        # unresampled read: a run of the block's rows, in
                        # order or reversed, copied without a gather
                        seg = blk[:, rows.min():rows.max() + 1]
                        if len(rows) > 1 and rows[1] < rows[0]:
                            seg = seg[:, ::-1]
                        out[:, io[0]:io[-1] + 1, jo[0]:jo[-1] + 1] = seg
                    else:
                        out[:, io[:, None], jo[None, :]] = (
                            blk[:, rows][:, :, cols])
        return out, inside

    def _decode_north(self, ny0, ny1, nx0, nx1, quality, batch):
        """int8 (count, rows, cols) of north-up pixels [ny0:ny1, nx0:nx1]."""
        W = self.window_px
        wy0, wy1 = ny0 // W, -(-ny1 // W)
        wx0, wx1 = nx0 // W, -(-nx1 // W)
        gh, gw = wy1 - wy0, wx1 - wx0
        trace = self.reader.trace
        with span(trace, "cpu", "resolve rungs"):
            rung_of = self._depth_map(quality, wy0, wy1, wx0, wx1)
        depth_of = np.array(self.rungs)[rung_of]           # (gh, gw, tiers)

        # One fetch covers the mask, base and every level any window
        # needs. Windows without valid pixels resolve to the cheapest
        # rung, so they do not raise the depth read.
        need = depth_of.reshape(-1, depth_of.shape[-1]).max(0)
        got = self.reader.read(wy0, wy1, wx0, wx1, depths=[int(v) for v in need])
        vwin = got["valid"].reshape(gh, W, gw, W).transpose(0, 2, 1, 3)
        live = vwin.any((2, 3))

        out = np.full((self.count, gh * W, gw * W), NODATA, np.int8)
        if live.any():
            base = got["base"]
            tiers = [got.get(name) for name in self.names[1:]]
            for k in np.unique(rung_of[live]):
                r = self.rungs[k]
                ys, xs = np.nonzero(live & (rung_of == k))
                for i in range(0, len(ys), batch):
                    y, x = ys[i:i + batch], xs[i:i + batch]
                    fin = [tiers[t - 1][y, x, :r[t]] if r[t] else None
                           for t in range(1, len(r))]
                    kw = {} if trace is None else {"trace": trace}
                    dec = self.codec.decode_records(base[y, x], fin, r,
                                                    vwin[y, x], **kw)
                    with span(trace, "cpu", "assemble"):
                        for d, yy, xx in zip(dec, y, x):
                            out[:, yy * W:(yy + 1) * W,
                                xx * W:(xx + 1) * W] = d
        return out[:, ny0 - wy0 * W:ny1 - wy0 * W, nx0 - wx0 * W:nx1 - wx0 * W]

    def read(self, indexes=None, out=None, window=None, masked=False,
             out_shape=None, boundless=False, resampling=None,
             fill_value=None, out_dtype=None, *, quality=None, batch=256):
        """Decoded pixels, with rasterio's DatasetReader.read arguments
        and results: (bands, rows, cols), or (rows, cols) for an int
        `indexes` (1-based, validated against 1..count with IndexError).

        A window that extends off the raster is cropped to it unless
        `boundless`, where pixels outside the raster are `fill_value`
        (default nodata). Without `boundless`, a `fill_value` replaces
        nodata pixels, as in rasterio. Fractional windows and `out_shape`
        / `out` sample at nearest-neighbour positions as GDAL does; other
        `resampling` methods raise NotImplementedError. `masked` returns
        a numpy masked array masking nodata pixels and pixels outside the
        raster; for a boundless read with a `fill_value` it masks, as
        rasterio does, the pixels outside the raster and those equal to
        `fill_value`, and not nodata pixels. `out_dtype`
        converts as GDAL does, clamping integers to the type's range.

        `quality`: a rung ("8-8-2" or a tuple), a bound on a quality stat
        ("p99<=0.02", resolved per window to the cheapest rung meeting
        it, full depth where none does), or None for full depth."""
        self._check_open()
        bands, two_d = self._bands(indexes)
        if self.codec is None:
            raise ValueError("reading pixels needs a codec: open the file "
                             "with open_dataset, which finds the file's "
                             "bundle, or pass codec=")
        win = self._window(window, boundless)
        oh, ow = self._out_hw(win, len(bands), out, out_shape, resampling)
        sel = [b - 1 for b in bands]
        all_bands = sel == list(range(self.count))

        def fetch(y0, y1, x0, x1):
            a = self._decode_north(y0, y1, x0, x1, quality, batch)
            return a if all_bands else a[sel]

        vals, inside = self._gather(win, oh, ow, fetch, len(bands), NODATA)
        if vals is None:
            vals = np.full((len(bands), oh, ow), NODATA, np.int8)
        nod = vals == NODATA
        dtype = np.dtype(out.dtype if out is not None else out_dtype or np.int8)
        if dtype.kind in "iu" and dtype != vals.dtype:
            info = np.iinfo(dtype)
            res = np.clip(vals, max(info.min, -128), min(info.max, 127)).astype(dtype)
        else:
            res = vals.astype(dtype, copy=False)
        if fill_value is not None:
            if boundless:
                res[:, ~inside] = fill_value
            elif not masked:
                res[nod] = fill_value
        if masked:
            # rasterio reads a boundless window through a VRT whose nodata
            # is the fill_value: it masks the pixels outside the raster and
            # those equal to fill_value, and leaves nodata pixels unmasked
            mask = (~inside[None] | (res == fill_value)
                    if boundless and fill_value is not None else nod)
            res = np.ma.array(res, mask=mask, fill_value=(
                fill_value if fill_value is not None else self.nodata))
        if two_d:
            res = res[0]
        if out is not None:
            out[...] = np.ma.getdata(res).reshape(out.shape)
            return (np.ma.array(out, mask=np.ma.getmaskarray(res).reshape(
                out.shape), fill_value=res.fill_value) if masked else out)
        return res if masked else np.ascontiguousarray(res)

    def read_masks(self, indexes=None, out=None, out_shape=None, window=None,
                   boundless=False, resampling=None):
        """uint8 masks, 255 valid and 0 nodata or outside the raster, as
        rasterio's read_masks: (bands, rows, cols), or (rows, cols) for an
        int `indexes`. Every band shares the file's pixel validity. Needs
        no codec."""
        self._check_open()
        bands, two_d = self._bands(indexes)
        win = self._window(window, boundless)
        oh, ow = self._out_hw(win, len(bands), out, out_shape, resampling)
        m = self._mask(win, oh, ow)
        res = np.broadcast_to(m, (len(bands), oh, ow))
        if two_d:
            res = res[0]
        if out is not None:
            out[...] = res.reshape(out.shape)
            return out
        return np.ascontiguousarray(res)

    def dataset_mask(self, out=None, out_shape=None, window=None,
                     boundless=False, resampling=None):
        """uint8 (rows, cols), 255 where any band is valid, as rasterio's
        dataset_mask. Needs no codec."""
        self._check_open()
        win = self._window(window, boundless)
        oh, ow = self._out_hw(win, 1, out, out_shape, resampling)
        res = self._mask(win, oh, ow)
        if out is not None:
            out[...] = res
            return out
        return res

    def _mask(self, win, oh, ow):
        """uint8 (oh, ow) validity samples of `win`, 255 valid."""
        def fetch(y0, y1, x0, x1):
            return self.reader.read_mask(y0, y1, x0, x1)[None]
        m, _ = self._gather(win, oh, ow, fetch, 1, False)
        if m is None:
            return np.zeros((oh, ow), np.uint8)
        return m[0].astype(np.uint8) * np.uint8(255)


def open_dataset(path_or_url, model=None, channel=None, bundle=None,
                 onnx_dir=None, device=None, backend=None, trust_remote=False,
                 cache_dir=None, bf16=None, codec=None, threads=None,
                 record_cache_dir=None, **reader_kw) -> "AEV2Dataset":
    """Open an AE v2 file (a local path or an http(s) URL), or a
    prototype Zarr store of one (a path or URL ending in .zarr, or an
    open zarr.Group read through zarr-python; see
    georvq.inference.georvq_zarr).

    The codec that reads pixels comes from the first of
    - `codec`, an AEV2Codec or AEV2OnnxCodec;
    - `model` (a checkpoint path or a model) and `channel` (a .npz path
      or a channel), torch backend;
    - `bundle`, a bundle directory, or `onnx_dir`, an `export_onnx`
      directory (onnx backend);
    - the bundle `georvq.ae.v2_resolve.locate_bundle` finds for the
      file: $GEORVQ_AE_BUNDLE, the cache under `cache_dir` (default
      $GEORVQ_CACHE, $XDG_CACHE_HOME/georvq or ~/.cache/georvq), or a
      download from the release the header names, when that release is
      trusted or `trust_remote` is set.
    Explicit arguments are loaded on open; the located bundle is loaded
    when the codec is first needed, so metadata can be read without it.
    A bundle for another bitstream is refused.

    backend: "torch" or "onnx"; default torch when it imports (onnx
    when `onnx_dir` is given). device: default cuda when the torch
    backend sees a GPU, else cpu. `record_cache_dir` is the
    RVQParquetReader's on-disk record cache; `reader_kw` go to it too
    (`first_read`, `gap`, `concurrency`, `cache`, `cache_bytes`, and
    `trace`, a georvq_parquet.Trace that also records the decode stages)."""
    if record_cache_dir is not None:
        reader_kw["cache_dir"] = record_cache_dir
    explicit = codec is not None
    if codec is None and (model is not None or channel is not None):
        if backend not in (None, "torch"):
            raise ValueError("model= and channel= load the torch backend")
        if model is None or channel is None:
            raise ValueError("a model needs its mean channel")
        with _quiet_build():
            codec = load_codec(model, channel, device=device or
                               default_device("torch"), bf16=bf16)
        explicit = True
    elif codec is None and bundle is None and onnx_dir is not None:
        if backend not in (None, "onnx"):
            raise ValueError("onnx_dir= loads the onnx backend")
        codec = load_codec(backend="onnx", onnx_dir=onnx_dir,
                           device=device or "cpu", threads=threads)
        explicit = True

    def resolver(header):
        with _quiet_build():
            return resolve_codec(header, bundle=bundle, backend=backend,
                                 device=device, bf16=bf16, threads=threads,
                                 cache_dir=cache_dir,
                                 trust_remote=trust_remote)

    ds = AEV2Dataset(path_or_url, codec, None if explicit else resolver,
                     **reader_kw)
    if bundle is not None and not explicit:
        try:
            ds.codec
        except BaseException:
            ds.close()
            raise
    return ds


open = open_dataset


def reconstruct(src, out_path, quality=None, rows=1024, creation_options=None):
    """Decode a whole file into a GeoTIFF with the source's profile,
    tags and band descriptions. `src` is an open AEV2Dataset."""
    import rasterio
    from rasterio.windows import Window
    prof = src.profile
    with rasterio.Env(**_GDAL_ENV), rasterio.open(
            out_path, "w", **prof, num_threads="all_cpus",
            **(creation_options or {})) as dst:
        dst.update_tags(**src.tags())
        for b in range(1, src.count + 1):
            if src.tags(b):
                dst.update_tags(b, **src.tags(b))
            if src.descriptions[b - 1] is not None:
                dst.set_band_description(b, src.descriptions[b - 1])
        for r0 in range(0, src.height, rows):
            win = Window(0, r0, src.width, min(rows, src.height - r0))
            dst.write(src.read(window=win, quality=quality), window=win)


def column_bytes(reader: RVQParquetReader) -> dict:
    """Bytes per column group: the chunk metadata columns, `base`,
    `mask`, each tier outside base, `quality`, and the footer."""
    out = {}
    tier_of = {c: t.name for t in reader.layout.tiers for c in t.columns
               if not t.in_base}
    for name, b in reader.column_bytes().items():
        key = tier_of.get(name, name)
        if key in ("kind", "product_id", "chunk_row", "chunk_col", "geometry",
                   "bbox"):
            key = "chunk_meta"
        out[key] = out.get(key, 0) + b
    out["footer"] = reader.metadata.serialized_size + 8
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _window_arg(s):
    if s is None:
        return None
    r, c, h, w = (int(v) for v in s.split(","))
    return ((r, r + h), (c, c + w))


def _fetch_header(target, uri, revision) -> dict:
    """Header of the file `target`, or a stand-in naming a bitstream id
    and the release to download it from."""
    if os.path.exists(target) or target.startswith(("http://", "https://")):
        with RVQParquetReader(target) as rd:
            return rd.header
    if target.endswith(".parquet") or os.sep in target:
        raise FileNotFoundError(f"{target}: no such file")
    return {"codec": {"bitstream_id": target,
                      "release": {"uri": uri, "revision": revision}}}


def main(argv=None):
    from georvq.ae.v2_resolve import locate_bundle
    ap = argparse.ArgumentParser(
        prog="python -m georvq.ae.v2",
        description="AlphaEarth v2 codec: one annual tile <-> one "
                    "georvq GeoParquet file.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cache_args(p):
        p.add_argument("--cache-dir", default=None,
                       help="georvq cache (default $GEORVQ_CACHE, else "
                            "$XDG_CACHE_HOME/georvq or ~/.cache/georvq); "
                            "bundles live in bundles/<bitstream_id>/")
        p.add_argument("--trust-remote", action="store_true",
                       help="download a release that is not on the "
                            "trusted list")

    def model_args(p, onnx=False):
        p.add_argument("--model", default=None, help="checkpoint path")
        p.add_argument("--channel", default=None, help="mean channel .npz")
        p.add_argument("--bundle", default=None, help="bundle directory")
        p.add_argument("--device", default=None,
                       help="default: cuda when torch sees one, else cpu "
                            "(onnx backend: cpu)")
        p.add_argument("--fp32", action="store_true",
                       help="fp32 decoder passes on CUDA (default bf16)")
        p.add_argument("--batch", type=int, default=256)
        if onnx:
            p.add_argument("--backend", choices=["torch", "onnx"],
                           default=None,
                           help="default torch when it imports, else onnx")
            p.add_argument("--onnx-dir", default=None,
                           help="export-onnx directory (backend onnx)")
            p.add_argument("--threads", type=int, default=None,
                           help="onnxruntime threads (default: the CPUs "
                                "this process may use)")
            cache_args(p)

    e = sub.add_parser("encode", help="GeoTIFF tile -> .georvq.parquet")
    e.add_argument("src")
    e.add_argument("out", nargs="?", default=None,
                   help="output file or directory (default: "
                        "{source stem}.georvq.parquet in the current directory)")
    model_args(e)
    e.add_argument("--no-quality", action="store_true",
                   help="skip the per-rung quality column (one decode per rung)")
    e.add_argument("--release", default=RELEASE_NAME,
                   help="release name in the header ('' for no release "
                        "record)")
    e.add_argument("--release-uri", default=RELEASE_URI,
                   help="where the release's bundle is published")
    e.add_argument("--revision", default=None,
                   help="revision of the release's bundle (default: none, "
                        "the repository's default branch)")
    e.add_argument("--source", default=None,
                   help="path or URL of the source tile to record in the "
                        "header (default: SRC as given)")

    d = sub.add_parser("decode", help="window -> .npy (source row order)")
    d.add_argument("src")
    d.add_argument("out")
    model_args(d, onnx=True)
    d.add_argument("--window", default=None, help="row,col,height,width")
    d.add_argument("--quality", default=None, help="e.g. 8-8-2 or p99<=0.02")

    r = sub.add_parser("reconstruct", help="file -> GeoTIFF with source profile")
    r.add_argument("src")
    r.add_argument("out")
    model_args(r, onnx=True)
    r.add_argument("--quality", default=None)

    x = sub.add_parser("export-onnx", help="checkpoint + channel -> ONNX dir "
                                           "for the onnx backend")
    x.add_argument("out")
    model_args(x)
    x.add_argument("--opset", type=int, default=18)

    f = sub.add_parser("fetch", help="find or download the bundle of a file "
                                     "or bitstream id; prints its directory")
    f.add_argument("target", help="a v2 file (path or URL) or a bitstream id")
    f.add_argument("--uri", default=RELEASE_URI,
                   help="release to download a bitstream id from")
    f.add_argument("--revision", default=None)
    cache_args(f)

    i = sub.add_parser("info", help="header summary and column sizes")
    i.add_argument("src")
    i.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "info":
        with RVQParquetReader(args.src) as rd:
            h = rd.header
            print(json.dumps({k: h[k] for k in ("format", "version", "status",
                                                "codec", "product")}, indent=1))
            print(f"  windows {rd.window_grid}")
            for kind, g in rd.layout.groups.items():
                print(f"  {kind:<8} {g.chunk_px:>5} px chunks, grid "
                      f"{rd.chunk_grids[kind]}")
            for rr in h["rungs"]:
                print(f"  {'-'.join(map(str, rr['tuple'])):>8}  "
                      f"{rr['bits_window']:>7} b/window  {rr['bits_px']:.4f} b/px")
            for n, b in column_bytes(rd).items():
                print(f"  {n:<10} {b:>12,} B")
            for n, b in rd.kind_bytes().items():
                print(f"  {n + ' rows':<10} {b:>12,} B")
            print(f"  {'file':<10} {rd.size:>12,} B")
            try:
                where = locate_bundle(h, args.cache_dir, download=False)
            except BundleError as err:
                where = f"not found ({err})"
            print(f"  bundle     {where}")
        return 0

    if args.cmd == "fetch":
        try:
            header = _fetch_header(args.target, args.uri, args.revision)
            print(locate_bundle(header, args.cache_dir, args.trust_remote))
        except (BundleError, FileNotFoundError) as err:
            print(err, file=sys.stderr)
            return 1
        return 0

    t0 = time.time()
    if args.cmd in ("decode", "reconstruct"):
        try:
            ds = open_dataset(args.src, model=args.model, channel=args.channel,
                              bundle=args.bundle, onnx_dir=args.onnx_dir,
                              device=args.device, backend=args.backend,
                              bf16=not args.fp32, threads=args.threads,
                              cache_dir=args.cache_dir,
                              trust_remote=args.trust_remote)
        except BundleError as err:
            print(err, file=sys.stderr)
            return 1
        with ds:
            try:
                ds.codec
            except BundleError as err:
                print(err, file=sys.stderr)
                return 1
            if args.cmd == "decode":
                a = ds.read(window=_window_arg(args.window),
                            quality=args.quality, batch=args.batch)
                np.save(args.out, a)
                print(f"{a.shape} in {time.time() - t0:.1f}s, "
                      f"{ds.reader.stats['requests']} requests, "
                      f"{ds.reader.stats['bytes']:,} B")
            else:
                reconstruct(ds, args.out, quality=args.quality)
                print(f"{args.out} in {time.time() - t0:.1f}s")
        return 0

    # the export traces the model on CPU in fp32
    export = args.cmd == "export-onnx"
    device = "cpu" if export else args.device or default_device("torch")
    bf16 = False if export else not args.fp32
    if args.bundle is not None:
        from georvq.ae.v2_bundle import load_bundle
        try:
            with _quiet_build():
                codec = load_bundle(args.bundle, device=device, bf16=bf16)
        except BundleError as err:
            print(err, file=sys.stderr)
            return 1
    elif args.model is not None and args.channel is not None:
        with _quiet_build():
            codec = AEV2Codec.load(args.model, args.channel, device=device,
                                   bf16=bf16)
    else:
        ap.error("--model and --channel, or --bundle, are needed")
    if export:
        meta = export_onnx(codec, args.out, opset=args.opset)
        print(json.dumps({k: meta[k] for k in ("format", "bitstream_id",
                                               "decoder_id")}))
        return 0
    out = args.out
    if out is None or os.path.isdir(out):
        out = os.path.join(out or ".", output_name(args.src))
    t = encode_tile(args.src, out, codec, stats=not args.no_quality,
                    batch=args.batch, release=args.release or None,
                    uri=args.release_uri, revision=args.revision,
                    source=args.source, log=lambda s: print(s, file=sys.stderr))
    print(json.dumps(dict(t, out=out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
