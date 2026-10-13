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

The v1 release (georvq.ae.cli, georvq.ae.cog) is separate and unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time

import numpy as np
import torch

from georvq.inference.api import Codec
from georvq.inference.georvq_parquet import (Layout, RVQParquetReader,
                                             RVQParquetWriter, output_name,
                                             tier_names)
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

    @torch.no_grad()
    def encode_windows(self, raw, stats=True):
        """raw: (B, C, W, W) int8 north-up windows. Returns ({group:
        (levels, B, n_values)}, x, valid) with x, valid the prepared
        device input."""
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

    @torch.no_grad()
    def decode_records(self, base, tiers, depths, valid):
        """base: (n, n_values) base records; tiers: per finer tier (n,
        levels_read, cells) codes, or None at depth 0; depths: the tuple; valid: (n, W, W)
        bool. -> (n, C, W, W) int8 with invalid pixels at -128."""
        dev = self.device
        n = base.shape[0]
        sp0 = self.specs[0]
        mean = torch.from_numpy(
            self.channel.decode(base[:, :self.n_mean])).to(dev)
        coarse = base[:, self.n_mean:].reshape(n, sp0.depth, sp0.cells)
        idx = [coarse.transpose(0, 2, 1)] + [
            None if t is None else t.transpose(0, 2, 1) for t in tiers]
        zqs = []
        for rvq, sp, i, L in zip(self.model.rvqs, self.specs, idx, depths):
            if int(L) == 0:
                zqs.append(torch.zeros(n, sp.cells, sp.dim, device=dev))
                continue
            i = torch.from_numpy(np.ascontiguousarray(i)).to(dev)
            zqs.append(_cumulative(rvq, i, int(L))[int(L)])
        v = torch.from_numpy(np.ascontiguousarray(valid)).to(dev)
        out = self._forward(zqs, depths, mean, v)
        return requant(out, v)


def requant(out, valid):
    """Decoded float field (n, C, W, W) -> int8, the signed-sqrt inverse of
    the AE dequantization, with invalid pixels at -128. On CUDA it runs
    on the device, where sqrt is correctly rounded; on CPU it runs in
    numpy, because torch's vectorized CPU sqrt is not correctly rounded
    and would differ from `georvq.ae.cli.requant_ae_np` in rare cases."""
    if out.device.type != "cuda":
        from georvq.ae.cli import requant_ae_np
        return requant_ae_np(out.cpu().numpy(), valid.cpu().numpy())
    q = torch.clamp(torch.round(torch.sign(out) * torch.sqrt(out.abs())
                                * 127.5), -127, 127).to(torch.int8)
    q = torch.where(valid[:, None], q, torch.full_like(q, NODATA))
    return q.cpu().numpy()


def _cumulative(rvq, idx, L):
    """Partial codebook sums at depths 0..L for (B, cells, >=L) indices,
    added level by level in the order Codec.decode adds them."""
    B, cells = idx.shape[:2]
    out = [torch.zeros(B, cells, rvq.levels[0].dim, device=idx.device)]
    z = None
    for lev in range(L):
        v = rvq.levels[lev].codebook(idx[..., lev])
        z = v if z is None else z + v
        out.append(z)
    return out


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
                chunk_px=None, strip_px=STRIP_PX, batch=256, release=None,
                log=None) -> dict:
    """Encode one AE tile (GeoTIFF) to `out_path`, conventionally
    `output_name(src_path)`. Strips of `strip_px` rows are encoded in
    turn. chunk_px: per-kind chunk sizes, `chunk_sizes` by default.
    Returns timings."""
    import rasterio
    from rasterio.windows import Window
    W = codec.window
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
            "codec": codec.header_codec(release),
            "product": {
                "kind": KIND, "id": pid,
                "source": os.fspath(src_path),
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

class AEV2Dataset:
    """Read-only, rasterio-like view of an AE v2 file.

    Georeferencing, shape and `read` follow the SOURCE tile, including a
    bottom-up row order, so code written against the source dataset can
    read this instead. `read` needs a codec; the metadata does not."""

    def __init__(self, path_or_url, codec: AEV2Codec | None = None, **reader_kw):
        self.reader = RVQParquetReader(path_or_url, **reader_kw)
        self.name = self.reader.source
        h = self.header = self.reader.header
        p = self.product = h["product"]
        if p.get("kind") != KIND:
            raise ValueError(f"{self.name}: product kind {p.get('kind')!r}, "
                             f"not {KIND}")
        self.codec = codec
        if codec is not None and codec.bitstream_id != h["codec"]["bitstream_id"]:
            raise ValueError(f"{self.name}: bitstream {h['codec']['bitstream_id']}"
                             f", codec holds {codec.bitstream_id}")
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

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- quality ----

    def _depth_map(self, quality, wy0, wy1, wx0, wx1) -> np.ndarray:
        """(rows, cols) rung index per window, or -1 for full depth when
        the full tuple is not a rung."""
        shape = (wy1 - wy0, wx1 - wx0)
        if quality is None:
            t = self.full
        else:
            m = _BOUND.match(quality) if isinstance(quality, str) else None
            if m is None:
                t = parse_rung(quality, self.full)
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
        """Source-pixel window -> north-up pixel bounds (y0, y1, x0, x1)."""
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

    def read(self, indexes=None, window=None, quality=None, batch=256):
        """int8 (bands, rows, cols), or (rows, cols) for an int `indexes`
        (1-based, as rasterio). `quality`: a rung ("8-8-2" or a tuple),
        a bound on a quality stat ("p99<=0.02", resolved per window to
        the cheapest rung meeting it, full depth where none does), or
        None for full depth."""
        if self.codec is None:
            raise ValueError("reading pixels needs a codec: "
                             "open(path, model=..., channel=...)")
        W = self.window_px
        ny0, ny1, nx0, nx1 = self._north_px(window)
        wy0, wy1 = ny0 // W, -(-ny1 // W)
        wx0, wx1 = nx0 // W, -(-nx1 // W)
        gh, gw = wy1 - wy0, wx1 - wx0
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
                    dec = self.codec.decode_records(base[y, x], fin, r,
                                                    vwin[y, x])
                    for d, yy, xx in zip(dec, y, x):
                        out[:, yy * W:(yy + 1) * W, xx * W:(xx + 1) * W] = d
        out = out[:, ny0 - wy0 * W:ny1 - wy0 * W, nx0 - wx0 * W:nx1 - wx0 * W]
        if self.bottom_up:
            out = out[:, ::-1]
        if indexes is None:
            return np.ascontiguousarray(out)
        if isinstance(indexes, (int, np.integer)):
            return np.ascontiguousarray(out[int(indexes) - 1])
        return np.ascontiguousarray(out[[int(i) - 1 for i in indexes]])

    def read_masks(self, window=None) -> np.ndarray:
        """Validity in the source's row order, (rows, cols) bool."""
        ny0, ny1, nx0, nx1 = self._north_px(window)
        m = self.reader.read_mask(ny0, ny1, nx0, nx1)
        return m[::-1] if self.bottom_up else m


def open_dataset(path_or_url, model=None, channel=None, device="cpu",
                 bf16=None, codec=None, **reader_kw) -> "AEV2Dataset":
    """Open an AE v2 file (a local path or an http(s) URL). `model` (a checkpoint path or a model)
    and `channel` (a .npz path or a channel) are needed to read pixels;
    `codec` passes an AEV2Codec instead. `reader_kw` go to
    RVQParquetReader (`first_read`, `gap`, `concurrency`, `cache`,
    `cache_bytes`, `cache_dir`)."""
    if codec is None and model is not None:
        if channel is None:
            raise ValueError("a model needs its mean channel")
        codec = AEV2Codec.load(model, channel, device=device, bf16=bf16)
    return AEV2Dataset(path_or_url, codec, **reader_kw)


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


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m georvq.ae.v2",
        description="AlphaEarth v2 codec: one annual tile <-> one "
                    "georvq GeoParquet file.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p):
        p.add_argument("--model", required=True, help="checkpoint path")
        p.add_argument("--channel", required=True, help="mean channel .npz")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                       else "cpu")
        p.add_argument("--fp32", action="store_true",
                       help="fp32 decoder passes on CUDA (default bf16)")
        p.add_argument("--batch", type=int, default=256)

    e = sub.add_parser("encode", help="GeoTIFF tile -> .georvq.parquet")
    e.add_argument("src")
    e.add_argument("out", nargs="?", default=None,
                   help="output file or directory (default: "
                        "{source stem}.georvq.parquet in the current directory)")
    model_args(e)
    e.add_argument("--no-quality", action="store_true",
                   help="skip the per-rung quality column (one decode per rung)")
    e.add_argument("--release", default=None)

    d = sub.add_parser("decode", help="window -> .npy (source row order)")
    d.add_argument("src")
    d.add_argument("out")
    model_args(d)
    d.add_argument("--window", default=None, help="row,col,height,width")
    d.add_argument("--quality", default=None, help="e.g. 8-8-2 or p99<=0.02")

    r = sub.add_parser("reconstruct", help="file -> GeoTIFF with source profile")
    r.add_argument("src")
    r.add_argument("out")
    model_args(r)
    r.add_argument("--quality", default=None)

    i = sub.add_parser("info", help="header summary and column sizes")
    i.add_argument("src")
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
        return 0

    codec = AEV2Codec.load(args.model, args.channel, device=args.device,
                           bf16=not args.fp32)
    t0 = time.time()
    if args.cmd == "encode":
        out = args.out
        if out is None or os.path.isdir(out):
            out = os.path.join(out or ".", output_name(args.src))
        t = encode_tile(args.src, out, codec, stats=not args.no_quality,
                        batch=args.batch, release=args.release,
                        log=lambda s: print(s, file=sys.stderr))
        print(json.dumps(dict(t, out=out)))
    elif args.cmd == "decode":
        with AEV2Dataset(args.src, codec) as ds:
            a = ds.read(window=_window_arg(args.window), quality=args.quality,
                        batch=args.batch)
            np.save(args.out, a)
            print(f"{a.shape} in {time.time() - t0:.1f}s, "
                  f"{ds.reader.stats['requests']} requests, "
                  f"{ds.reader.stats['bytes']:,} B")
    elif args.cmd == "reconstruct":
        with AEV2Dataset(args.src, codec) as ds:
            reconstruct(ds, args.out, quality=args.quality)
        print(f"{args.out} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
