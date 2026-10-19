"""A release codec over georvq Zarr stores, configured by a `ReleaseSpec`.

A release that ships a stats_cond model as georvq Zarr stores
(georvq.inference.records) decides a fixed set of things: the names
written into headers and bundles, the bands, the chunk sizes, the rungs
offered, the quality statistics and the error they measure, how a
product's raw pixels become the model's input, and how the decoded field
becomes output pixels. `ReleaseSpec` holds those decisions, as constants
and code hooks; everything else is the same for every release and lives
here:

- `StoreCodec`, the model and its mean channel (georvq.inference.
  mean_codec) as the store's records: the layout, the header entries,
  encoding windows and blocks of pixels to records, the quality values,
  and decoding records to output pixels;
- `OnnxStoreCodec`, the decode side over an `export_onnx` directory,
  with onnxruntime and numpy only;
- `encode_product`, which writes one product to a store from a
  `ProductSource`, in strips of north-up rows;
- `quality_code`, `quality_value`, `parse_rung` and `rung_bits`.

A store's base record holds the mean channel's fields and every level of
the coarsest tier; each finer tier has its own column per level, and the
optional quality column holds, per rung, one byte per statistic. The
reader of stores is georvq.inference.store_dataset.

This module imports torch only when a torch codec is used.
"""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

from georvq.inference.records import Layout, RecordWriter, span, tier_names

# quality code = round(scale * (log10(e) + offset)), clipped to 0..255
Q_SCALE, Q_OFFSET, Q_FLOOR = 32.0, 6.0, 1e-6
ONNX_CHANNEL_NAME = "mean_channel.npz"


@contextlib.contextmanager
def quiet_build():
    """Silence the UserWarning torch's TransformerEncoder gives for the
    tokenizer's norm_first layers, on every model build."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="enable_nested_tensor",
                                category=UserWarning)
        yield


def _no_grad(fn):
    """torch.no_grad, imported on call: reading with the ONNX backend does
    not import torch."""
    @functools.wraps(fn)
    def run(*a, **kw):
        import torch
        with torch.no_grad():
            return fn(*a, **kw)
    return run


def path_rungs(specs, min_depths) -> list:
    """The rungs of the fill order: each tier filled to its depth in turn
    from the model's minimum depths (georvq.model.codec.path_tuples)."""
    from georvq.model.codec import path_tuples
    return [tuple(r) for r in path_tuples(specs, min_depths)]


# --------------------------------------------------------------------------
# the release's decisions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReleaseSpec:
    """What one release of georvq stores decides.

    Names. `name` is the release name a header's `codec.release` carries
    by default and `release_uri` where its bundle is published (an
    `hf://org/repo` uri); `codec_format` is the header's `codec.format`,
    which `georvq.open` looks the release up by; `product_kind` is the
    header's `product.kind`; `bundle_format` the `format` of the release's
    bundle.json; `onnx_meta_key` the key of its entry in an ONNX export's
    meta.json; `env_bundle` an environment variable that may name a bundle
    directory. `trusted` are patterns over "hf://org/repo" from which a
    bundle is downloaded without `trust_remote`.

    Data. `bands` is the model's band count. `chunk_px` gives the chunk
    side in pixels of `base`, of the tiers between the coarsest and the
    finest, and of the finest. `strip_px` is the height of the strips a
    product is encoded in. `rungs(specs, min_depths)` lists the rungs a
    store offers.

    Raw pixels. A product's pixels reach the codec as a tuple of arrays
    whose last two axes are rows and columns (AE: the int8 bands; Tessera:
    the int8 bands and the float32 scale plane). `pad` is the value each
    array is padded with to whole windows. `raw_valid(arrays)` is the
    (H, W) pixel validity the source's own convention gives, computed on
    the host so that windows without a valid pixel are never encoded, and
    `invalidate(arrays, mask)` returns the arrays with the pixels of
    `mask` made invalid by that convention. `check_raw(arrays, bands)`
    raises ValueError for arrays of the wrong type or shape.
    `prep(model, arrays, device)` turns a batch of windows (each array
    (B, ..., W, W)) into the model's input `(x, valid)` on the device; it
    must agree with `raw_valid`.

    Output. `postprocess(out, valid, **options)` turns the decoded field
    (n, C, W, W), a torch tensor on the codec's device, into output
    pixels as a numpy array of `out_dtype`, with `out_nodata` at invalid
    pixels; `postprocess_np` does the same for a numpy field (the ONNX
    backend). `output_options` are the keyword options both accept, with
    their defaults. `band_names` are the output bands' names, or None.

    Quality. `quality_stats` name the statistics stored per rung, over
    the valid pixels of a window, of the per-pixel error
    `quality_error(out, x)` ((B, C, W, W) decoded field and prepared
    source -> (B, W, W)); "mean" and "p<N>" (a percentile) are known.
    `quality_error_text` and `quality_encoding_text` describe them in the
    header, and `quality_scale`, `quality_offset` and `quality_floor` set
    the log code.

    `check_model(model, channel)` raises ValueError for a model and
    channel the release cannot ship. `torch_codec` and `onnx_codec` are
    the codec classes a resolved bundle is loaded as (StoreCodec and
    OnnxStoreCodec by default). `bundle_readme(meta)` returns the text of
    a bundle's README. `licence` is written into each store's header when
    given."""

    name: str
    codec_format: str
    product_kind: str
    bundle_format: str
    onnx_meta_key: str
    bands: int
    prep: Callable
    raw_valid: Callable
    invalidate: Callable
    postprocess: Callable
    postprocess_np: Callable
    out_dtype: str
    out_nodata: Any
    pad: tuple
    check_raw: Optional[Callable] = None
    release_uri: Optional[str] = None
    trusted: tuple = ("hf://earth-compress/*",)
    env_bundle: Optional[str] = None
    chunk_px: tuple = (8192, 8192, 2048)
    strip_px: int = 1024
    rungs: Callable = path_rungs
    quality_stats: tuple = ("mean", "p99")
    quality_error: Optional[Callable] = None
    quality_error_text: str = ""
    quality_encoding_text: str = ("code = round(32 * (log10(e) + 6)) clipped "
                                  "to 0..255; 0 means e <= 1e-6")
    quality_scale: float = Q_SCALE
    quality_offset: float = Q_OFFSET
    quality_floor: float = Q_FLOOR
    output_options: dict = field(default_factory=dict)
    band_names: Optional[tuple] = None
    check_model: Optional[Callable] = None
    torch_codec: Optional[type] = None
    onnx_codec: Optional[type] = None
    bundle_readme: Optional[Callable] = None
    licence: Optional[dict] = None

    def with_(self, **kw) -> "ReleaseSpec":
        return dataclasses.replace(self, **kw)

    def chunk_sizes(self, n_tiers: int) -> dict:
        """Chunk side in pixels per kind of a store."""
        names = tier_names(n_tiers)
        base, mid, fine = self.chunk_px
        out = {"base": base}
        out.update({n: mid for n in names[1:-1]})
        if n_tiers > 1:
            out[names[-1]] = fine
        return out

    def output_header(self, codec) -> dict:
        """The header's `output`: what reading the store returns."""
        names = self.band_names
        return {"dtype": self.out_dtype, "nodata": self.out_nodata,
                "count": int(codec.bands),
                "band_names": None if names is None else list(names)}

    def options(self, given: dict) -> dict:
        """Output options: the defaults updated by `given`, refusing an
        option the release does not have."""
        bad = sorted(set(given) - set(self.output_options))
        if bad:
            raise TypeError(f"{self.name}: unknown output option(s) {bad}; "
                            f"this release takes {sorted(self.output_options)}")
        return {**self.output_options, **given}


# --------------------------------------------------------------------------
# layout, rates and quality codes
# --------------------------------------------------------------------------

def index_bits(codes: int) -> int:
    b = int(codes).bit_length() - 1
    if codes != 1 << b:
        raise ValueError(f"codebook size {codes} is not a power of two")
    return b


def layout(specs, window: int, channel, n_rungs: int = 0, chunk_px=None,
           quality_stats=("mean", "p99")) -> Layout:
    """Columns of a store. `n_rungs` 0 leaves out the quality column. The
    coarsest tier's levels follow the mean channel's fields in `base`.
    chunk_px: per-kind chunk sizes."""
    names = tier_names(len(specs))
    sp0 = specs[0]
    base = list(channel.record_fields()) + [
        (names[0], sp0.cells * sp0.depth, index_bits(sp0.codes))]
    tiers = [(names[0], sp0.depth, None)] + [
        (name, sp.depth, [(name, sp.cells, index_bits(sp.codes))])
        for name, sp in zip(names[1:], specs[1:])]
    quality = [(st, n_rungs, 8) for st in quality_stats] if n_rungs else None
    return Layout(window, chunk_px, base, tiers, quality)


def rung_bits(specs, depths, channel) -> int:
    """Bits per window at a depth tuple, the mean record included."""
    return int(channel.cost_bits()) + sum(
        sp.cells * int(L) * index_bits(sp.codes) for sp, L in zip(specs, depths))


def quality_code(e, scale=Q_SCALE, offset=Q_OFFSET, floor=Q_FLOOR) -> np.ndarray:
    """Error e -> uint8 code round(scale * (log10(e) + offset)), clipped to
    0..255; 0 means e <= floor. NaN (a window with no valid pixels) codes
    as 0."""
    e = np.asarray(e, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.rint(scale * (np.log10(np.maximum(e, 1e-30)) + offset))
    c = np.clip(np.nan_to_num(c, nan=0.0), 0, 255)
    c[~(e > floor)] = 0
    return c.astype(np.uint8)


def quality_value(code, scale=Q_SCALE, offset=Q_OFFSET) -> np.ndarray:
    """The error a code stands for (the floor for code 0)."""
    return 10.0 ** (np.asarray(code, dtype=np.float64) / scale - offset)


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


def _stat_of(e, stat, dim=1):
    """Statistic `stat` ("mean" or "p<N>") of e over `dim`, NaN ignored."""
    import torch
    if stat == "mean":
        return e.nanmean(dim)
    m = re.fullmatch(r"p(\d+(?:\.\d+)?)", stat)
    if m is None:
        raise ValueError(f"quality stat {stat!r}: expected 'mean' or 'p<N>'")
    return torch.nanquantile(e, float(m.group(1)) / 100, dim=dim)


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
        """Add the region's windows to a RecordWriter, its top-left
        window at (window_row, window_col) of the product's grid."""
        writer.add_windows(window_row, window_col, self.dense(), self.valid)


def _as_arrays(raw) -> tuple:
    return tuple(raw) if isinstance(raw, (tuple, list)) else (raw,)


class StoreCodec:
    """A stats_cond model with its mean channel, as the records of a
    release's stores. Encoder passes run in fp32; decoder passes run in
    bf16 on CUDA unless `bf16=False` (on CPU always fp32). `spec` defaults
    to the class's own (a release's subclass sets it)."""

    spec: ReleaseSpec = None

    def __init__(self, model, channel, bf16=None, rungs=None, spec=None):
        from georvq.inference.api import Codec
        from georvq.inference.release import bitstream_id, decoder_id
        self.spec = spec = spec or type(self).spec
        if spec is None:
            raise ValueError("a StoreCodec needs a ReleaseSpec")
        self.model = model.eval()
        self.channel = channel
        self.device = next(model.parameters()).device
        if spec.check_model is not None:
            spec.check_model(model, channel)
        if channel.bands != model.bands:
            raise ValueError(f"mean channel has {channel.bands} bands, model "
                             f"{model.bands}")
        self.codec = Codec(model)
        self.specs = list(model.specs)
        self.window = int(model.window)
        self.bands = int(model.bands)
        self.full = tuple(sp.depth for sp in self.specs)
        self.rungs = [tuple(r) for r in (rungs or spec.rungs(
            self.specs, model.cfg.min_depths))]
        cuda = self.device.type == "cuda"
        self.bf16 = cuda if bf16 is None else (bool(bf16) and cuda)
        self.names = tier_names(len(self.specs))
        self.n_mean = len(channel.field_widths())
        self.bitstream_id = bitstream_id(model, extra=channel.id_state())
        self.decoder_id = decoder_id(model)

    @classmethod
    def load(cls, model, channel, device="cpu", bf16=None, spec=None):
        """model: a checkpoint path or a model; channel: a .npz path or a
        channel."""
        from georvq.inference.mean_codec import load_channel
        if isinstance(model, (str, os.PathLike)):
            from georvq.factory import from_checkpoint
            model = from_checkpoint(os.fspath(model), device)
        if isinstance(channel, (str, os.PathLike)):
            channel = load_channel(os.fspath(channel))
        return cls(model, channel, bf16=bf16, spec=spec)

    def _ac(self):
        import torch
        return (torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16
                else contextlib.nullcontext())

    def layout(self, quality=True, chunk_px=None) -> Layout:
        return layout(self.specs, self.window, self.channel,
                      len(self.rungs) if quality else 0,
                      chunk_px or self.spec.chunk_sizes(len(self.specs)),
                      self.spec.quality_stats)

    def header_codec(self, release=None) -> dict:
        return {"format": self.spec.codec_format,
                "bitstream_id": self.bitstream_id,
                "decoder_id": self.decoder_id,
                "tiers": [str(sp) for sp in self.specs],
                "window": self.window, "bands": self.bands,
                "min_depths": list(self.model.cfg.min_depths),
                "mean_channel": {"kind": self.channel.kind,
                                 "bits": int(self.channel.cost_bits())},
                "release": release}

    def header_rungs(self) -> list:
        W2 = self.window ** 2
        return [{"tuple": list(r),
                 "bits_window": rung_bits(self.specs, r, self.channel),
                 "bits_px": rung_bits(self.specs, r, self.channel) / W2}
                for r in self.rungs]

    def header_quality(self) -> dict:
        s = self.spec
        return {"column": "quality", "stats": list(s.quality_stats),
                "error": s.quality_error_text,
                "encoding": s.quality_encoding_text,
                "scale": s.quality_scale, "offset": s.quality_offset,
                "order": "per stat, one byte per rung in header rungs order",
                "decode_precision": "bf16" if self.bf16 else "fp32"}

    # ---- encode ----

    @_no_grad
    def encode_windows(self, raw, stats=True):
        """raw: a batch of north-up windows, an array or a tuple of arrays
        each (B, ..., W, W). Returns ({group: (levels, B, n_values)}, x,
        valid) with x, valid the prepared device input."""
        import torch
        dev = self.device
        x, valid = self.spec.prep(self.model, _as_arrays(raw), dev)
        mu = self.model.patch_stats(x, valid).float().cpu().numpy()
        fields = self.channel.encode(mu)
        coded = torch.from_numpy(self.channel.decode(fields)).to(dev)
        stacked, _, _ = self.codec.encode_batch(x, valid=valid, mean=coded)
        B = x.shape[0]
        coarse = stacked[0].transpose(0, 2, 1).reshape(B, -1)
        rec = {"base": np.concatenate([fields, coarse], 1)[None]}
        for name, idx in zip(self.names[1:], stacked[1:]):
            rec[name] = idx.transpose(2, 0, 1)
        if stats:
            rec["quality"] = self._quality(stacked, coded, x, valid)[None]
        return rec, x, valid

    def _quality(self, stacked, mean, x, valid):
        """(B, n_stats * n_rungs) codes: per stat, one per rung."""
        import torch
        s = self.spec
        dev = self.device
        idx = [torch.from_numpy(t).to(dev) for t in stacked]
        cums = [cumulative(rvq, t, sp.depth)
                for rvq, t, sp in zip(self.model.rvqs, idx, self.specs)]
        B = x.shape[0]
        vf = valid.reshape(B, -1)
        per = {st: [] for st in s.quality_stats}
        for r in self.rungs:
            out = self._forward([c[L] for c, L in zip(cums, r)], r, mean, valid)
            e = s.quality_error(out, x).reshape(B, -1)
            e = torch.where(vf, e, torch.full_like(e, float("nan")))
            for st in s.quality_stats:
                per[st].append(_stat_of(e, st))
        codes = [quality_code(torch.stack(per[st], 1).cpu().numpy(),
                              s.quality_scale, s.quality_offset, s.quality_floor)
                 for st in s.quality_stats]
        return np.concatenate(codes, 1).astype(np.int64)

    def _forward(self, zqs, depths, mean, valid):
        import torch
        B = mean.shape[0]
        Ls = torch.tensor(list(depths), device=self.device)[None].expand(B, -1)
        with self._ac():
            out = self.model.decode(zqs, Ls, stats=mean, valid=valid)
        return out.float()

    def encode_block(self, raw, valid=None, stats=True, batch=256) -> BlockRecords:
        """raw: the north-up pixels of a region, an array or a tuple of
        arrays each (..., H, W), as the release's `prep` takes them.
        `valid` (H, W), when given, marks further pixels invalid;
        otherwise validity is the source's own (`spec.raw_valid`)."""
        s = self.spec
        arrs = tuple(np.asarray(a) for a in _as_arrays(raw))
        if s.check_raw is not None:
            s.check_raw(arrs, self.bands)
        H, Wd = arrs[0].shape[-2:]
        W = self.window
        gh, gw = -(-H // W), -(-Wd // W)
        padded = []
        for a, fill in zip(arrs, s.pad):
            p = np.full(a.shape[:-2] + (gh * W, gw * W), fill, a.dtype)
            p[..., :H, :Wd] = a
            padded.append(p)
        padded = tuple(padded)
        if valid is not None:
            v = np.zeros((gh * W, gw * W), bool)
            v[:H, :Wd] = valid
            padded = tuple(s.invalidate(padded, ~v))
        pv = s.raw_valid(padded)
        live = pv.reshape(gh, W, gw, W).any((1, 3))
        rows, cols = np.nonzero(live)
        wins = [p.reshape(p.shape[:-2] + (gh, W, gw, W)) for p in padded]
        parts = []
        for i in range(0, len(rows), batch):
            r, c = rows[i:i + batch], cols[i:i + batch]
            chunk = tuple(np.ascontiguousarray(w[..., r, :, c, :]) for w in wins)
            parts.append(self.encode_windows(
                chunk if len(chunk) > 1 else chunk[0], stats=stats)[0])
        if parts:
            recs = {k: np.concatenate([p[k] for p in parts], 1) for k in parts[0]}
        else:
            recs = {}
        return BlockRecords(rows.astype(np.int64), cols.astype(np.int64), recs,
                            pv[:H, :Wd], (gh, gw))

    # ---- decode ----

    @_no_grad
    def decode_records(self, base, tiers, depths, valid, trace=None, **options):
        """base: (n, n_values) base records; tiers: per finer tier (n,
        levels_read, cells) codes, or None at depth 0; depths: the tuple;
        valid: (n, W, W) bool. -> (n, C, W, W) output pixels of the
        release's dtype, its nodata at invalid pixels. `options` are the
        release's output options. `trace`, a georvq.inference.records.
        Trace, records the decode stages; on CUDA each stage then waits
        for the device before it ends."""
        out, v = self.decode_field(base, tiers, depths, valid, trace)
        opts = self.spec.options(options)
        with span(trace, "decode", "postprocess", sync=self._sync(trace)):
            return self.spec.postprocess(out, v, **opts)

    def _sync(self, trace):
        if trace is None or self.device.type != "cuda":
            return None
        import torch
        return torch.cuda.synchronize

    @_no_grad
    def decode_field(self, base, tiers, depths, valid, trace=None):
        """As decode_records, but returns the decoded float field (n, C,
        W, W) on the codec's device before postprocessing, and the
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
                zqs.append(cumulative(rvq, i, int(L))[int(L)])
            v = torch.from_numpy(np.ascontiguousarray(valid)).to(dev)
        with span(trace, "decode", "decoder forward", sync=sync):
            return self._forward(zqs, depths, mean, v), v


def cumulative(rvq, idx, L):
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


def make_codec(spec, model, channel, bf16=None):
    """The release's torch codec class (`spec.torch_codec`, StoreCodec by
    default) for a model and its channel."""
    cls = spec.torch_codec or StoreCodec
    return cls(model, channel, bf16=bf16, spec=spec)


# --------------------------------------------------------------------------
# ONNX
# --------------------------------------------------------------------------

def channel_sha(channel) -> str:
    h = hashlib.sha256()
    for a in channel.id_state():
        h.update(np.asarray(a).tobytes())
    return h.hexdigest()[:16]


def export_onnx(codec: StoreCodec, out_dir, opset=18) -> dict:
    """Export a StoreCodec for `OnnxStoreCodec`: the stats_cond graphs and
    codebooks (georvq.inference.onnx_stats) and the mean channel's
    tables, under the codec's bitstream id. Returns the meta dict."""
    from georvq.inference.onnx_stats import export_stats_onnx
    spec = codec.spec
    os.makedirs(out_dir, exist_ok=True)
    codec.channel.save(os.path.join(out_dir, ONNX_CHANNEL_NAME))
    extra = {spec.onnx_meta_key: {
        "format": spec.codec_format, "mean_channel": ONNX_CHANNEL_NAME,
        "mean_channel_sha": channel_sha(codec.channel),
        "min_depths": list(codec.model.cfg.min_depths),
        "rungs": [list(r) for r in codec.rungs]}}
    meta = export_stats_onnx(codec.model, out_dir,
                             extra_id=codec.channel.id_state(),
                             meta_extra=extra, opset=opset)
    if meta["bitstream_id"] != codec.bitstream_id:
        raise RuntimeError(f"export bitstream id {meta['bitstream_id']} != "
                           f"codec {codec.bitstream_id}")
    return meta


class OnnxStoreCodec:
    """Decode side of a StoreCodec over an `export_onnx` directory, with
    onnxruntime and numpy only. Decoding runs in fp32 on any device, so
    on CPU it agrees with the torch fp32 decode up to float rounding in
    the graphs."""

    spec: ReleaseSpec = None

    def __init__(self, onnx_dir, device="cpu", threads=None, spec=None):
        from georvq.config import TierSpec
        from georvq.inference.mean_codec import load_channel
        from georvq.inference.onnx_stats import OnnxStatsCodec
        self.spec = spec = spec or type(self).spec
        if spec is None:
            raise ValueError("an OnnxStoreCodec needs a ReleaseSpec")
        self.ort = OnnxStatsCodec(onnx_dir, device=device, threads=threads)
        meta = self.ort.meta
        rel = meta.get(spec.onnx_meta_key)
        if rel is None or rel.get("format") != spec.codec_format:
            raise ValueError(f"{onnx_dir}: not an ONNX export of "
                             f"{spec.codec_format}")
        self.channel = load_channel(os.path.join(onnx_dir, rel["mean_channel"]))
        if channel_sha(self.channel) != rel["mean_channel_sha"]:
            raise ValueError(f"{onnx_dir}: mean channel tables do not match "
                             "the export's record of them")
        self.specs = [TierSpec(**t) for t in meta["tiers"]]
        self.window = int(meta["window"])
        self.bands = int(meta["bands"])
        self.full = tuple(sp.depth for sp in self.specs)
        self.rungs = [tuple(r) for r in rel["rungs"]]
        self.names = tier_names(len(self.specs))
        self.n_mean = len(self.channel.field_widths())
        self.bitstream_id = meta["bitstream_id"]
        self.decoder_id = meta["decoder_id"]
        self.bf16 = False

    def decode_records(self, base, tiers, depths, valid, trace=None, **options):
        """As StoreCodec.decode_records. The trace records the mean
        channel, the ONNX graphs (codebook lookup and decoder together)
        and postprocessing."""
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
        opts = self.spec.options(options)
        with span(trace, "decode", "postprocess"):
            return self.spec.postprocess_np(out, valid, **opts)


def make_onnx_codec(spec, onnx_dir, device="cpu", threads=None):
    cls = spec.onnx_codec or OnnxStoreCodec
    return cls(onnx_dir, device=device, threads=threads, spec=spec)


# --------------------------------------------------------------------------
# writing a product
# --------------------------------------------------------------------------

class ProductSource(Protocol):
    """A source data product as `encode_product` reads it.

    height, width: the product's size in pixels. transform: the north-up
    affine (a, b, c, d, e, f) of its pixel grid. crs: a rasterio CRS, or
    None. orientation: "north-up", or "bottom-up" for a product stored
    with its rows south to north (readers then return rows in that
    order). product_id: the product's id, unique in its collection.
    source: the path or URL recorded as the header's `product.source`.
    product_meta: further entries of the header's `product` describing
    the source data (count, dtype, nodata, ...). aux: JSON-able source
    metadata stored under `georvq_aux`, or None. `read_strip(y0, y1)`
    returns north-up rows y0..y1 of every column as the codec's raw
    arrays."""

    height: int
    width: int
    transform: Any
    crs: Any
    orientation: str
    product_id: str
    source: str
    product_meta: dict
    aux: Any

    def read_strip(self, y0: int, y1: int):
        ...


def projjson(crs):
    if crs is None:
        return None
    try:
        return crs.to_dict(projjson=True)
    except (TypeError, AttributeError):
        from pyproj import CRS
        return CRS.from_wkt(crs.to_wkt()).to_json_dict()


def release_record(codec, name, uri, revision=None) -> dict | None:
    """The header's `codec.release` for a store coded by `codec`; None
    when `name` is None."""
    if name is None:
        return None
    return {"name": name, "bitstream_id": codec.bitstream_id,
            "decoder_id": codec.decoder_id, "uri": uri,
            "revision": revision}


def product_header(src: ProductSource, kind) -> dict:
    crs = src.crs
    return {"kind": kind, "id": str(src.product_id), "source": src.source,
            "crs_wkt": crs.to_wkt() if crs else None,
            "epsg": crs.to_epsg() if crs else None,
            "transform": [float(v) for v in list(src.transform)[:6]],
            "width": int(src.width), "height": int(src.height),
            **src.product_meta, "orientation": src.orientation}


def encode_product(src: ProductSource, out_path, codec: StoreCodec,
                   stats=True, chunk_px=None, strip_px=None, batch=256,
                   release=None, uri=None, revision=None, assets=None,
                   log=None) -> dict:
    """Encode one product to the Zarr store `out_path`, replacing a store
    already there once the new one is written. Strips of `strip_px` rows
    (the release's by default) are encoded in turn. chunk_px: per-kind
    chunk sizes, the release's by default. `release`, `uri` and
    `revision` are written to the header's `codec.release`;
    `release=None` writes none. `assets` ({name: (bytes, media_type,
    filename)}) are stored in the store's `assets` group. Returns
    timings."""
    spec = codec.spec
    W = codec.window
    strip_px = strip_px or spec.strip_px
    if strip_px % W:
        raise ValueError(f"strip_px {strip_px} is not a multiple of {W}")
    t0 = time.time()
    H, Wd = int(src.height), int(src.width)
    header = {"codec": codec.header_codec(
                  release_record(codec, release, uri, revision)),
              "product": product_header(src, spec.product_kind),
              "rungs": codec.header_rungs()}
    if stats:
        header["quality"] = codec.header_quality()
    header["output"] = spec.output_header(codec)
    if spec.licence is not None:
        header["licence"] = dict(spec.licence)
    lay = codec.layout(quality=stats, chunk_px=chunk_px)
    aux = src.aux
    w = RecordWriter(out_path, header, lay, src.product_id,
                     list(src.transform)[:6], H, Wd, crs=projjson(src.crs),
                     aux=aux if aux is None or isinstance(aux, str)
                     else json.dumps(aux, default=str), assets=assets)
    t_read = t_enc = t_write = 0.0
    for y0 in range(0, H, strip_px):
        y1 = min(H, y0 + strip_px)
        ta = time.time()
        a = src.read_strip(y0, y1)
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
