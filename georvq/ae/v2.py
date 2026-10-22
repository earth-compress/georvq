"""AlphaEarth v2 release layer: annual tiles as georvq Zarr stores.

An AE v2 store holds one AlphaEarth annual tile (64-band int8, nodata
-128) coded by a `stats_cond` model whose per-window mean travels through
an `RVQPCAMeanChannel`, as the arrays of georvq.inference.records, with
a chunk size per kind:

- `base`, one 8192 px chunk (the whole tile): per window the mean
  channel's fields and every level of the coarsest tier; `mask`, the
  tile's pixel validity (north-up), is chunked as base;
- `mid`, 8192 px chunks: levels 1 to 8, one inner chunk per level of a
  chunk;
- `fine`, 2048 px chunks: levels 1 to 8;
- `quality/<decoder_id>`, optional, chunked as base: per window and per
  rung, the mean and p99 of per-pixel `1 - cos` measured at encode time
  with that decoder.

A codec with other than three tiers names them `tier0` and up, the
finest taking the fine chunk size and the others the mid one.

This module holds the release's decisions as `AE_V2`, a
georvq.inference.store_codec.ReleaseSpec: the names, chunk sizes and
quality definition above, the AE source's prep, validity (band 0 is not
-128) and the signed-sqrt int8 requantisation of the decoded field. The
codec, the store writer and the reader are the shared ones; this module
adds the GeoTIFF tile reader (`GeoTiffTile`), `encode_tile`, an
`open_dataset` that takes AE codec arguments, and the command line.

The source's profile, tags and orientation are JSON in the store's
`georvq_aux` attribute. Coded data is north-up. A bottom-up source
(transform.e > 0, as the AEF tiles are stored) is flipped on encode, and
the dataset's `read` returns rows in the source's own order, so it can
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

The header names the release a store was coded with (`codec.release`),
so `georvq.open(path)` finds the bundle that decodes it, in the
environment, the local cache, or by downloading it from a trusted
release (georvq.ae.v2_resolve). The bundle is located when pixels are
first read, so reading metadata needs none.

"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

from georvq.inference import store_codec as _sc
from georvq.inference.records import RecordReader, output_name
from georvq.inference.store_codec import (BlockRecords,  # noqa: F401
                                          OnnxStoreCodec, ReleaseSpec,
                                          StoreCodec, cumulative,
                                          export_onnx, parse_rung,
                                          quiet_build as _quiet_build)
from georvq.inference.store_dataset import (StoreDataset, flip_rows,
                                            reconstruct)  # noqa: F401

NODATA = -128
KIND = "aef-annual-tile"
FORMAT = "georvq-ae/v2"
BUNDLE_FORMAT = "georvq-ae/v2-bundle"
RELEASE_NAME = "ae-v2"
RELEASE_URI = "hf://earth-compress/georvq-ae-v2"
# patterns over "hf://<org>/<repo>"; a release outside them is downloaded
# only with trust_remote=True
TRUSTED_RELEASES = ("hf://earth-compress/*",)
ENV_BUNDLE = "GEORVQ_AE_BUNDLE"
BASE_CHUNK_PX, MID_CHUNK_PX, FINE_CHUNK_PX = 8192, 8192, 2048
STRIP_PX = 1024
QUALITY_STATS = ("mean", "p99")
# quality code = round(Q_SCALE * (log10(e) + Q_OFFSET)), clipped to 0..255
Q_SCALE, Q_OFFSET, Q_FLOOR = _sc.Q_SCALE, _sc.Q_OFFSET, _sc.Q_FLOOR
ONNX_CHANNEL_NAME = _sc.ONNX_CHANNEL_NAME

_GDAL_ENV = {"GDAL_NUM_THREADS": "ALL_CPUS"}
TRAINING_DATA = ("the AlphaEarth Foundations Satellite Embedding dataset "
                 "(CC-BY 4.0), produced by Google and Google DeepMind")
LICENCE = {
    "spdx": "CC-BY-4.0",
    "url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": "The AlphaEarth Foundations Satellite Embedding dataset "
                   "is produced by Google and Google DeepMind.",
    "changes": "lossy georvq encoding of the int8 embeddings; the decoded "
               "values are approximations",
}


# --------------------------------------------------------------------------
# the release's hooks
# --------------------------------------------------------------------------

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


def _prep(model, arrays, device):
    import torch
    return model.source.prep(
        torch.from_numpy(np.ascontiguousarray(arrays[0])), device)


def _raw_valid(arrays):
    return arrays[0][0] != NODATA


def _invalidate(arrays, mask):
    a = arrays[0]
    a[:, mask] = NODATA
    return (a,)


def _check_raw(arrays, bands):
    raw = arrays[0]
    if len(arrays) != 1 or raw.dtype != np.int8 or raw.ndim != 3 \
            or raw.shape[0] != bands:
        raise ValueError(f"expected int8 ({bands}, H, W), got "
                         f"{raw.dtype} {raw.shape}")


def _check_model(model, channel):
    sc = model.stats_cond
    if sc is None or not sc.subtract_mean or sc.use_std:
        raise ValueError("AE v2 needs a stats_cond model with a "
                         "transmitted mean and no std")


def _cos_error(out, x):
    return (1.0 - (out * x).sum(1)).clamp_min(0.0)


def _bundle_readme(meta) -> str:
    from georvq.inference.bundle import bundle_readme
    return bundle_readme(
        meta, "georvq AE v2 bundle",
        "Decoder and encoder for AlphaEarth Foundations annual embedding "
        "tiles stored as georvq\nZarr stores "
        f"(`{meta['codec_format']}`, see docs/GEORVQ_ZARR.md).",
        """
import georvq
with georvq.open("tile.georvq.zarr") as ds:     # finds this bundle
    a = ds.read(quality="8-8-2")              # int8 (64, H, W)

from georvq.ae import v2
from georvq.ae.v2_bundle import load_bundle
codec = load_bundle("bundle_dir", device="cuda")
v2.encode_tile("tile.tif", "tile.georvq.zarr", codec)
""")


AE_V2 = ReleaseSpec(
    name=RELEASE_NAME, codec_format=FORMAT, product_kind=KIND,
    bundle_format=BUNDLE_FORMAT, onnx_meta_key="ae_v2", bands=64,
    prep=_prep, raw_valid=_raw_valid, invalidate=_invalidate,
    check_raw=_check_raw, pad=(NODATA,),
    postprocess=requant, postprocess_np=requant_ae_np,
    out_dtype="int8", out_nodata=NODATA,
    release_uri=RELEASE_URI, trusted=TRUSTED_RELEASES, env_bundle=ENV_BUNDLE,
    chunk_px=(BASE_CHUNK_PX, MID_CHUNK_PX, FINE_CHUNK_PX), strip_px=STRIP_PX,
    quality_stats=QUALITY_STATS, quality_error=_cos_error,
    quality_name="1 - cos",
    quality_error_text="1 - cos(decoded, source) per valid pixel, decoded "
                       "float field against the prepared source",
    quality_encoding_text="code = round(32 * (log10(e) + 6)) clipped to "
                          "0..255; 0 means e <= 1e-6",
    check_model=_check_model, bundle_readme=_bundle_readme, licence=LICENCE,
    training_data=TRAINING_DATA)


class AEV2Codec(StoreCodec):
    """A stats_cond AE model with its mean channel, as used by the v2
    stores (georvq.inference.store_codec.StoreCodec under AE_V2). Encoder
    passes run in fp32; decoder passes run in bf16 on CUDA unless
    `bf16=False` (on CPU always fp32)."""

    spec = AE_V2

    def __init__(self, model, channel, bf16=None, rungs=None, spec=None):
        super().__init__(model, channel, bf16=bf16, rungs=rungs,
                         spec=spec or AE_V2)


class AEV2OnnxCodec(OnnxStoreCodec):
    """Decode side of AEV2Codec over an `export_onnx` directory, with
    onnxruntime and numpy only. Decoding runs in fp32 on any device, so
    on CPU it agrees with the torch fp32 decode up to float rounding in
    the graphs, which can move a rare int8 value by one step."""

    spec = AE_V2

    def __init__(self, onnx_dir, device="cpu", threads=None, spec=None):
        super().__init__(onnx_dir, device=device, threads=threads,
                         spec=spec or AE_V2)


AE_V2 = AE_V2.with_(torch_codec=AEV2Codec, onnx_codec=AEV2OnnxCodec)
AEV2Codec.spec = AEV2OnnxCodec.spec = AE_V2
SPEC = AE_V2


# --------------------------------------------------------------------------
# layout, rates and quality codes
# --------------------------------------------------------------------------

def mean_fields(channel) -> list:
    """The mean channel's record fields (`channel.record_fields()`)."""
    return channel.record_fields()


def chunk_sizes(n_tiers: int) -> dict:
    """Chunk side in pixels per kind of an AE v2 store."""
    return AE_V2.chunk_sizes(n_tiers)


def layout(specs, window: int, channel, n_rungs: int = 0, chunk_px=None):
    """Columns of an AE v2 store. `n_rungs` 0 leaves out the quality
    column. The coarsest tier's levels follow the mean in `base`.
    chunk_px: per-kind chunk sizes, `chunk_sizes` by default."""
    return _sc.layout(specs, window, channel, n_rungs,
                      chunk_px or chunk_sizes(len(specs)), QUALITY_STATS)


rung_bits = _sc.rung_bits


def quality_code(e) -> np.ndarray:
    """Error e = 1 - cos -> uint8 code; 0 means e <= 1e-6. NaN (a window
    with no valid pixels) codes as 0."""
    return _sc.quality_code(e, Q_SCALE, Q_OFFSET, Q_FLOOR)


def quality_value(code) -> np.ndarray:
    """The error a code stands for (1e-6 for code 0)."""
    return _sc.quality_value(code, Q_SCALE, Q_OFFSET)


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


def _header_crs(crs):
    """The CRS a store header records for a source CRS: the one
    `crs_from_epsg` builds for its EPSG code when the two are equal, as
    every release's headers do, and the source's own otherwise."""
    if crs is None:
        return None
    epsg = crs.to_epsg()
    if epsg is not None:
        built = _sc.crs_from_epsg(epsg)
        if built == crs:
            return built
    return crs


class GeoTiffTile:
    """An AE tile, an open rasterio dataset, as a
    georvq.inference.store_codec.ProductSource: north-up strips of its
    int8 bands, whatever its row order. `source` is the path or URL the
    header records."""

    def __init__(self, ds, src_path, source=None, bands=64):
        if ds.count != bands or ds.dtypes[0] != "int8":
            raise ValueError(f"{src_path}: {ds.count} x {ds.dtypes[0]}, "
                             f"expected {bands} x int8")
        if ds.nodata is not None and int(ds.nodata) != NODATA:
            raise ValueError(f"{src_path}: nodata {ds.nodata}, expected {NODATA}")
        self.ds = ds
        self.height, self.width = ds.height, ds.width
        self.bottom_up = ds.transform.e > 0
        self.transform = (flip_rows(ds.transform, ds.height) if self.bottom_up
                          else ds.transform)
        self.crs = _header_crs(ds.crs)
        meta = _source_meta(ds)
        self.orientation = meta["orientation"]
        self.aux = json.dumps(meta, default=str)
        self.product_id = os.path.splitext(os.path.basename(
            os.fspath(src_path)))[0]
        self.source = os.fspath(src_path if source is None else source)
        self.product_meta = {"count": ds.count, "dtype": "int8",
                             "nodata": NODATA}

    def read_strip(self, y0, y1):
        from rasterio.windows import Window
        H = self.height
        s0, s1 = (H - y1, H - y0) if self.bottom_up else (y0, y1)
        a = self.ds.read(window=Window(0, s0, self.width, s1 - s0))
        return a[:, ::-1] if self.bottom_up else a


def encode_tile(src_path, out_path, codec: AEV2Codec, stats=True,
                chunk_px=None, strip_px=STRIP_PX, batch=256,
                release=RELEASE_NAME, uri=RELEASE_URI, revision=None,
                source=None, log=None) -> dict:
    """Encode one AE tile (GeoTIFF) to the Zarr store `out_path`,
    conventionally `output_name(src_path)`, replacing a store already
    there once the new one is written. Strips of `strip_px` rows are encoded in
    turn. chunk_px: per-kind chunk sizes, `chunk_sizes` by default.
    `release`, `uri` and `revision` are written to the header's
    `codec.release` (see georvq.ae.v2_resolve); `release=None` writes
    none. `source` is recorded as the header's `product.source`, the
    path or URL a reader is pointed to for the source tile (default
    `src_path`). Returns timings."""
    import rasterio
    path = os.fspath(src_path)
    if source is None and os.path.isabs(path) and not path.startswith("/vsi"):
        warnings.warn(f"the header's product.source records the local path "
                      f"{path}; pass source= (--source) with the path or URL "
                      "readers of the file should use", stacklevel=2)
    with rasterio.Env(**_GDAL_ENV), rasterio.open(src_path) as ds:
        tile = GeoTiffTile(ds, src_path, source, bands=codec.bands)
        return _sc.encode_product(tile, out_path, codec, stats=stats,
                                  chunk_px=chunk_px, strip_px=strip_px,
                                  batch=batch, release=release, uri=uri,
                                  revision=revision, log=log)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def open_dataset(store, model=None, channel=None, bundle=None,
                 onnx_dir=None, device=None, backend=None, trust_remote=False,
                 cache_dir=None, bf16=None, codec=None, threads=None,
                 **reader_kw) -> StoreDataset:
    """Open an AE v2 store with AE codec arguments: an open zarr.Group, or
    a local path or a URL that zarr-python opens through fsspec
    (https://, s3://, hf://, ...), with `storage_options` for fsspec, for
    example {"headers": {"Authorization": "Bearer <token>"}} for a private
    repository over https. `georvq.open` opens any release's store
    without arguments; this function is for passing a model and channel
    or an ONNX directory directly.

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
    backend sees a GPU, else cpu. `reader_kw` go to
    georvq.inference.records.RecordReader (`storage_options`, `cache`,
    `cache_bytes`, and `trace`, a records.Trace that also records the
    decode stages)."""
    from georvq.inference.bundle_resolve import default_device, resolve_codec
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
        return resolve_codec(header, AE_V2, bundle=bundle, backend=backend,
                             device=device, bf16=bf16, threads=threads,
                             cache_dir=cache_dir, trust_remote=trust_remote)

    ds = StoreDataset(store, AE_V2, codec, None if explicit else resolver,
                      **reader_kw)
    if bundle is not None and not explicit:
        try:
            ds.codec
        except BaseException:
            ds.close()
            raise
    return ds


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _window_arg(s):
    if s is None:
        return None
    r, c, h, w = (int(v) for v in s.split(","))
    return ((r, r + h), (c, c + w))


def _fetch_header(target, uri, revision) -> dict:
    """Header of the store `target`, or a stand-in naming a bitstream id
    and the release to download it from."""
    if os.path.exists(target) or "://" in target:
        return RecordReader(target).header
    if target.rstrip("/").endswith(".zarr") or os.sep in target:
        raise FileNotFoundError(f"{target}: no such store")
    return {"codec": {"bitstream_id": target,
                      "release": {"uri": uri, "revision": revision}}}


def main(argv=None):
    from georvq.ae.v2_resolve import BundleError, default_device, locate_bundle
    ap = argparse.ArgumentParser(
        prog="python -m georvq.ae.v2",
        description="AlphaEarth v2 codec: one annual tile <-> one "
                    "georvq Zarr store.")
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

    e = sub.add_parser("encode", help="GeoTIFF tile -> .georvq.zarr")
    e.add_argument("src")
    e.add_argument("out", nargs="?", default=None,
                   help="output store, or a directory to write it in "
                        "(default: {source stem}.georvq.zarr in the current "
                        "directory)")
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

    r = sub.add_parser("reconstruct", help="store -> GeoTIFF with source profile")
    r.add_argument("src")
    r.add_argument("out")
    model_args(r, onnx=True)
    r.add_argument("--quality", default=None)

    x = sub.add_parser("export-onnx", help="checkpoint + channel -> ONNX dir "
                                           "for the onnx backend")
    x.add_argument("out")
    model_args(x)
    x.add_argument("--opset", type=int, default=18)

    f = sub.add_parser("fetch", help="find or download the bundle of a store "
                                     "or bitstream id; prints its directory")
    f.add_argument("target", help="a v2 store (path or URL) or a bitstream id")
    f.add_argument("--uri", default=RELEASE_URI,
                   help="release to download a bitstream id from")
    f.add_argument("--revision", default=None)
    cache_args(f)

    i = sub.add_parser("info", help="header summary and array sizes")
    i.add_argument("src")
    i.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "info":
        with RecordReader(args.src) as rd:
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
            print(f"  quality for decoders {sorted(rd.quality_decoders)}")
            try:
                sizes = rd.stored_bytes()
            except Exception as err:
                print(f"  stored bytes unavailable ({type(err).__name__})")
            else:
                for n, b in sizes.items():
                    print(f"  {n:<26} {b:>12,} B")
                print(f"  {'store':<26} {sum(sizes.values()):>12,} B")
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
                      f"{ds.reader.stats['bytes']:,} B of records")
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
    if out is None or (os.path.isdir(out)
                       and not out.rstrip("/").endswith(".zarr")):
        out = os.path.join(out or ".", output_name(args.src))
    t = encode_tile(args.src, out, codec, stats=not args.no_quality,
                    batch=args.batch, release=args.release or None,
                    uri=args.release_uri, revision=args.revision,
                    source=args.source, log=lambda s: print(s, file=sys.stderr))
    print(json.dumps(dict(t, out=out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
