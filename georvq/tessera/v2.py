"""Tessera v2 release layer: geotessera tiles as georvq Zarr stores.

A Tessera v2 store holds one geotessera tile (a 0.1 degree cell of one
year: 128-band int8 embeddings with a float32 scale per pixel, x = q *
scale, in the tile's UTM zone at 10 m) coded by the `tessera_v2` model, a
stats_cond model whose per-window mean travels through an
`RVQPCAMeanChannel`. The store layout is that of every release
(georvq.inference.records, docs/GEORVQ_ZARR.md), with these chunk sizes,
chosen because a tile is about 1100 x 800 px and each request to a CDN
costs more than the bytes of a small chunk:

- `base` and `mid`, 2048 px chunks (the whole tile), and `mask` chunked
  as base;
- `fine`, 1024 px chunks (two or four per tile);
- `quality/<decoder_id>`, optional, chunked as base: per window and per
  rung, the mean and p99 of the per-pixel relative L2 error
  ||decoded - source|| / ||source|| of the 128-band vector, measured at
  encode time with that decoder. No cosine statistic is kept, because
  Tessera vectors carry magnitude.

The release's decisions are `TESSERA_V2`, a georvq.inference.store_codec
.ReleaseSpec: the prep of georvq.tessera.io (x = q * scale / XNORM, valid
where the vector is not all zero and 0 < scale < SCALE_MAX), the rungs of
the fill order from the model's minimum depths, and float32 output in
geotessera's dequantised units (`units="source"`, the default; `units=
"model"` returns the model's x = source / XNORM), with zero vectors at
invalid pixels and nodata 0.0. `to_geotessera` turns output back into
geotessera's int8 codes and per-pixel scale.

Tiles are not multiples of the 64 px window. The last row and column of
windows are padded with invalid pixels and coded, never cropped, and the
store's mask restores the tile's extent and its nodata.

`GeotesseraTile` reads one tile, from Tessera's public bucket or a local
copy of its layout, and georeferences it: by the tile's landmask GeoTIFF
when one is found, and otherwise by the rule the landmasks follow (CRS
the standard UTM zone of the tile centre, origin the minimum x and
maximum y of the projected corners of the 0.1 degree cell, 10 m pixels,
size the rounded extent over 10 m). When both are available they are
compared and a disagreement is reported. One store is written per tile,
`{year}/grid_{lon}_{lat}.georvq.zarr`, and `build_index` joins stores to
geotessera's manifest on (year, lon, lat).

The stores are derived from the TESSERA embeddings, which are licensed
CC-BY-SA 4.0, so each store's header carries that licence and its
attribution (`licence`).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import warnings

import numpy as np

from georvq.inference.store_codec import (OnnxStoreCodec, ReleaseSpec,
                                          StoreCodec, encode_product)

XNORM = 32.0          # georvq.sources.tessera.TesseraSource.XNORM
SCALE_MAX = 1.0       # georvq.sources.tessera.TesseraSource.SCALE_MAX
BANDS = 128
KIND = "tessera-tile"
FORMAT = "georvq-tessera/v2"
BUNDLE_FORMAT = "georvq-tessera/v2-bundle"
RELEASE_NAME = "tessera-v2"
# not yet published: the bundle's repository is the maintainer's to create
RELEASE_URI = "hf://earth-compress/georvq-tessera-v2"
ENV_BUNDLE = "GEORVQ_TESSERA_BUNDLE"
BASE_CHUNK_PX, MID_CHUNK_PX, FINE_CHUNK_PX = 2048, 2048, 1024
STRIP_PX = 2048
TESSERA_BUCKET = "https://s3.us-west-2.amazonaws.com/tessera-embeddings/v1/"
EMBEDDINGS_DIR = "global_0.1_degree_representation"
LANDMASK_DIR = "global_0.1_degree_tiff_all"
LICENCE = {
    "spdx": "CC-BY-SA-4.0",
    "url": "https://creativecommons.org/licenses/by-sa/4.0/",
    "attribution": "TESSERA embeddings, Energy and Environment Group, "
                   "Department of Computer Science and Technology, "
                   "University of Cambridge, via geotessera "
                   "(https://github.com/ucam-eo/geotessera)",
    "changes": "lossy georvq encoding of the int8 embeddings and their "
               "per-pixel scales; the decoded values are approximations",
}


# --------------------------------------------------------------------------
# the release's hooks
# --------------------------------------------------------------------------

def _prep(model, arrays, device):
    from georvq.tessera.io import prep
    q, s = arrays
    return prep(np.ascontiguousarray(q), np.ascontiguousarray(s),
                device=device, channels_last=False, bands=BANDS)


def raw_valid(q, s) -> np.ndarray:
    """(H, W) validity of geotessera pixels, as georvq.tessera.io.prep
    decides it: a vector that is not all zero, and 0 < scale < SCALE_MAX
    (NaN and infinite scales are invalid)."""
    s = np.asarray(s, np.float32)
    with np.errstate(invalid="ignore"):
        return (np.asarray(q) != 0).any(0) & (s > 0) & (s < SCALE_MAX)


def _raw_valid(arrays):
    return raw_valid(*arrays)


def _invalidate(arrays, mask):
    q, s = arrays
    q[:, mask] = 0
    s[mask] = 0
    return q, s


def _check_raw(arrays, bands):
    if len(arrays) != 2:
        raise ValueError("Tessera pixels are (int8 embeddings, float32 scale)")
    q, s = arrays
    if q.dtype != np.int8 or q.ndim != 3 or q.shape[0] != bands:
        raise ValueError(f"expected int8 ({bands}, H, W) embeddings, got "
                         f"{q.dtype} {q.shape}")
    if s.shape != q.shape[1:] or s.dtype.kind != "f":
        raise ValueError(f"expected a float {q.shape[1:]} scale plane, got "
                         f"{s.dtype} {s.shape}")


def _units(units):
    if units == "source":
        return XNORM
    if units == "model":
        return 1.0
    raise ValueError(f"units {units!r}: expected 'source' (geotessera's "
                     "dequantised values) or 'model' (source / XNORM)")


def postprocess_np(out, valid, units="source"):
    """Decoded field (n, C, W, W) -> float32 in `units`, zero vectors at
    invalid pixels."""
    o = np.asarray(out, np.float32) * np.float32(_units(units))
    o[np.broadcast_to(~np.asarray(valid)[:, None], o.shape)] = 0.0
    return o


def postprocess(out, valid, units="source"):
    import torch
    o = out.float() * _units(units)
    o = torch.where(valid[:, None], o, torch.zeros((), device=o.device))
    return o.cpu().numpy()


def rel_l2(out, x):
    """Per-pixel relative L2 error of the vectors, as the training run
    reports it (georvq.sources.tessera)."""
    return ((out - x).pow(2).sum(1).sqrt()
            / x.pow(2).sum(1).sqrt().clamp_min(1e-6))


def _check_model(model, channel):
    sc = model.stats_cond
    if sc is None or not sc.subtract_mean or sc.use_std:
        raise ValueError("Tessera v2 needs a stats_cond model with a "
                         "transmitted mean and no std")
    if model.conditioner is not None or model.selector is not None:
        raise ValueError("Tessera v2 ships codecs without a location prior "
                         "or selector (the ONNX export covers no others)")


def _bundle_readme(meta) -> str:
    tiers = "\n".join(f"- {t}" for t in meta["tiers"])
    rows = "\n".join(
        f"| {'-'.join(map(str, r['tuple']))} | {r['bits_window']} | "
        f"{r['bits_px']:.4f} |" for r in meta["rungs"])
    size = sum(f["bytes"] for f in meta["files"].values())
    src = meta.get("source") or {}
    onnx = (f"\n`{meta['onnx']}/` holds the ONNX export used by the torch-free "
            "backend (`backend=\"onnx\"`).\n" if meta.get("onnx") else "")
    return f"""# georvq Tessera v2 bundle

<!-- TODO(maintainer): the licence of these weights is undecided
(Apache-2.0, as the AE v2 weights, or CC-BY-SA-4.0, as the TESSERA
embeddings they were trained on). Choose one before publishing, and
state it here and in NOTICE. -->

Licence of the weights: **TODO, undecided (Apache-2.0 or CC-BY-SA-4.0)**.
The stores this bundle decodes are derived from the TESSERA embeddings
and are licensed CC-BY-SA 4.0 (TESSERA embeddings, Energy and
Environment Group, Department of Computer Science and Technology,
University of Cambridge, via geotessera).

Decoder and encoder for geotessera tiles stored as georvq Zarr stores
(`{meta['codec_format']}`, see docs/GEORVQ_ZARR.md).

- bitstream_id `{meta['bitstream_id']}`, decoder_id `{meta['decoder_id']}`
- window {meta['window']} px, {meta['bands']} bands
- source checkpoint {src.get('checkpoint')} at step {src.get('step')}

Tiers (grid:dim:codes:depth), with {meta['book_bits']}-bit codebooks:

{tiers}

Every window also carries its mean, {meta['mean_channel']['bits']} bits
through the {meta['mean_channel']['kind']} mean channel. The rate of each
rung below includes it.

| rung | bits/window | bits/px |
|---|---|---|
{rows}

Files ({size / 1e6:.1f} MB): `weights.npz` (model tensors), `codebooks.npz`
(per level an int8 `q` and a float32 per-codeword `scale`, codebook =
q * scale, with q = -128 standing for -0.0), `mean_channel.npz`,
`config.json`, `bundle.json`. All arrays load with `allow_pickle=False`.
{onnx}
```python
import georvq
ds = georvq.open("2024/grid_17.25_46.75.georvq.zarr")
a = ds.read(quality="20-13-0")       # (128, H, W) float32
```
"""


TESSERA_V2 = ReleaseSpec(
    name=RELEASE_NAME, codec_format=FORMAT, product_kind=KIND,
    bundle_format=BUNDLE_FORMAT, onnx_meta_key="tessera_v2", bands=BANDS,
    prep=_prep, raw_valid=_raw_valid, invalidate=_invalidate,
    check_raw=_check_raw, pad=(0, 0.0),
    postprocess=postprocess, postprocess_np=postprocess_np,
    out_dtype="float32", out_nodata=0.0,
    output_options={"units": "source"},
    release_uri=RELEASE_URI, env_bundle=ENV_BUNDLE,
    chunk_px=(BASE_CHUNK_PX, MID_CHUNK_PX, FINE_CHUNK_PX), strip_px=STRIP_PX,
    quality_stats=("mean", "p99"), quality_error=rel_l2,
    quality_error_text="||decoded - source|| / ||source|| per valid pixel "
                       "(relative L2 of the 128-band vector), decoded float "
                       "field against the prepared source",
    check_model=_check_model, bundle_readme=_bundle_readme, licence=LICENCE)


class TesseraV2Codec(StoreCodec):
    spec = TESSERA_V2

    def __init__(self, model, channel, bf16=None, rungs=None, spec=None):
        super().__init__(model, channel, bf16=bf16, rungs=rungs,
                         spec=spec or TESSERA_V2)


class TesseraV2OnnxCodec(OnnxStoreCodec):
    spec = TESSERA_V2

    def __init__(self, onnx_dir, device="cpu", threads=None, spec=None):
        super().__init__(onnx_dir, device=device, threads=threads,
                         spec=spec or TESSERA_V2)


TESSERA_V2 = TESSERA_V2.with_(torch_codec=TesseraV2Codec,
                              onnx_codec=TesseraV2OnnxCodec)
TesseraV2Codec.spec = TesseraV2OnnxCodec.spec = TESSERA_V2
SPEC = TESSERA_V2


def to_geotessera(x, valid=None):
    """Float32 (128, H, W) vectors in source units -> geotessera's
    representation: int8 (H, W, 128) codes and a float32 (H, W) scale,
    with x ~ q * scale. The scale is each pixel's max |x| / 127, as in
    the distributed tiles (every valid pixel has a code of +-127), and
    invalid pixels (`valid` False, or an all-zero vector) get zero codes
    and a zero scale."""
    x = np.asarray(x, np.float32)
    m = np.abs(x).max(0)
    if valid is not None:
        m = np.where(valid, m, 0.0)
    scale = (m / 127.0).astype(np.float32)
    safe = np.where(scale > 0, scale, 1.0)
    q = np.clip(np.rint(x / safe), -127, 127).astype(np.int8)
    q[:, scale == 0] = 0
    return np.ascontiguousarray(q.transpose(1, 2, 0)), scale


# --------------------------------------------------------------------------
# tiles
# --------------------------------------------------------------------------

def tile_name(lon, lat) -> str:
    return f"grid_{lon:.2f}_{lat:.2f}"


def utm_epsg(lon, lat) -> int:
    """EPSG code of the standard UTM zone of a point."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def rule_georef(lon, lat) -> tuple:
    """(epsg, (a, b, c, d, e, f) north-up transform, (height, width)) of
    a 0.1 degree tile centred on (lon, lat), by the rule the landmask
    GeoTIFFs follow: the standard UTM zone of the centre, origin the
    minimum x and maximum y of the projected corners (not snapped), 10 m
    pixels, and the extent over 10 m rounded."""
    from pyproj import Transformer
    epsg = utm_epsg(lon, lat)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform([lon - .05, lon + .05, lon - .05, lon + .05],
                          [lat - .05, lat - .05, lat + .05, lat + .05])
    t = (10.0, 0.0, float(min(xs)), 0.0, -10.0, float(max(ys)))
    shape = (int(round((max(ys) - min(ys)) / 10)),
             int(round((max(xs) - min(xs)) / 10)))
    return epsg, t, shape


def _read(path, cache_dir=None) -> bytes:
    """The bytes at a local path or an http(s) URL, kept in `cache_dir`
    when given."""
    if "://" not in path:
        with open(path, "rb") as f:
            return f.read()
    if cache_dir:
        local = os.path.join(cache_dir, path.split("://", 1)[1])
        if os.path.exists(local):
            with open(local, "rb") as f:
                return f.read()
    with urllib.request.urlopen(path, timeout=600) as r:
        data = r.read()
    if cache_dir:
        os.makedirs(os.path.dirname(local), exist_ok=True)
        tmp = local + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, local)
    return data


def _exists(path) -> bool:
    if "://" not in path:
        return os.path.exists(path)
    req = urllib.request.Request(path, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status == 200
    except Exception:
        return False


class GeotesseraTile:
    """One geotessera tile as a georvq.inference.store_codec.ProductSource.

    `root` is the bucket (default Tessera's public one) or a local
    directory with its layout (`global_0.1_degree_representation/{year}/
    grid_{lon}_{lat}/grid_{lon}_{lat}.npy` and `_scales.npy`, landmasks in
    `global_0.1_degree_tiff_all/`). `landmask` is a landmask GeoTIFF to
    georeference by, overriding the one under `root`; `cache_dir` keeps
    downloads. `source_root`, when given, is the root whose paths the
    store's header records as the tile's source, for a tile read from a
    local copy of the bucket (default: `root`). The tile is georeferenced
    by its landmask when one is found, else by `rule_georef`; when both
    are available a disagreement warns and the landmask is used."""

    def __init__(self, year, lon, lat, root=TESSERA_BUCKET, landmask=None,
                 cache_dir=None, check_rule=True, source_root=None):
        from rasterio.crs import CRS
        self.year, self.lon, self.lat = int(year), float(lon), float(lat)
        name = tile_name(lon, lat)
        root = root.rstrip("/") + "/"
        rel = f"{EMBEDDINGS_DIR}/{self.year}/{name}/{name}"
        q = np.load(io.BytesIO(_read(f"{root}{rel}.npy", cache_dir)),
                    allow_pickle=False)
        s = np.load(io.BytesIO(_read(f"{root}{rel}_scales.npy", cache_dir)),
                    allow_pickle=False)
        rec = (source_root or root).rstrip("/") + "/" + rel
        self.embedding_path, self.scales_path = rec + ".npy", rec + "_scales.npy"
        if q.ndim != 3 or q.shape[-1] != BANDS or q.dtype != np.int8:
            raise ValueError(f"{self.embedding_path}: {q.dtype} {q.shape}, "
                             f"expected int8 (H, W, {BANDS})")
        if s.shape != q.shape[:2]:
            raise ValueError(f"{self.scales_path}: {s.shape}, expected "
                             f"{q.shape[:2]}")
        self.q = np.ascontiguousarray(q.transpose(2, 0, 1))
        s = s.astype(np.float32)
        self.s = np.where(np.isfinite(s), s, 0.0).astype(np.float32)
        self.height, self.width = q.shape[:2]

        rule = rule_georef(self.lon, self.lat)
        lm = landmask or f"{root}{LANDMASK_DIR}/{name}.tiff"
        lm_geo = self._landmask(lm, cache_dir) if (
            landmask or _exists(lm)) else None
        self.rule_agrees = None
        if lm_geo is not None:
            if check_rule:
                self.rule_agrees = _same_georef(lm_geo, rule)
                if not self.rule_agrees:
                    warnings.warn(f"{name} {self.year}: the landmask's "
                                  f"georeferencing {lm_geo} differs from the "
                                  f"rule's {rule}; using the landmask",
                                  stacklevel=2)
            epsg, t, shape, how = (*lm_geo, "landmask")
        else:
            epsg, t, shape, how = (*rule, "rule")
        if shape != (self.height, self.width):
            raise ValueError(f"{name} {self.year}: {self.height}x{self.width} "
                             f"pixels, its {how} gives {shape}")
        self.georef = how
        self.crs = CRS.from_epsg(epsg)
        self.transform = t
        self.orientation = "north-up"
        self.product_id = f"{self.year}/{name}"
        self.source = self.embedding_path
        self.product_meta = {
            "count": BANDS, "dtype": "int8", "nodata": None,
            "scale": "float32 per pixel, x = q * scale",
            "year": self.year, "lon": self.lon, "lat": self.lat,
            "scales": self.scales_path, "georef": how}
        self.aux = {"geotessera": {"year": self.year, "lon": self.lon,
                                   "lat": self.lat,
                                   "embedding": self.embedding_path,
                                   "scales": self.scales_path,
                                   "georef": how,
                                   "rule_agrees": self.rule_agrees}}

    @staticmethod
    def _landmask(path, cache_dir):
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(_read(path, cache_dir)) as mf, mf.open() as ds:
            return (ds.crs.to_epsg(), tuple(float(v) for v in
                                            list(ds.transform)[:6]),
                    (ds.height, ds.width))

    def read_strip(self, y0, y1):
        return self.q[:, y0:y1], self.s[y0:y1]


def _same_georef(a, b, tol=0.01) -> bool:
    (ea, ta, sa), (eb, tb, sb) = a, b
    return ea == eb and sa == sb and all(abs(x - y) <= tol
                                         for x, y in zip(ta, tb))


def store_path(out_dir, year, lon, lat) -> str:
    return os.path.join(out_dir, str(int(year)), tile_name(lon, lat)
                        + ".georvq.zarr")


def encode_tile(tile: GeotesseraTile, out_path, codec: StoreCodec,
                stats=True, chunk_px=None, batch=256, release=RELEASE_NAME,
                uri=RELEASE_URI, revision=None, assets=None, log=None) -> dict:
    """Encode one geotessera tile to the store `out_path`. Returns
    timings."""
    return encode_product(tile, out_path, codec, stats=stats,
                          chunk_px=chunk_px, batch=batch, release=release,
                          uri=uri, revision=revision, assets=assets, log=log)


# --------------------------------------------------------------------------
# bundles and the index
# --------------------------------------------------------------------------

def build_bundle(ckpt, mean_npz, out_dir, notes="", onnx=False) -> dict:
    """Bundle a training checkpoint and its mean channel tables."""
    from georvq.inference.bundle import build_bundle as _build
    return _build(ckpt, mean_npz, out_dir, TESSERA_V2, notes=notes, onnx=onnx,
                  extra={"licence_stores": LICENCE,
                         "licence_weights": "TODO: undecided (Apache-2.0 or "
                                            "CC-BY-SA-4.0)"})


def load_bundle(bundle_dir, device="cpu", bf16=None, verify=True):
    from georvq.inference.bundle import load_bundle as _load
    return _load(bundle_dir, TESSERA_V2, device, bf16, verify)


def build_index(stores, manifest, out_path=None, url_prefix=None, root=None):
    """Index of Tessera v2 stores joined to geotessera's manifest
    (`manifest.parquet`, one row per (year, lon, lat) tile) on (year,
    lon_i, lat_i): the manifest's columns, `path` (the store, `url_prefix`
    joined to its path relative to `root` when given, or that relative
    path alone when `url_prefix` is ""), and the store facts of georvq.inference.store_index.tile_record. Writes Parquet to
    `out_path` when given; returns the table."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from georvq.inference.records import RecordReader
    from georvq.inference.store_index import find_stores, tile_record
    if isinstance(stores, (str, os.PathLike)) and os.path.isdir(stores) \
            and not os.fspath(stores).rstrip("/").endswith(".georvq.zarr"):
        root = stores if root is None else root
        stores = find_stores(stores)
    stores = [os.fspath(f) for f in stores]
    rows, keys = [], []
    for f in stores:
        p = RecordReader(f).header["product"]
        if p.get("kind") != KIND:
            raise ValueError(f"{f}: product kind {p.get('kind')!r}, not {KIND}")
        key = (int(p["year"]), int(round(p["lon"] * 100)),
               int(round(p["lat"] * 100)))
        if key in keys:
            raise ValueError(f"{f}: a second store for {key}")
        keys.append(key)
        rec = tile_record(f)
        path = os.path.abspath(f)
        if url_prefix is not None:
            prefix = url_prefix.rstrip("/")
            path = (prefix + "/" if prefix else "") + os.path.relpath(
                f, os.fspath(root)).replace(os.sep, "/")
        rows.append({"path": path, **{k: rec[k] for k in (
            "bitstream_id", "decoder_id", "store_bytes", "metadata_bytes",
            "quality_bytes", "rungs")}})
    years = sorted({k[0] for k in keys})
    man = pq.read_table(manifest, filters=[("year", "in", years)])
    ty = {c: man.schema.field(c).type for c in ("year", "lon_i", "lat_i")}
    want = pa.table({c: pa.array([k[i] for k in keys], ty[c])
                     for i, c in enumerate(("year", "lon_i", "lat_i"))})
    want = want.append_column("_row", pa.array(range(len(keys)), pa.int64()))
    joined = want.join(man, keys=["year", "lon_i", "lat_i"], join_type="inner")
    if joined.num_rows != len(keys):
        found = set(zip(*[joined.column(c).to_pylist()
                          for c in ("year", "lon_i", "lat_i")]))
        missing = [k for k in keys if k not in found]
        raise ValueError(f"tiles not in {manifest}: {missing}")
    joined = joined.sort_by("_row")
    order = joined.column("_row").to_pylist()
    joined = joined.drop(["_row"])
    for c in ("path", "bitstream_id", "decoder_id", "store_bytes",
              "metadata_bytes", "quality_bytes"):
        joined = joined.append_column(c, pa.array([rows[i][c] for i in order]))
    rung_t = pa.list_(pa.struct([("rung", pa.string()),
                                 ("bits_px", pa.float64()),
                                 ("bytes", pa.int64())]))
    joined = joined.append_column("rungs", pa.array(
        [rows[i]["rungs"] for i in order], rung_t))
    if out_path is not None:
        pq.write_table(joined, out_path, compression="zstd")
    return joined


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _tile_arg(s):
    lon, lat = (float(v) for v in s.split(","))
    return lon, lat


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m georvq.tessera.v2",
        description="Tessera v2 codec: one geotessera tile <-> one georvq "
                    "Zarr store.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="geotessera tiles -> stores")
    e.add_argument("out_dir")
    e.add_argument("--year", type=int, required=True)
    e.add_argument("--tile", type=_tile_arg, action="append", required=True,
                   help="lon,lat of a tile centre, e.g. 17.25,46.75; repeat")
    e.add_argument("--bundle", required=True)
    e.add_argument("--root", default=TESSERA_BUCKET)
    e.add_argument("--source-root", default=None,
                   help="root recorded as the tiles' source in the headers "
                        "(default: --root)")
    e.add_argument("--cache-dir", default=None,
                   help="keep downloaded tiles here")
    e.add_argument("--device", default=None)
    e.add_argument("--no-quality", action="store_true")
    e.add_argument("--batch", type=int, default=256)
    b = sub.add_parser("bundle", help="checkpoint + mean channel -> bundle")
    b.add_argument("ckpt")
    b.add_argument("mean_npz")
    b.add_argument("out_dir")
    b.add_argument("--onnx", action="store_true")
    b.add_argument("--notes", default="")
    x = sub.add_parser("index", help="stores + geotessera manifest -> parquet")
    x.add_argument("stores")
    x.add_argument("manifest")
    x.add_argument("out")
    x.add_argument("--url-prefix", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "bundle":
        meta = build_bundle(args.ckpt, args.mean_npz, args.out_dir,
                            notes=args.notes, onnx=args.onnx)
        print(json.dumps({k: meta[k] for k in ("bitstream_id", "decoder_id")}))
        return 0
    if args.cmd == "index":
        t = build_index(args.stores, args.manifest, args.out,
                        url_prefix=args.url_prefix)
        print(f"{args.out}: {t.num_rows} stores")
        return 0
    from georvq.inference.bundle_resolve import default_device
    from georvq.inference.store_codec import quiet_build
    with quiet_build():
        codec = load_bundle(args.bundle, device=args.device or
                            default_device("torch"))
    for lon, lat in args.tile:
        tile = GeotesseraTile(args.year, lon, lat, root=args.root,
                              cache_dir=args.cache_dir,
                              source_root=args.source_root)
        out = store_path(args.out_dir, args.year, lon, lat)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t = encode_tile(tile, out, codec, stats=not args.no_quality,
                        batch=args.batch,
                        log=lambda m: print(m, file=sys.stderr))
        print(json.dumps(dict(t, out=out, georef=tile.georef,
                              rule_agrees=tile.rule_agrees)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
