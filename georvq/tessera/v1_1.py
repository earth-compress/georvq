"""Tessera v1.1 release layer: geotessera v1.1 tiles as georvq Zarr stores.

A Tessera v1.1 store holds one tile of the Tessera v1.1 embeddings (the
global run produced and hosted by dClimate; georvq.tessera.tiles): a
0.1 degree cell of one year, 128-band int8 codes with a float32 scale
per pixel, x = q * scale, on the 10 m grid of its UTM zone. It is coded
by the `tessera_v11` model (step 133,000), a stats_cond model whose
per-window mean travels through a `RotatedChannel` of 1280 bits: the
principal axes of training window means, each component companded with
asinh over the hard bound that every value of a valid pixel obeys,
|x_d| <= 127 * SCALE_MAX / XNORM, so that no mean is clipped. The store
layout is that of every release (georvq.inference.records,
docs/GEORVQ_ZARR.md), with these chunk sizes, chosen because a tile is
about 1100 x 800 px and each request to a CDN costs more than the bytes
of a small chunk:

- `base` and `mid`, 2048 px chunks (the whole tile), and `mask` chunked
  as base;
- `fine`, 1024 px chunks (two or four per tile);
- `quality/<decoder_id>`, optional, chunked as base: per window and per
  rung, the mean and p99 of the per-pixel relative L2 error
  ||decoded - source|| / ||source|| of the 128-band vector, measured at
  encode time with that decoder. No cosine statistic is kept, because
  Tessera vectors carry magnitude.

The release's decisions are `TESSERA_V1_1`, a georvq.inference
.store_codec.ReleaseSpec: the prep of georvq.tessera.io (x = q * scale /
XNORM, valid where the vector is not all zero and 0 < scale <
SCALE_MAX), the rungs of the fill order from the model's minimum depths,
and float32 output in the embeddings' dequantised units (`units=
"source"`, the default; `units="model"` returns the model's x = source /
XNORM), with zero vectors at invalid pixels and nodata 0.0.
`to_geotessera` turns output back into int8 codes and a per-pixel scale.

The embeddings are named in the codec format (`georvq-tessera/v1.1`),
in the bundle format, in each store header's `product.embeddings` and in
bundle.json's `embeddings`, so a store or bundle for another version of
the embeddings (Tessera v1, whose embedding space differs) is never
decoded with this codec: `georvq.open` refuses a bundle of another
format, and `load_bundle` one whose `embeddings` differ.

Tiles are not multiples of the 64 px window. The last row and column of
windows are padded with invalid pixels and coded, never cropped, and the
store's mask restores the tile's extent and its nodata. One store is
written per tile, `{year}/grid_{lon}_{lat}.georvq.zarr`, and
`georvq.tessera.index.build_index` makes the release's tile index.

The stores are derived from the TESSERA embeddings, which are licensed
CC-BY-SA 4.0, so each store's header carries that licence and its
attribution (`licence`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from georvq.inference.store_codec import (OnnxStoreCodec, ReleaseSpec,
                                          StoreCodec, encode_product)
from georvq.tessera.tiles import (EMBEDDINGS, V11_STORE, TesseraTile,
                                  tile_name)

XNORM = 32.0          # georvq.sources.tessera.TesseraSource.XNORM
SCALE_MAX = 1.0       # georvq.sources.tessera.TesseraSource.SCALE_MAX
BANDS = 128
KIND = "tessera-tile"
FORMAT = "georvq-tessera/v1.1"
BUNDLE_FORMAT = "georvq-tessera/v1.1-bundle"
RELEASE_NAME = "tessera-v1.1"
RELEASE_URI = "hf://earth-compress/georvq-tessera-v1.1"
SAMPLE_REPO = "earth-compress/georvq-tessera-v1.1-sample"
ENV_BUNDLE = "GEORVQ_TESSERA_BUNDLE"
BASE_CHUNK_PX, MID_CHUNK_PX, FINE_CHUNK_PX = 2048, 2048, 1024
STRIP_PX = 2048
MEAN_CHANNEL_KIND = "rotated"
# every value of a valid pixel lies in [-BOX, BOX]: max |q| is 127 and
# the scale is below SCALE_MAX; the channel's ranges are built from it
# (scripts/fit_mean_channel.py --box)
BOX = 127 * SCALE_MAX / XNORM
TRAINING_DATA = ("TESSERA v1.1 embeddings (CC-BY-SA 4.0), Energy and "
                 "Environment Group, Department of Computer Science and "
                 "Technology, University of Cambridge, in the global run "
                 "produced and hosted by dClimate, via geotessera")
LICENCE = {
    "spdx": "CC-BY-SA-4.0",
    "url": "https://creativecommons.org/licenses/by-sa/4.0/",
    "attribution": "TESSERA embeddings v1.1, Energy and Environment Group, "
                   "Department of Computer Science and Technology, "
                   "University of Cambridge, via geotessera "
                   "(https://github.com/ucam-eo/geotessera); v1.1 global "
                   "run produced and hosted by dClimate "
                   "(https://github.com/dClimate/tessera-embeddings)",
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
    """(H, W) validity of Tessera pixels, as georvq.tessera.io.prep
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
    raise ValueError(f"units {units!r}: expected 'source' (the embeddings' "
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
        raise ValueError("Tessera v1.1 needs a stats_cond model with a "
                         "transmitted mean and no std")
    if model.conditioner is not None or model.selector is not None:
        raise ValueError("Tessera v1.1 ships codecs without a location prior "
                         "or selector (the ONNX export covers no others)")
    if channel.kind != MEAN_CHANNEL_KIND:
        raise ValueError(f"Tessera v1.1 sends the mean through a "
                         f"{MEAN_CHANNEL_KIND!r} channel, not {channel.kind!r}")


def _bundle_readme(meta) -> str:
    from georvq.inference.bundle import bundle_readme
    return bundle_readme(
        meta, "georvq Tessera v1.1 bundle",
        "Decoder and encoder for Tessera v1.1 embedding tiles (the global "
        "run produced by\ndClimate) stored as georvq Zarr stores "
        f"(`{meta['codec_format']}`, see docs/GEORVQ_ZARR.md).",
        """
import georvq
with georvq.open("2024/grid_26.15_44.45.georvq.zarr") as ds:
    a = ds.read(quality="20-13-0")       # (128, H, W) float32

from georvq.tessera import v1_1
codec = v1_1.load_bundle("bundle_dir", device="cuda")
tile = v1_1.TesseraTile.read(2024, 26.15, 44.45)
v1_1.encode_tile(tile, "grid_26.15_44.45.georvq.zarr", codec)
""")


TESSERA_V1_1 = ReleaseSpec(
    name=RELEASE_NAME, codec_format=FORMAT, product_kind=KIND,
    bundle_format=BUNDLE_FORMAT, onnx_meta_key="tessera_v1_1", bands=BANDS,
    prep=_prep, raw_valid=_raw_valid, invalidate=_invalidate,
    check_raw=_check_raw, pad=(0, 0.0),
    postprocess=postprocess, postprocess_np=postprocess_np,
    out_dtype="float32", out_nodata=0.0,
    output_options={"units": "source"},
    release_uri=RELEASE_URI, env_bundle=ENV_BUNDLE,
    chunk_px=(BASE_CHUNK_PX, MID_CHUNK_PX, FINE_CHUNK_PX), strip_px=STRIP_PX,
    quality_stats=("mean", "p99"), quality_error=rel_l2,
    quality_name="rel_l2",
    quality_error_text="||decoded - source|| / ||source|| per valid pixel "
                       "(relative L2 of the 128-band vector), decoded float "
                       "field against the prepared source",
    check_model=_check_model, bundle_readme=_bundle_readme, licence=LICENCE,
    training_data=TRAINING_DATA)


class TesseraV11Codec(StoreCodec):
    spec = TESSERA_V1_1

    def __init__(self, model, channel, bf16=None, rungs=None, spec=None):
        super().__init__(model, channel, bf16=bf16, rungs=rungs,
                         spec=spec or TESSERA_V1_1)


class TesseraV11OnnxCodec(OnnxStoreCodec):
    spec = TESSERA_V1_1

    def __init__(self, onnx_dir, device="cpu", threads=None, spec=None):
        super().__init__(onnx_dir, device=device, threads=threads,
                         spec=spec or TESSERA_V1_1)


TESSERA_V1_1 = TESSERA_V1_1.with_(torch_codec=TesseraV11Codec,
                                  onnx_codec=TesseraV11OnnxCodec)
TesseraV11Codec.spec = TesseraV11OnnxCodec.spec = TESSERA_V1_1
SPEC = TESSERA_V1_1


def to_geotessera(x, valid=None):
    """Float32 (128, H, W) vectors in source units -> the embeddings'
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
# stores and bundles
# --------------------------------------------------------------------------

def store_path(out_dir, year, lon, lat) -> str:
    return os.path.join(out_dir, str(int(year)), tile_name(lon, lat)
                        + ".georvq.zarr")


def encode_tile(tile: TesseraTile, out_path, codec: StoreCodec,
                stats=True, chunk_px=None, batch=256, release=RELEASE_NAME,
                uri=RELEASE_URI, revision=None, assets=None, log=None) -> dict:
    """Encode one tile to the store `out_path`. Returns timings."""
    if tile.product_meta.get("embeddings") != EMBEDDINGS:
        raise ValueError(f"{tile.product_id}: embeddings "
                         f"{tile.product_meta.get('embeddings')}, the codec "
                         f"codes {EMBEDDINGS}")
    return encode_product(tile, out_path, codec, stats=stats,
                          chunk_px=chunk_px, batch=batch, release=release,
                          uri=uri, revision=revision, assets=assets, log=log)


def build_bundle(ckpt, mean_npz, out_dir, notes="", onnx=False) -> dict:
    """Bundle a training checkpoint and its mean channel tables."""
    from georvq.inference.bundle import build_bundle as _build
    return _build(ckpt, mean_npz, out_dir, TESSERA_V1_1, notes=notes,
                  onnx=onnx, extra={"embeddings": dict(EMBEDDINGS)})


def write_decoder_bundle(bundle_dir, out_dir=None, verify=True) -> dict:
    """The decoder-only bundle of the full bundle at `bundle_dir`, in
    `out_dir` (default `bundle_dir/decoder`); build_bundle writes it
    (georvq.inference.bundle.write_decoder_bundle)."""
    from georvq.inference.bundle import write_decoder_bundle as _write
    return _write(bundle_dir, out_dir, TESSERA_V1_1, verify=verify)


def load_bundle(bundle_dir, device="cpu", bf16=None, verify=True):
    """The release's codec from a bundle directory, refused when the
    bundle was made for other embeddings."""
    from georvq.inference.bundle import load_bundle as _load
    codec = _load(bundle_dir, TESSERA_V1_1, device, bf16, verify)
    got = codec.bundle.get("embeddings")
    if got != EMBEDDINGS:
        raise ValueError(f"{bundle_dir}: a bundle for embeddings {got}, "
                         f"this release codes {EMBEDDINGS}")
    return codec


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _tile_arg(s):
    lon, lat = (float(v) for v in s.split(","))
    return lon, lat


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m georvq.tessera.v1_1",
        description="Tessera v1.1 codec: one tile <-> one georvq Zarr "
                    "store.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="Tessera v1.1 tiles -> stores")
    e.add_argument("out_dir")
    e.add_argument("--year", type=int, required=True)
    e.add_argument("--tile", type=_tile_arg, action="append", required=True,
                   help="lon,lat of a tile centre, e.g. 26.15,44.45; repeat")
    e.add_argument("--bundle", required=True)
    e.add_argument("--store", default=V11_STORE,
                   help="the v1.1 Zarr store (URL or local path)")
    e.add_argument("--device", default=None)
    e.add_argument("--no-quality", action="store_true")
    e.add_argument("--batch", type=int, default=256)
    b = sub.add_parser("bundle", help="checkpoint + mean channel -> bundle")
    b.add_argument("ckpt")
    b.add_argument("mean_npz")
    b.add_argument("out_dir")
    b.add_argument("--onnx", action="store_true")
    b.add_argument("--notes", default="")
    x = sub.add_parser("index", help="stores -> GeoParquet index")
    x.add_argument("stores")
    x.add_argument("out")
    x.add_argument("--url-prefix", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "bundle":
        meta = build_bundle(args.ckpt, args.mean_npz, args.out_dir,
                            notes=args.notes, onnx=args.onnx)
        print(json.dumps({k: meta[k] for k in ("bitstream_id", "decoder_id")}))
        return 0
    if args.cmd == "index":
        from georvq.tessera.index import build_index
        t = build_index(args.stores, args.out, url_prefix=args.url_prefix)
        print(f"{args.out}: {t.num_rows} stores")
        return 0
    from georvq.inference.bundle_resolve import default_device
    from georvq.inference.store_codec import quiet_build
    with quiet_build():
        codec = load_bundle(args.bundle, device=args.device or
                            default_device("torch"))
    for lon, lat in args.tile:
        tile = TesseraTile.read(args.year, lon, lat, store=args.store)
        out = store_path(args.out_dir, args.year, lon, lat)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t = encode_tile(tile, out, codec, stats=not args.no_quality,
                        batch=args.batch,
                        log=lambda m: print(m, file=sys.stderr))
        print(json.dumps(dict(t, out=out, source=tile.source)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
