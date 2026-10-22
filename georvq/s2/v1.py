"""S2 v1 release layer: Sentinel-2 L1C SAFE products as georvq Zarr stores.

An S2 v1 store holds one L1C product (one 100 km tile, `<name>.SAFE`)
coded by the `beta_arch_w36` model: 36 px windows on the 10 m grid (18 px
at 20 m, 6 px at 60 m), a coarse tier of one 128-d vector per window and
a fine tier of one vector per 60 m pixel, both 8 levels of 8192 codes.
The model reads the 13 bands on their native grids, each window
normalised per band by the mean and standard deviation of its valid
pixels, and decodes the three grids separately. Those 26 statistics
travel in each window's base record through an `RVQPCAMeanChannel` with
a per-band transform (means in DN over a per-band error budget, standard
deviations in log space), so the decoded pixels are the decoder's
standardised field times the decoded deviation plus the decoded mean.

The store layout is that of every release (georvq.inference.records,
docs/GEORVQ_ZARR.md), with 2196 px chunks for `base`, the fine tier and
the mask (5 x 5 chunks per 10980 px tile; 1098 px of the 20 m grid and
366 px of the 60 m grid), and

- `quality/<decoder_id>`, chunked as base: per window and per rung the
  RMSE in DN of the uint16 output against the source DN, one statistic
  per grid (`rmse_10m`, `rmse_20m`, `rmse_60m`), over the window's valid
  pixels and that grid's bands, measured at encode time. Per grid
  because the grids carry different bands at different pixel counts, a
  pooled RMSE would be set by the 10 m bands, and a reader of the 60 m
  bands wants their own error;
- `assets/safe_meta`, a zip of everything in the SAFE but the 13 band
  images (georvq.s2.safe.safe_meta_zip), so that `reconstruct_safe`
  writes the product back.

Output is uint16 DN on the three native grids (header `output.grids`),
in the convention of the input: the radiometric offset of processing
baselines 04.00 and later (-1000 DN, MTD_MSIL1C.xml RADIO_ADD_OFFSET)
is not removed, since each window's statistics carry it. Valid pixels
are rounded and clipped to 1..65535, and nodata is 0 as in L1C. A pixel
is valid on the 10 m grid when all 13 bands hold data there (a coarser
pixel covering it counts), and on a coarser grid when every 10 m pixel
it covers is, so near a swath edge the store's nodata reaches a little
further than each band's own.

`SafeProduct` reads a SAFE directory as a ProductSource; `encode_safe`
writes its store; `reconstruct_safe` writes a SAFE directory back from
a store. Stores made from ESA SAFE products carry "Contains modified
Copernicus Sentinel data <year>" in their header's `licence`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from georvq.inference.store_codec import (OnnxStoreCodec, ReleaseSpec,
                                          StoreCodec, _as_arrays, _no_grad,
                                          crs_from_epsg, cumulative,
                                          encode_product, quality_code)
from georvq.s2 import safe as _safe
from georvq.s2.safe import (BANDS, GRIDS, S2_ORDER, safe_meta_zip,
                            safe_name)

KIND = "sentinel2-l1c"
FORMAT = "georvq-s2/v1"
BUNDLE_FORMAT = "georvq-s2/v1-bundle"
RELEASE_NAME = "s2-v1"
# not yet published: the bundle's repository is the maintainer's to create
RELEASE_URI = "hf://earth-compress/georvq-s2-v1"
ENV_BUNDLE = "GEORVQ_S2_BUNDLE"
CHUNK_PX = 2196
STRIP_PX = 2196
MAX_VALUE = 10000.0       # georvq.sources.s2.MAX_VALUE: DN per unit of x
FACTORS = tuple(f for _, f, _ in GRIDS)
QUALITY_STATS = tuple(f"rmse_{g}" for g, _, _ in GRIDS)
Q_SCALE, Q_OFFSET, Q_FLOOR = 32.0, 2.0, 0.01
META_ASSET = "safe_meta"
WEIGHTS_LICENCE = "Apache-2.0"
TRAINING_DATA = ("Major TOM Core-S2L1C (CC-BY-SA 4.0), "
                 "https://huggingface.co/datasets/Major-TOM/Core-S2L1C; "
                 "contains modified Copernicus Sentinel data")
COPERNICUS_NOTICE = ("https://sentinels.copernicus.eu/documents/247904/"
                     "690755/Sentinel_Data_Legal_Notice")


def licence_of(year) -> dict:
    """The header `licence` of a store coded from an ESA SAFE product
    sensed in `year`."""
    return {"spdx": None, "name": "Copernicus Sentinel data terms",
            "url": COPERNICUS_NOTICE,
            "attribution": f"Contains modified Copernicus Sentinel data "
                           f"{year}",
            "changes": "lossy georvq encoding of the 13 L1C band images; "
                       "the decoded values are approximations"}


# --------------------------------------------------------------------------
# the release's hooks
# --------------------------------------------------------------------------

def _prep(model, arrays, device):
    """Raw uint16 windows (r10, r20, r60) -> the model's normalised input
    and 10 m validity, through georvq.sources.s2.S2Source.prep, which
    also keeps each window's statistics (`last_stats`)."""
    import torch
    raw = {g: torch.from_numpy(np.ascontiguousarray(a).astype(np.int32))
           for (g, _, _), a in zip(GRIDS, arrays)}
    return model.source.prep(raw, device)


def raw_valid_np(arrays) -> np.ndarray:
    """(H, W) validity on the 10 m grid: every band of every grid holds
    data (DN != 0) at the pixel, as georvq.sources.s2.pixel_valid."""
    v = None
    for a, f in zip(arrays, FACTORS):
        g = (np.asarray(a) != 0).all(0)
        if f > 1:
            g = g.repeat(f, 0).repeat(f, 1)
        v = g if v is None else v & g
    return v


def _invalidate(arrays, mask):
    """Set to nodata the 10 m pixels of `mask`, and the coarser pixels
    every 10 m pixel of which is in it."""
    out = []
    H, W = mask.shape
    for a, f in zip(arrays, FACTORS):
        a = np.array(a, copy=True)
        m = mask if f == 1 else mask.reshape(H // f, f, W // f, f).all((1, 3))
        a[:, m] = 0
        out.append(a)
    return tuple(out)


def _check_raw(arrays, bands):
    if len(arrays) != len(GRIDS):
        raise ValueError("S2 pixels are three uint16 arrays (10, 20, 60 m)")
    h, w = arrays[0].shape[-2:]
    for a, (g, f, names) in zip(arrays, GRIDS):
        if a.dtype != np.uint16 or a.ndim != 3 or a.shape[0] != len(names):
            raise ValueError(f"{g}: expected uint16 ({len(names)}, h, w), got "
                             f"{a.dtype} {a.shape}")
        if a.shape[1:] != (h // f, w // f) or h % f or w % f:
            raise ValueError(f"{g}: shape {a.shape[1:]} does not match the "
                             f"10 m grid {h} x {w} at factor {f}")


def _pool_valid(valid, f):
    """(n, h, w) bool -> (n, h / f, w / f), True where all of each f x f
    block is."""
    if f == 1:
        return valid
    if isinstance(valid, np.ndarray):
        n, h, w = valid.shape
        return valid.reshape(n, h // f, f, w // f, f).all((2, 4))
    import torch.nn.functional as F
    return F.max_pool2d((~valid).float()[:, None], f)[:, 0] < 0.5


def postprocess(out, valid):
    """Decoded field {grid: (n, C, w, w)} in DN (a torch tensor) -> {grid:
    uint16 numpy}, rounded and clipped to 1..65535 at valid pixels, 0 at
    invalid ones."""
    import torch
    res = {}
    for g, f, _ in GRIDS:
        v = _pool_valid(valid, f)
        p = out[g].float().round().clamp(1, 65535)
        p = torch.where(v[:, None], p, torch.zeros((), device=p.device))
        res[g] = p.to(torch.int32).cpu().numpy().astype(np.uint16)
    return res


def postprocess_np(out, valid):
    valid = np.asarray(valid, bool)
    res = {}
    for g, f, _ in GRIDS:
        v = _pool_valid(valid, f)
        p = np.clip(np.rint(np.asarray(out[g], np.float32)), 1, 65535)
        p[np.broadcast_to(~v[:, None], p.shape)] = 0
        res[g] = p.astype(np.uint16)
    return res


def _check_model(model, channel):
    src = getattr(model, "source", None)
    if type(src).__name__ != "S2Source":
        raise ValueError("S2 v1 needs a model of the s2 source")
    if model.stats_cond is not None:
        raise ValueError("S2 v1 ships the S2 normaliser's statistics, not a "
                         "stats_cond mean")
    if model.conditioner is not None or model.selector is not None:
        raise ValueError("S2 v1 ships codecs without a location prior or "
                         "selector")
    if model.window % 6 or len(model.specs) != 2:
        raise ValueError("S2 v1 expects two tiers and a window divisible by 6")
    if channel.bands != 2 * len(BANDS):
        raise ValueError(f"the S2 stats channel codes {2 * len(BANDS)} "
                         f"values, this one {channel.bands}")


def _bundle_readme(meta) -> str:
    tiers = "\n".join(f"- {t}" for t in meta["tiers"])
    rows = "\n".join(
        f"| {'-'.join(map(str, r['tuple']))} | {r['bits_window']} | "
        f"{r['bits_px']:.4f} |" for r in meta["rungs"])
    size = sum(f["bytes"] for f in meta["files"].values())
    src = meta.get("source") or {}
    onnx = (f"\n`{meta['onnx']}/` holds the ONNX export used by the torch-free "
            "backend (`backend=\"onnx\"`).\n" if meta.get("onnx") else "")
    return f"""# georvq Sentinel-2 v1 bundle

<!-- TODO(maintainer): draft text, to be replaced by the model card of
phase 2. -->

Licence of the weights: Apache-2.0. The model was trained on {TRAINING_DATA}.
Stores coded from ESA SAFE products contain modified Copernicus Sentinel
data of their sensing year, as each store's header states.

Decoder and encoder for Sentinel-2 L1C products stored as georvq Zarr
stores (`{meta['codec_format']}`, see docs/GEORVQ_ZARR.md).

- bitstream_id `{meta['bitstream_id']}`, decoder_id `{meta['decoder_id']}`
- window {meta['window']} px at 10 m, {meta['bands']} bands on the 10, 20
  and 60 m grids
- source checkpoint {src.get('checkpoint')} at step {src.get('step')}

Tiers (grid:dim:codes:depth):

{tiers}

Every window also carries the mean and standard deviation of each band,
{meta['mean_channel']['bits']} bits through the
{meta['mean_channel']['kind']} channel. The rate of each rung below
includes them; bits/px are per 10 m pixel.

| rung | bits/window | bits/px |
|---|---|---|
{rows}

Files ({size / 1e6:.1f} MB): `weights.npz` (model tensors), `codebooks.npz`
(float32 codebooks, `<level>.weight`), `mean_channel.npz` (the statistics
channel), `config.json`, `bundle.json`. All arrays load with
`allow_pickle=False`.
{onnx}
```python
import georvq
ds = georvq.open("S2A_MSIL1C_..._T33TTG_....georvq.zarr")
b04 = ds.read(3, quality="8-4")          # 10 m grid: B02 B03 B04 B08
swir = ds.grid("20m").read(quality="8-4")
```
"""


S2_V1 = ReleaseSpec(
    name=RELEASE_NAME, codec_format=FORMAT, product_kind=KIND,
    bundle_format=BUNDLE_FORMAT, onnx_meta_key="s2_v1", bands=len(BANDS),
    prep=_prep, raw_valid=raw_valid_np, invalidate=_invalidate,
    check_raw=_check_raw, pad=(0, 0, 0),
    postprocess=postprocess, postprocess_np=postprocess_np,
    out_dtype="uint16", out_nodata=0, band_names=S2_ORDER,
    release_uri=RELEASE_URI, env_bundle=ENV_BUNDLE,
    chunk_px=(CHUNK_PX, CHUNK_PX, CHUNK_PX), strip_px=STRIP_PX,
    quality_stats=QUALITY_STATS,
    quality_error_text="RMSE in DN of the uint16 output against the source "
                       "DN, over the window's valid pixels of one grid "
                       "(rmse_10m: B02 B03 B04 B08; rmse_20m: B05 B06 B07 "
                       "B8A B11 B12; rmse_60m: B01 B09 B10), all of the "
                       "grid's bands pooled",
    quality_encoding_text="code = round(32 * (log10(e) + 2)) clipped to "
                          "0..255; 0 means e <= 0.01 DN",
    quality_scale=Q_SCALE, quality_offset=Q_OFFSET, quality_floor=Q_FLOOR,
    check_model=_check_model, bundle_readme=_bundle_readme,
    raw_factors=FACTORS, output_grids=GRIDS, channel_values=2 * len(BANDS))


def stats_vector(stats) -> np.ndarray:
    """georvq.sources.s2.S2Normalizer statistics -> (B, 26) float32: the
    13 means, then the 13 standard deviations, in the model's band order,
    in units of x (DN / 10000)."""
    import torch
    mus = torch.cat([stats[g][0].flatten(1) for g, _, _ in GRIDS], 1)
    sds = torch.cat([stats[g][1].flatten(1) for g, _, _ in GRIDS], 1)
    return torch.cat([mus, sds], 1).float().cpu().numpy()


class S2V1Codec(StoreCodec):
    """The S2 model and its statistics channel as S2 v1 records. The
    channel's fields replace the stats_cond mean of other releases at the
    head of the base record; the encoder sees the window normalised by
    its exact statistics, and the decoder's output is denormalised by the
    decoded ones."""

    spec = S2_V1

    def __init__(self, model, channel, bf16=None, rungs=None, spec=None):
        super().__init__(model, channel, bf16=bf16, rungs=rungs,
                         spec=spec or S2_V1)

    @_no_grad
    def encode_windows(self, raw, stats=True):
        arrays = _as_arrays(raw)
        x, valid = self.spec.prep(self.model, arrays, self.device)
        vec = stats_vector(self.model.source.last_stats())
        fields = self.channel.encode(vec)
        stacked, _, _ = self.codec.encode_batch(x, valid=valid)
        B = vec.shape[0]
        coarse = stacked[0].transpose(0, 2, 1).reshape(B, -1)
        rec = {"base": np.concatenate([fields, coarse], 1)[None]}
        for name, idx in zip(self.names[1:], stacked[1:]):
            rec[name] = idx.transpose(2, 0, 1)
        if stats:
            rec["quality"] = self._quality_dn(stacked, fields, arrays,
                                              valid)[None]
        return rec, x, valid

    def _quality_dn(self, stacked, fields, arrays, valid):
        """(B, 3 * n_rungs) codes: per grid, the RMSE in DN of the output
        at each rung."""
        import torch
        dev = self.device
        stats = torch.from_numpy(self.channel.decode(fields)).to(dev)
        idx = [torch.from_numpy(t).to(dev) for t in stacked]
        cums = [cumulative(rvq, t, sp.depth)
                for rvq, t, sp in zip(self.model.rvqs, idx, self.specs)]
        src = {g: torch.from_numpy(np.ascontiguousarray(a).astype(np.float32))
               .to(dev) for (g, _, _), a in zip(GRIDS, arrays)}
        per = {g: [] for g, _, _ in GRIDS}
        for r in self.rungs:
            out = self._forward([c[L] for c, L in zip(cums, r)], r, stats,
                                valid)
            for g, f, _ in GRIDS:
                v = _pool_valid(valid, f)[:, None].float()
                p = out[g].float().round().clamp(1, 65535)
                se = ((p - src[g]) ** 2 * v).sum((1, 2, 3))
                n = v.sum((1, 2, 3)) * src[g].shape[1]
                e = torch.where(n > 0, (se / n.clamp_min(1)).sqrt(),
                                torch.full_like(se, float("nan")))
                per[g].append(e)
        codes = [quality_code(torch.stack(per[g], 1).cpu().numpy(), Q_SCALE,
                              Q_OFFSET, Q_FLOOR) for g, _, _ in GRIDS]
        return np.concatenate(codes, 1).astype(np.int64)

    def _forward(self, zqs, depths, stats, valid):
        """Decoder pass at a depth tuple, denormalised by the decoded
        statistics (B, 26): {grid: (B, C, w, w)} float32 in DN."""
        import torch
        B = stats.shape[0]
        Ls = torch.tensor(list(depths), device=self.device)[None].expand(B, -1)
        with self._ac():
            out = self.model.decode(zqs, Ls, stats=None, valid=valid)
        nb = len(BANDS)
        res, i = {}, 0
        for g, _, names in GRIDS:
            k = len(names)
            mu = stats[:, i:i + k, None, None]
            sd = stats[:, nb + i:nb + i + k, None, None]
            res[g] = (out[g].float() * sd + mu) * MAX_VALUE
            i += k
        return res


class S2V1OnnxCodec(OnnxStoreCodec):
    spec = S2_V1

    def __init__(self, onnx_dir, device="cpu", threads=None, spec=None):
        raise NotImplementedError("S2 v1 has no ONNX export; use the torch "
                                  "backend")


S2_V1 = S2_V1.with_(torch_codec=S2V1Codec, onnx_codec=S2V1OnnxCodec)
S2V1Codec.spec = S2V1OnnxCodec.spec = S2_V1
SPEC = S2_V1


# --------------------------------------------------------------------------
# SAFE products
# --------------------------------------------------------------------------

class SafeProduct:
    """An L1C SAFE directory as a georvq.inference.store_codec.
    ProductSource: north-up strips of its 13 bands as (10 m, 20 m, 60 m)
    uint16 arrays, georeferenced by the band images (one CRS, the 10 m
    transform as the product's). `source` is the path or URL the header
    records (default the directory)."""

    def __init__(self, safe_dir, source=None, threads=13):
        safe_dir = os.fspath(safe_dir).rstrip("/")
        self.safe_dir = safe_dir
        self.name = safe_name(safe_dir)
        self.files = _safe.band_files(safe_dir)
        got = _safe.read_bands(safe_dir, self.files, threads=threads)
        epsg, t, (h, w) = _safe.check_geometry({b: v[1] for b, v in
                                                got.items()})
        self.arrays = tuple(np.stack([got[b][0] for b in names])
                            for _, _, names in GRIDS)
        self.height, self.width = h, w
        self.transform = t
        self.crs = crs_from_epsg(epsg)
        self.orientation = "north-up"
        self.product_id = self.name
        self.source = os.fspath(source if source is not None else safe_dir)
        meta = _safe.product_metadata(safe_dir)
        self.year = _year(meta.get("sensing_start"), self.name)
        jp2 = {b: _safe.jp2_params(os.path.join(safe_dir, self.files[b]))
               for b in BANDS}
        self.product_meta = {
            "count": len(BANDS), "dtype": "uint16", "nodata": 0,
            "band_names": list(S2_ORDER), "safe": self.name + ".SAFE",
            **{k: v for k, v in meta.items() if v is not None}}
        self.aux = {"safe": {"name": self.name + ".SAFE",
                             "band_files": {b: self.files[b]
                                            for b in S2_ORDER},
                             "jp2": {b: jp2[b] for b in S2_ORDER},
                             "meta_asset": META_ASSET}}

    def read_strip(self, y0, y1):
        return tuple(a[:, y0 // f:y1 // f]
                     for a, f in zip(self.arrays, FACTORS))


def _year(sensing_start, name):
    if sensing_start:
        return int(sensing_start[:4])
    # S2A_MSIL1C_20250531T100051_...: the sensing date is the third field
    return int(name.split("_")[2][:4])


def encode_safe(safe_dir, out_path, codec: S2V1Codec, stats=True,
                chunk_px=None, strip_px=None, batch=256, release=RELEASE_NAME,
                uri=RELEASE_URI, revision=None, source=None, log=None,
                product=None) -> dict:
    """Encode one L1C SAFE product to the store `out_path`, keeping the
    rest of the product as the `safe_meta` asset. `product`, when given,
    is a SafeProduct of `safe_dir` already read. Returns timings."""
    src = product or SafeProduct(safe_dir, source=source)
    meta = safe_meta_zip(src.safe_dir)
    assets = {META_ASSET: (meta, "application/zip",
                           f"{src.name}.SAFE.meta.zip")}
    return encode_product(src, out_path, codec, stats=stats,
                          chunk_px=chunk_px, strip_px=strip_px, batch=batch,
                          release=release, uri=uri, revision=revision,
                          assets=assets, log=log,
                          licence=licence_of(src.year))


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------

def reconstruct_safe(store, out_dir, quality=None, rows=STRIP_PX, log=None,
                     **open_kw) -> str:
    """Write the SAFE product a store was coded from into `out_dir`, with
    its 13 band images decoded at `quality` (a rung such as "8-4", a
    bound such as "rmse_10m<=20", or None for full depth). Returns the
    product directory, `out_dir/<name>.SAFE`, which must not exist.

    The metadata zip is extracted first, so every file but the band
    images (TCI, PVI, the QI_DATA masks, MTD_*.xml, manifest.safe,
    AUX_DATA, DATASTRIP, HTML, rep_info) is the original's, byte for
    byte. The bands are decoded strip by strip and each is written
    losslessly as JPEG 2000 with the coding parameters of the original
    image (tile size, resolution levels, progression, code-blocks,
    precincts, 15-bit samples) and a GML box with the original's
    georeferencing, under its original name. Decoded values above the
    largest value the original's samples hold (32767 for 15 bits) are
    clipped to it. In manifest.safe the size and checksum (with the
    algorithm it names, SHA3-256 or MD5) of each band image are
    recomputed, and an XML comment after the declaration says that the
    product is a georvq reconstruction at the given quality. A file
    `GEORVQ_RECONSTRUCTION.json` at the product root records the store,
    its bitstream and decoder ids, the quality, and the files changed.
    `open_kw` go to georvq.open (bundle=, backend=, device=, ...)."""
    import georvq
    say = log or (lambda m: None)
    ds = store if hasattr(store, "read_grids") else georvq.open(store,
                                                                **open_kw)
    try:
        aux = ds.source_meta.get("safe")
        if aux is None or META_ASSET not in ds.assets:
            raise ValueError(f"{ds.name}: not an S2 v1 store of a SAFE "
                             "product (no safe_meta asset)")
        root = os.path.join(os.fspath(out_dir), aux["name"])
        _safe.unzip_meta(ds.assets[META_ASSET], root)
        say(f"metadata extracted to {root}")
        H, W = ds.product_height, ds.product_width
        full = {g: np.zeros((len(names), H // f, W // f), np.uint16)
                for g, f, names in GRIDS}
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            got = ds.read_grids(((r0, r1), (0, W)), quality=quality)
            for g, f, _ in GRIDS:
                full[g][:, r0 // f:-(-r1 // f)] = got[g]
            say(f"decoded rows {r1}/{H}")
        from affine import Affine
        t10 = Affine(*ds.product["transform"])
        written = []
        for g, f, names in GRIDS:
            for k, b in enumerate(names):
                rel = aux["band_files"][b]
                params = aux["jp2"][b]
                top = (1 << int(params["precision"])) - 1
                a = np.minimum(full[g][k], top).astype(np.uint16)
                _safe.write_band_jp2(os.path.join(root, rel), a, ds.crs,
                                     t10 * Affine.scale(f), params)
                written.append(rel)
                say(f"wrote {rel}")
        rung = ("full depth" if quality is None else quality
                if isinstance(quality, str) else "-".join(map(str, quality)))
        h = ds.header
        man = os.path.join(root, "manifest.safe")
        if os.path.exists(man):
            text = open(man, encoding="utf-8", newline="").read()
            note = (f"georvq reconstruction: the 13 band images in "
                    f"GRANULE/*/IMG_DATA were decoded from a georvq S2 v1 "
                    f"store (bitstream {h['codec']['bitstream_id']}, decoder "
                    f"{h['codec']['decoder_id']}) at quality {rung} and are "
                    f"lossy approximations of the originals; their sizes "
                    f"and checksums below were recomputed. See {_safe.MARKER}.")
            text = _safe.rewrite_manifest(text, root, written, note)
            with open(man, "w", encoding="utf-8", newline="") as f:
                f.write(text)
        marker = {
            "georvq_reconstruction": True,
            "store": ds.name, "product_id": ds.product.get("id"),
            "codec_format": h["codec"]["format"],
            "bitstream_id": h["codec"]["bitstream_id"],
            "decoder_id": getattr(ds.codec, "decoder_id", None),
            "release": h["codec"].get("release"),
            "quality": quality, "rung": rung,
            "band_images": sorted(written),
            "note": "The band images are lossy reconstructions decoded from "
                    "a georvq store; every other file is the original's, "
                    "except manifest.safe, whose sizes and checksums of the "
                    "band images were recomputed and which carries a comment "
                    "saying so.",
            "licence": h.get("licence")}
        with open(os.path.join(root, _safe.MARKER), "w") as f:
            json.dump(marker, f, indent=1)
            f.write("\n")
        return root
    finally:
        if ds is not store:
            ds.close()


# --------------------------------------------------------------------------
# bundles
# --------------------------------------------------------------------------

def build_bundle(ckpt, channel_npz, out_dir, notes="") -> dict:
    """Bundle a training checkpoint and its statistics channel."""
    from georvq.inference.bundle import build_bundle as _build
    return _build(ckpt, channel_npz, out_dir, S2_V1, notes=notes, onnx=False,
                  extra={"licence_weights": WEIGHTS_LICENCE,
                         "training_data": TRAINING_DATA,
                         "licence_stores": "per store, in its header: "
                                           "Contains modified Copernicus "
                                           "Sentinel data <year>"})


def load_bundle(bundle_dir, device="cpu", bf16=None, verify=True):
    from georvq.inference.bundle import load_bundle as _load
    return _load(bundle_dir, S2_V1, device, bf16, verify)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="georvq s2",
        description="Sentinel-2 v1 codec: one L1C SAFE product <-> one "
                    "georvq Zarr store.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="SAFE products -> stores")
    e.add_argument("out_dir")
    e.add_argument("safe", nargs="+")
    e.add_argument("--bundle", required=True)
    e.add_argument("--device", default=None)
    e.add_argument("--no-quality", action="store_true")
    e.add_argument("--batch", type=int, default=256)
    r = sub.add_parser("reconstruct", help="store -> SAFE product")
    r.add_argument("store")
    r.add_argument("out_dir")
    r.add_argument("--quality", default=None,
                   help="a rung such as 8-4, or a bound such as "
                        "rmse_10m<=20 (default full depth)")
    r.add_argument("--backend", default=None, choices=["torch"])
    r.add_argument("--device", default=None)
    r.add_argument("--bundle", default=None)
    b = sub.add_parser("bundle", help="checkpoint + stats channel -> bundle")
    b.add_argument("ckpt")
    b.add_argument("channel_npz")
    b.add_argument("out_dir")
    b.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    def log(m):
        print(m, file=sys.stderr)

    if args.cmd == "bundle":
        meta = build_bundle(args.ckpt, args.channel_npz, args.out_dir,
                            notes=args.notes)
        print(json.dumps({k: meta[k] for k in ("bitstream_id", "decoder_id")}))
        return 0
    if args.cmd == "reconstruct":
        out = reconstruct_safe(args.store, args.out_dir, quality=args.quality,
                               log=log, backend=args.backend,
                               device=args.device, bundle=args.bundle)
        print(out)
        return 0
    from georvq.inference.bundle_resolve import default_device
    from georvq.inference.records import output_name
    from georvq.inference.store_codec import quiet_build
    with quiet_build():
        codec = load_bundle(args.bundle, device=args.device or
                            default_device("torch"))
    os.makedirs(args.out_dir, exist_ok=True)
    for s in args.safe:
        out = os.path.join(args.out_dir, output_name(s))
        t = encode_safe(s, out, codec, stats=not args.no_quality,
                        batch=args.batch, log=log)
        print(json.dumps(dict(t, out=out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
