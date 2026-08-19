"""COG asset container for AlphaEarth tile blobs — the AE archive format.

This layout is a RESPONSE TO THE AEF SOURCE, not a general georvq rule:
the source ships as 1024 px-blocked band-interleaved COGs, so the
archive mirrors that structure 1:1 (same ground quantum at every pyramid
level) and speaks the ecosystem the source's users already use. A
different source (DEM, S2) should re-derive its own container from its
own access patterns; the measurements behind these choices are in
docs/AE_RELEASE.md.

One encoded tile = a directory (or object prefix) of georeferenced COGs,
1:1 with the source tile, all blocks aligned to the source's 1024 px
ground quantum:

  coarse.tif   grid/64 px cells,  8 uint16 bands (RVQ levels), 16px blocks
  mid.tif      grid/16,           8 uint16 bands,              64px blocks
  fine.tif     grid/4,            8 uint16 bands,             256px blocks
  star.tif     grid/64, 66 uint8 bands (64 packed 2-bit residual code
               bytes + f16 step) — the 8* coarse-residual sidecar
  mask.tif     full res, 1 uint8 band (validity), 1024px blocks
  metadata.json  source profile, quality stats, bitstream_id, shapes

Bands are RVQ levels, so rate control is band selection and depth-
truncated reads never touch deeper bands; spatial windows map to
internal tiles, so region reads are ranged COG access (rasterio /
GDAL /vsicurl do the client work). Values are code indices: NEVER
interpolate or build averaged overviews. Assets are written north-up
regardless of source orientation (metadata records the source layout so
decode reproduces the original file byte-layout exactly).

bitstream_id rides in every asset's TIFF tags; readers refuse
mismatched models (same-architecture codebooks decode each other's
indices into silent garbage).
"""
from __future__ import annotations

import json
import os

import numpy as np

NAMES = ("coarse", "mid", "fine", "tier3", "tier4")
GROUND_BLOCK = 1024          # px: mirror the source COGs' block quantum
META_NAME = "metadata.json"


def tier_names(n):
    return list(NAMES[:n])


def _rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError as e:
        raise SystemExit("the COG tile container needs rasterio: "
                         "pip install 'georvq[eval]'") from e


def _northup_transform(geo, cellpx):
    """Asset transform at cellpx ground pixels/cell, always north-up."""
    from rasterio.transform import Affine
    a, b, x0, d, e, y0 = geo["transform"]
    H = geo["src_height"]
    if e > 0:                      # bottom-up source: top edge = y0 + H*e
        y_top, e = y0 + H * e, -e
    else:
        y_top = y0
    return Affine(a * cellpx, b, x0, d, e * cellpx, y_top)


QUALITY_SCALE = 1e-5     # uint16 quality bands: value * SCALE = mean dcos


def write_assets(outdir, packets, keep, meta, valid, geo, specs_sig,
                 star=None, quality=None):
    """packets: kept windows' Packets. keep: their slot indices in the
    (gh*gw) window grid. valid: (H, W) bool. star: (codes, step) packed
    sidecar arrays or None. quality: (decoder_id, ladder, win_means) —
    per-window mean dcos per ladder point, written as a DECODER-SPECIFIC
    asset quality/<decoder_id>.tif (other decoders — deblocked, lite —
    publish alongside rather than overwrite). Returns per-asset sizes."""
    rasterio = _rasterio()
    os.makedirs(outdir, exist_ok=True)
    gh, gw = meta["grid"]
    W = meta["window"]

    def write(name, arr, cellpx):
        blockcells = max(16, GROUND_BLOCK // cellpx)
        prof = {"driver": "GTiff", "height": arr.shape[0],
                "width": arr.shape[1], "count": arr.shape[2],
                "dtype": str(arr.dtype),
                "compress": "deflate", "interleave": "band",
                "transform": _northup_transform(geo, cellpx)}
        if min(arr.shape[:2]) >= blockcells:   # tiny rasters: strips
            prof.update(tiled=True, blockxsize=blockcells,
                        blockysize=blockcells)
        if geo.get("crs"):
            prof["crs"] = geo["crs"]
        path = os.path.join(outdir, f"{name}.tif")
        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), \
                rasterio.open(path, "w", **prof) as ds:
            ds.write(arr.transpose(2, 0, 1))
            ds.update_tags(BITSTREAM_ID=meta["bitstream_id"],
                           GEORVQ_ROLE=name,
                           NOTE="RVQ code indices - never interpolate")
        return os.path.getsize(path)

    sizes = {}
    for ti, name in enumerate(tier_names(len(specs_sig))):
        cells, depth, codes = specs_sig[ti]
        g = int(round(cells ** 0.5))
        grid = np.zeros((gh * g, gw * g, depth), np.uint16)
        idx = np.stack([np.asarray(p.indices[ti]) for p in packets])
        for k, w in enumerate(keep):
            r, c = divmod(int(w), gw)
            grid[r*g:(r+1)*g, c*g:(c+1)*g] = idx[k].reshape(g, g, depth)
        sizes[name] = write(name, grid, W // g)
    if star is not None:
        codes, step = star
        sg = np.zeros((gh, gw, codes.shape[1] + 2), np.uint8)
        for k, w in enumerate(keep):
            r, c = divmod(int(w), gw)
            sg[r, c, :codes.shape[1]] = codes[k]
            sg[r, c, codes.shape[1]:] = np.frombuffer(
                step[k].tobytes(), np.uint8)
        sizes["star"] = write("star", sg, W)
    sizes["mask"] = write("mask",
                          valid.astype(np.uint8)[:, :, None], 1)
    if quality is not None:
        did, ladder, wm = quality
        names = ["-".join(map(str, t)) for t in ladder]
        q = np.zeros((gh, gw, len(ladder)), np.uint16)
        for bi, t in enumerate(ladder):
            v = np.clip(np.round(wm[t] / QUALITY_SCALE), 0, 65535)
            plane = np.zeros(gh * gw, np.uint16)
            plane[keep] = v.astype(np.uint16)
            q[:, :, bi] = plane.reshape(gh, gw)
        os.makedirs(os.path.join(outdir, "quality"), exist_ok=True)
        sizes[f"quality/{did}"] = write(f"quality/{did}", q, W)
        # band names + scale ride in the asset itself
        rio = _rasterio()
        with rio.open(os.path.join(outdir, f"quality/{did}.tif"),
                      "r+") as ds:
            ds.update_tags(QUALITY_SCALE=QUALITY_SCALE, DECODER_ID=did,
                           NOTE="per-window mean dcos per ladder point")
            for bi, n in enumerate(names):
                ds.set_band_description(bi + 1, n)
        meta = {**meta, "quality_rasters":
                {**meta.get("quality_rasters", {}),
                 did: {"points": names, "scale": QUALITY_SCALE}}}
    with open(os.path.join(outdir, META_NAME), "w") as f:
        json.dump(meta, f, indent=1)
    return sizes


def read_meta(indir):
    with open(os.path.join(indir, META_NAME)) as f:
        return json.load(f)


def read_assets(indir, depths=None, expect_bitstream=None):
    """-> (meta, packets [kept windows], keep slots, valid (H, W) bool).

    depths: per-tier level counts — only those bands are read (and the
    star asset only when the coarse tier is at full depth)."""
    rasterio = _rasterio()
    from georvq.inference.api import Packet
    meta = read_meta(indir)
    sig = [tuple(s) for s in meta["specs_sig"]]
    depths = depths or [d for _, d, _ in sig]
    gh, gw = meta["grid"]
    W = meta["window"]
    H, Wd = meta["shape"][1:]

    def openr(name):
        """Open one asset, refusing a bitstream_id mismatch.

        This check is load-bearing rather than defensive: codebooks of the
        same tier layout will happily decode each other's indices into
        plausible-looking garbage, so a mismatch cannot be detected
        downstream by looking at the output.
        """
        ds = rasterio.open(os.path.join(indir, f"{name}.tif"))
        tag = ds.tags().get("BITSTREAM_ID")
        if expect_bitstream and tag and tag != expect_bitstream:
            raise SystemExit(
                f"{indir}/{name}.tif: encoded against bitstream {tag}, "
                f"--model has {expect_bitstream} — same layout, different "
                "codebooks would decode to garbage. Use the matching "
                "release.")
        return ds

    with openr("mask") as ds:
        valid = ds.read(1).astype(bool)[:H, :Wd]
    padv = np.zeros((gh * W, gw * W), bool)
    padv[:H, :Wd] = valid
    keep = np.nonzero(padv.reshape(gh, W, gw, W).transpose(0, 2, 1, 3)
                      .reshape(gh * gw, W, W).any((1, 2)))[0]

    tiers = []
    for ti, (name, (cells, depth, _)) in enumerate(
            zip(tier_names(len(sig)), sig)):
        L = depths[ti]
        g = int(round(cells ** 0.5))
        if L == 0:
            tiers.append(np.zeros((len(keep), cells, 0), np.int64))
            continue
        with openr(name) as ds:
            grid = ds.read(indexes=list(range(1, L + 1)))  # (L, gh*g, gw*g)
        arr = np.empty((len(keep), cells, L), np.int64)
        for k, w in enumerate(keep):
            r, c = divmod(int(w), gw)
            arr[k] = grid[:, r*g:(r+1)*g, c*g:(c+1)*g] \
                .reshape(L, cells).T
        tiers.append(arr)

    star = None
    if depths[0] == sig[0][1] and \
            os.path.exists(os.path.join(indir, "star.tif")):
        from georvq.ae.cli import unpack_star
        with openr("star") as ds:
            sg = ds.read()                      # (66, gh, gw)
        nb = sg.shape[0] - 2
        codes = sg[:nb].reshape(nb, -1).T[keep].astype(np.uint8)
        step = np.ascontiguousarray(
            sg[nb:].reshape(2, -1).T[keep]).view(np.float16)[:, 0]
        cells = sig[0][0]
        star = unpack_star(codes, step, cells, nb * 8 // (2 * cells))

    packets = [Packet(indices=[t[i] for t in tiers],
                      refinements=({0: star[i]} if star is not None else None))
               for i in range(len(keep))]
    return meta, packets, keep, valid
