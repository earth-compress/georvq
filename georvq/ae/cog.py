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


def _write_cog(outdir, name, arr, cellpx, geo, bitstream_id):
    """One asset COG: arr (h, w, bands), cellpx ground px per cell."""
    rasterio = _rasterio()
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
        ds.update_tags(BITSTREAM_ID=bitstream_id,
                       GEORVQ_ROLE=name,
                       NOTE="RVQ code indices - never interpolate")
    return os.path.getsize(path)


INT_STATS = ("mean", "p50", "p90", "p99", "max")
INT_FAMILIES = ("max", "avg")   # per-pixel reduce over bands: L-inf, L1
QUALITY_INT_SCALE = 2 ** -8   # uint16 int bands: value * SCALE = int8 steps


def write_quality(outdir, meta, geo, did, ladder, win_means, keep,
                  win_int=None):
    """The decoder-specific quality asset quality/<decoder_id>.tif —
    per-window mean dcos per ladder point, plus (when `win_int` is
    given, shaped (n, 2, 5) per point) ten bands per point of the
    int-domain secondary metrics: per-window mean/p50/p90/p99/max of
    the per-pixel |recon_int8 - src_int8| reduced over bands two ways —
    max (int_max_*, worst-band guarantee) and mean (int_avg_*, average
    band error) — in quantizer steps, fixed-point 1/256. Updates the
    meta dict (returned; caller persists metadata.json). Callable at
    encode time or by a later `restat` pass (variant decoders publish
    alongside rather than overwrite)."""
    gh, gw = meta["grid"]
    names = ["-".join(map(str, t)) for t in ladder]

    def plane(vals, scale):
        p = np.zeros(gh * gw, np.uint16)
        p[keep] = np.clip(np.round(vals / scale), 0, 65535) \
            .astype(np.uint16)
        return p.reshape(gh, gw)

    bands, descs = [], []
    for bi, t in enumerate(ladder):
        bands.append(plane(win_means[t], QUALITY_SCALE))
        descs.append(names[bi])
    if win_int is not None:
        for bi, t in enumerate(ladder):
            for fi, fam in enumerate(INT_FAMILIES):
                for si, stat in enumerate(INT_STATS):
                    bands.append(plane(win_int[t][:, fi, si],
                                       QUALITY_INT_SCALE))
                    descs.append(f"int_{fam}_{stat}:{names[bi]}")
    q = np.stack(bands, -1)
    os.makedirs(os.path.join(outdir, "quality"), exist_ok=True)
    size = _write_cog(outdir, f"quality/{did}", q, meta["window"], geo,
                      meta["bitstream_id"])
    rio = _rasterio()
    with rio.open(os.path.join(outdir, f"quality/{did}.tif"), "r+") as ds:
        ds.update_tags(QUALITY_SCALE=QUALITY_SCALE, DECODER_ID=did,
                       QUALITY_INT_SCALE=QUALITY_INT_SCALE,
                       NOTE="per-window mean dcos per ladder point; "
                            "int_{max,avg}_* bands = per-pixel "
                            "|recon-src| int8 steps, max/mean over "
                            "bands")
        for bi, n in enumerate(descs):
            ds.set_band_description(bi + 1, n)
    entry = {"points": names, "scale": QUALITY_SCALE}
    if win_int is not None:
        entry.update(int_stats=list(INT_STATS),
                     int_families=list(INT_FAMILIES),
                     int_scale=QUALITY_INT_SCALE)
    meta = {**meta, "quality_rasters":
            {**meta.get("quality_rasters", {}), did: entry}}
    return meta, size


def write_assets(outdir, packets, keep, meta, valid, geo, specs_sig,
                 star=None, quality=None):
    """packets: kept windows' Packets. keep: their slot indices in the
    (gh*gw) window grid. valid: (H, W) bool. star: (codes, step) packed
    sidecar arrays or None. quality: (decoder_id, ladder, win_means) —
    per-window mean dcos per ladder point, written as a DECODER-SPECIFIC
    asset quality/<decoder_id>.tif (other decoders — deblocked, lite —
    publish alongside rather than overwrite). Returns per-asset sizes."""
    os.makedirs(outdir, exist_ok=True)
    gh, gw = meta["grid"]
    W = meta["window"]

    def write(name, arr, cellpx):
        return _write_cog(outdir, name, arr, cellpx, geo,
                          meta["bitstream_id"])

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
        meta, sizes[f"quality/{did}"] = write_quality(
            outdir, meta, geo, did, ladder, wm, keep)
    with open(os.path.join(outdir, META_NAME), "w") as f:
        json.dump(meta, f, indent=1)
    return sizes


def update_meta(indir, meta):
    with open(os.path.join(indir, META_NAME), "w") as f:
        json.dump(meta, f, indent=1)


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
                      coarse_residual=star[i] if star is not None else None)
               for i in range(len(keep))]
    return meta, packets, keep, valid
