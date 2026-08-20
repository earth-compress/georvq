"""Build a live COPDEM store: band VRTs + cell index, no pixel rewriting.

Groups the local GLO-30 COG tiles into six same-width latitude-band VRT
mosaics, then precomputes each Major-TOM Core-DEM cell's read window into
its band VRT. Total footprint: a few hundred MB of index + VRT XML over
the existing /data/COPDEM/terrain tiles.

Usage:
  python scripts/build_live_dem.py                        # full grid
  python scripts/build_live_dem.py --limit 3000           # subset (debug)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd
import rasterio

from georvq.data.live_dem import CHIP_PX, N_SRC, NODATA

RES = 1.0 / 3600.0                       # native latitude posting (deg/px)


def tile_lat(path):
    m = re.search(r"_10_([NS])(\d+)_00_", os.path.basename(path))
    return int(m.group(2)) * (1 if m.group(1) == "N" else -1)


def _headers(paths, workers=16):
    """(path, transform, width, height) per tile, read concurrently."""
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        with rasterio.open(p) as ds:
            return p, ds.transform, ds.width, ds.height, str(ds.crs.wkt)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, paths))


def _write_vrt(out, tiles, wkt):
    """Mosaic VRT over same-grid tiles (integer-aligned, no resampling)."""
    res_x = tiles[0][1].a
    res_y = tiles[0][1].e
    x0 = min(t[1].c for t in tiles)
    y0 = max(t[1].f for t in tiles)
    xs = max(int(round((t[1].c - x0) / res_x)) + t[2] for t in tiles)
    ys = max(int(round((t[1].f - y0) / res_y)) + t[3] for t in tiles)
    src = []
    for p, tr, w, h, _ in tiles:
        assert abs(tr.a - res_x) < 1e-12 and abs(tr.e - res_y) < 1e-12, p
        xo = (tr.c - x0) / res_x
        yo = (tr.f - y0) / res_y
        assert abs(xo - round(xo)) < 1e-6 and abs(yo - round(yo)) < 1e-6, \
            f"tile not grid-aligned: {p}"
        src.append(
            f'  <SimpleSource>\n'
            f'   <SourceFilename relativeToVRT="0">{p}</SourceFilename>\n'
            f'   <SourceBand>1</SourceBand>\n'
            f'   <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>\n'
            f'   <DstRect xOff="{int(round(xo))}" yOff="{int(round(yo))}" '
            f'xSize="{w}" ySize="{h}"/>\n'
            f'  </SimpleSource>')
    with open(out, "w") as f:
        f.write(
            f'<VRTDataset rasterXSize="{xs}" rasterYSize="{ys}">\n'
            f' <SRS>{wkt}</SRS>\n'
            f' <GeoTransform>{x0}, {res_x}, 0, {y0}, 0, {res_y}'
            f'</GeoTransform>\n'
            f' <VRTRasterBand dataType="Float32" band="1">\n'
            f'  <NoDataValue>{NODATA}</NoDataValue>\n'
            + "\n".join(src) + "\n"
            f' </VRTRasterBand>\n</VRTDataset>\n')


def build_vrts(terrain_dir, out_dir):
    """Group tiles by MEASURED longitude posting (the nominal lat-band
    table is hemisphere-asymmetric: tile names give the south edge, so
    e.g. S85 tiles are 720/deg while N85 are 360/deg) and mosaic each
    group. Returns (per-band vrt meta, name-lat -> band table)."""
    vrt_dir = os.path.join(out_dir, "vrt")
    os.makedirs(vrt_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(terrain_dir, "*_DEM.tif")))
    tiles = _headers(paths)
    groups, lat_band = {}, {}
    for t in tiles:
        wpd = round(1.0 / t[1].a)               # measured px per degree
        groups.setdefault(wpd, []).append(t)
        la = tile_lat(t[0])
        assert lat_band.setdefault(la, wpd) == wpd, \
            f"mixed widths at name-lat {la}"
    meta = {}
    for w, ts in sorted(groups.items()):
        out = os.path.join(vrt_dir, f"band_{w}.vrt")
        t0 = time.time()
        _write_vrt(out, ts, ts[0][4])
        with rasterio.open(out) as ds:
            assert abs(ds.res[1] - RES) < 1e-12, (w, ds.res)
            meta[w] = {"transform": list(ds.transform)[:6],
                       "width": ds.width, "height": ds.height}
        print(f"band_{w}.vrt: {len(ts)} tiles, {meta[w]['width']}x"
              f"{meta[w]['height']} ({time.time()-t0:.0f}s)", flush=True)
    return meta, lat_band


def build_index(md, vrt_meta, lat_band):
    """Per-cell (band, vrow0, vcol0): the chip's read window, centred on
    the cell centre, in its band VRT's pixel grid. band=0 marks cells in
    1-degree latitude rows with no source tile anywhere (dropped)."""
    lat = md.centre_lat.to_numpy()
    lon = md.centre_lon.to_numpy()
    band = np.array([lat_band.get(int(math.floor(la)), 0) for la in lat],
                    dtype=np.int32)
    vrow0 = np.zeros(len(md), dtype=np.int64)
    vcol0 = np.zeros(len(md), dtype=np.int64)
    for w in np.unique(band):
        if w == 0:
            continue
        m = band == w
        t = vrt_meta[int(w)]["transform"]     # [a, b, c, d, e, f]
        res_x, x0, res_y, y0 = t[0], t[2], t[4], t[5]
        # centre pixel coordinates in this VRT, then back off half a chip
        vrow0[m] = np.round((lat[m] - y0) / res_y - CHIP_PX / 2).astype(int)
        vcol0[m] = np.round((lon[m] - x0) / res_x
                            - N_SRC[int(w)] / 2).astype(int)
    return band, vrow0, vcol0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="/data/COPDEM/terrain")
    ap.add_argument("--metadata",
                    default="/data/Major-TOM/Core-DEM/metadata.parquet")
    ap.add_argument("--out-dir", default="/data/georvq-stores/copdem_live")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vrt_meta, lat_band = build_vrts(args.terrain, args.out_dir)

    md = pd.read_parquet(args.metadata)
    n_all = len(md)
    # drop cells with nothing to code: fully nodata or all-zero (open ocean)
    md = md[(md.nodata < 1.0)
            & ~((md.min_val == 0) & (md.max_val == 0))].reset_index(drop=True)
    if args.limit:
        md = md.head(args.limit)
    band, vrow0, vcol0 = build_index(md, vrt_meta, lat_band)
    covered = band > 0
    if (~covered).any():
        print(f"dropping {(~covered).sum()} cells in tile-less latitude "
              "rows", flush=True)
    md, band = md[covered].reset_index(drop=True), band[covered]
    vrow0, vcol0 = vrow0[covered], vcol0[covered]

    idx = pd.DataFrame({
        "chip_id": np.arange(len(md)),
        "shard": np.arange(len(md)) // 4096,
        "slot": np.arange(len(md)) % 4096,
        "grid_cell": md.grid_cell,
        "lat": md.centre_lat, "lon": md.centre_lon, "crs": md.crs,
        "year": 2021, "nodata_frac": md.nodata,
        "elev_min": md.min_val, "elev_max": md.max_val,
        "band": band, "vrow0": vrow0, "vcol0": vcol0,
    })
    idx.to_parquet(os.path.join(args.out_dir, "index.parquet"))

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({
            "store_version": 1, "kind": "live_dem", "source": "dem",
            "chip_px": CHIP_PX, "bands": 1, "dtype": "float32",
            "nodata": NODATA, "grid_period": None,
            "terrain_dir": args.terrain, "n_chips": len(idx),
            "n_cells_total": n_all,
            "lat_band": {str(k): v for k, v in sorted(lat_band.items())},
            "units": "metres above sea level (EGM2008)", "gsd_m": 30,
            "chip_grid": ("native 1-arcsec latitude rows; longitude "
                          "nearest-neighbour replicated from the band's "
                          "posting (invertible, see live_dem.invert_cols)"),
            "attribution": ("Copernicus DEM GLO-30 (ESA); Major-TOM "
                            "Core-DEM cell grid."),
            "license": "Copernicus DEM licence",
            "builder_config": vars(args),
        }, f, indent=1)
    print(f"live store: {len(idx)}/{n_all} cells -> {args.out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
