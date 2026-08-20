"""Build a Copernicus DEM chip store from a local Major-TOM Core-DEM mirror.

One chip per Major-TOM grid cell: /root/[row]/[row]_[col]/DEM.tif, a single
356x356 float32 elevation band (metres, 30m GSD, UTM per tile,
nodata=-32767). Stored raw float32 — elevation needs no dequant recipe.

Usage:
  python scripts/build_store_dem.py                       # full mirror
  python scripts/build_store_dem.py --limit 3000 --out-dir /tmp/dem-subset
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from georvq.data.build import ShardWriter

NODATA = -32767.0


def fetch_chip(path):
    """Read one DEM tile. Returns (array (1, px, px), meta) or None."""
    cell = os.path.basename(os.path.dirname(path))     # e.g. "142U_67L"
    try:
        with rasterio.open(path) as ds:
            a = ds.read()
            b = ds.bounds
            lons, lats = warp_transform(
                ds.crs, "EPSG:4326",
                [(b.left + b.right) / 2], [(b.top + b.bottom) / 2])
            nd = float((a[0] == NODATA).mean())
            if nd >= 1.0:
                return None
            v = a[0][a[0] != NODATA]
            return np.ascontiguousarray(a), {
                "grid_cell": cell, "lat": lats[0], "lon": lons[0],
                "crs": str(ds.crs), "year": 2021, "nodata_frac": nd,
                "elev_mean": float(v.mean()), "elev_std": float(v.std()),
            }
    except Exception as e:  # noqa: BLE001 - unreadable tile: skip it
        print(f"  ! read failed ({type(e).__name__}): {path}: "
              f"{str(e)[:120]}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/Major-TOM/Core-DEM")
    ap.add_argument("--out-dir", default="/data/georvq-stores/copdem_356")
    ap.add_argument("--limit", type=int, default=None,
                    help="build from the first N tiles only (subset store)")
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chip-px", type=int, default=356)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "*", "*", "DEM.tif")))
    if args.limit:
        paths = paths[:args.limit]

    writer = ShardWriter(args.out_dir, args.shard_size, bands=1,
                         chip_px=args.chip_px, dtype=np.float32,
                         append=args.append)
    done = set()
    prior = writer.prior_index
    if len(prior):
        done = set(prior.grid_cell)
        paths = [p for p in paths
                 if os.path.basename(os.path.dirname(p)) not in done]
    print(f"{len(paths)} tiles -> {args.out_dir}", flush=True)

    target = len(paths)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        it = iter(paths)
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < args.workers * 2:
                p = next(it, None)
                if p is None:
                    exhausted = True
                    break
                pending.add(ex.submit(fetch_chip, p))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                res = f.result()
                if res is not None:
                    writer.add(*res, log_every=1000, target=target)

    manifest = writer.finalize(
        "dem", nodata=NODATA, grid_period=None,
        extra={
            "units": "metres above sea level (EGM2008)",
            "gsd_m": 30,
            "attribution": ("Copernicus DEM GLO-30 (ESA); Major-TOM "
                            "Core-DEM grid."),
            "license": "Copernicus DEM licence / CC-BY 4.0 (Major-TOM)",
            "builder_config": vars(args) | {"out_dir": str(args.out_dir)},
        })
    el = time.time() - t0
    print(f"\ndone: {manifest['n_chips']} chips in {el/60:.1f}min -> "
          f"{args.out_dir} "
          f"({manifest['n_chips'] * args.chip_px**2 * 4 / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
