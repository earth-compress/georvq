"""Build a NextGEMS ICON (ngc4008) store of precipitation (pr) and
outgoing longwave radiation (rlut) on the 0.125-degree grid that
ClimateBenchPress evaluates on.

The DKRZ zarr for HEALPix zoom 9 at 3-hourly resolution is chunked as
248 time steps x 16384 cells, so the cheapest unit to fetch is a full
31-day block of the global field (192 chunks, ~2.7 GB per variable).
The store therefore samples whole blocks: a seeded choice of block
indices over the 2020-2049 run, excluding the block that holds the
benchmark day (2020-03-01), which is written separately as the eval
store. Each block is regridded from HEALPix (nest) to a regular
2880 x 1440 lon-lat grid by nearest neighbour, exactly as the
benchmark's data loader does (np.linspace(-180, 180, 2880) and
np.linspace(-90, 90, 1440), both endpoints included), and written as
float32 (time, lat, lon) chunked one day at a time.

Usage:
  python scripts/build_store_nextgems.py --out-dir /data/NextGEMS-samples --n-blocks 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import xarray as xr

URL = ("https://s3.waterpark.dkrz.de/misc/wcrp-hackathon/ICON/"
       "rechunked_ngc4008/ngc4008_PT3H_9.zarr")
ZOOM = 9
VARS = ["pr", "rlut"]
NUM_LON, NUM_LAT = 2880, 1440
TBLOCK = 248
BENCH_DAY = np.datetime64("2020-03-01")


def nn_index(nside, lons, lats):
    import healpy
    lons2, lats2 = np.meshgrid(lons, lats)
    return healpy.ang2pix(nside, lons2, lats2, nest=True, lonlat=True)


def write_block(ds, b, idx, lons, lats, path):
    t0, t1 = b * TBLOCK, min((b + 1) * TBLOCK, ds.sizes["time"])
    tmp = path + ".tmp"
    out = {}
    times = ds.time.values[t0:t1]
    for v in VARS:
        tv = time.time()
        raw = ds[v].isel(time=slice(t0, t1)).values        # (T, ncell)
        grid = raw[:, idx.ravel()].reshape(raw.shape[0], NUM_LAT, NUM_LON)
        del raw
        out[v] = xr.DataArray(grid, dims=("time", "lat", "lon"),
                              attrs=dict(ds[v].attrs))
        print(f"    {v}: {grid.shape} min={np.nanmin(grid):.3e} "
              f"max={np.nanmax(grid):.3e} zeros={(grid == 0).mean():.3f} "
              f"({time.time() - tv:.0f}s)", flush=True)
    dso = xr.Dataset(out, coords={"time": times, "lat": lats, "lon": lons})
    dso.attrs.update({
        "source": URL, "healpix_zoom": ZOOM, "regrid": "nearest neighbour, "
        "healpy.ang2pix(nside=2**zoom, nest=True) on the lon/lat mesh",
        "block": int(b), "block_time_steps": TBLOCK,
    })
    dso = dso.chunk({"time": 8, "lat": -1, "lon": -1})
    for v in dso.variables:
        dso[v].encoding = {}
    dso.to_zarr(tmp, mode="w", consolidated=True)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/NextGEMS-samples")
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = xr.open_zarr(URL, consolidated=True, chunks=None)
    n_blocks_total = int(np.ceil(ds.sizes["time"] / TBLOCK))
    tvals = ds.time.values
    bench_block = int(np.searchsorted(tvals, BENCH_DAY) // TBLOCK)
    rng = np.random.default_rng(args.seed)
    candidates = [b for b in range(n_blocks_total - 1) if b != bench_block]
    chosen = sorted(int(b) for b in rng.choice(candidates, args.n_blocks,
                                               replace=False))
    print(f"time steps {ds.sizes['time']} -> {n_blocks_total} blocks; "
          f"bench block {bench_block}; train blocks {chosen}", flush=True)

    lons = np.linspace(-180, 180, NUM_LON)
    lats = np.linspace(-90, 90, NUM_LAT)
    idx = nn_index(2 ** ZOOM, lons, lats)

    plan = ([] if args.skip_eval else [("eval", bench_block)]) + \
        [("train", b) for b in chosen]
    for split, b in plan:
        path = os.path.join(args.out_dir, f"{split}_block{b:03d}.zarr")
        if os.path.exists(path):
            print(f"{path}: present", flush=True)
            continue
        t0 = str(tvals[b * TBLOCK])[:13]
        print(f"{split} block {b} starting {t0}", flush=True)
        tb = time.time()
        write_block(ds, b, idx, lons, lats, path)
        print(f"  written in {time.time() - tb:.0f}s", flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"source": URL, "zoom": ZOOM, "vars": VARS,
                   "grid": [NUM_LAT, NUM_LON], "block_time_steps": TBLOCK,
                   "bench_block": bench_block, "train_blocks": chosen,
                   "seed": args.seed,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=1)


if __name__ == "__main__":
    main()
