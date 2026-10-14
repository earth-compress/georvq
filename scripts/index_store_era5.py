"""Write an ERA5 chip-store directory over an existing cubes.zarr.

One store = one variable of one split at one hour stride. The directory
holds only manifest.json and index.parquet; slices are read live by
georvq.data.era5.Era5Store. The default 6 h stride keeps slices whose
anomaly autocorrelation has fallen to ~0.6-0.75, so consecutive kept
hours are related but not near-duplicates.

    python scripts/index_store_era5.py \
        --zarr-dir /data/ERA5-samples/train --var t \
        --out /data/ERA5-samples/train/store_t
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr-dir", required=True,
                    help="split dir holding cubes.zarr + manifest.parquet")
    ap.add_argument("--var", required=True)
    ap.add_argument("--stride-h", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--single-level", action="store_true",
                    help="one row per (cube, hour, LEVEL): slices are "
                         "single standardized planes (slot = hour * "
                         "n_levels + level), for bands=1 models")
    args = ap.parse_args()

    root = zarr.open(os.path.join(args.zarr_dir, "cubes.zarr"), mode="r")
    a = root[args.var]
    man = pd.read_parquet(os.path.join(args.zarr_dir, "manifest.parquet"))
    hours = list(range(0, a.shape[-3], args.stride_h))
    if args.single_level:
        assert a.ndim == 5, "--single-level needs a pressure-level var"
        nl = a.shape[1]
        rows = pd.DataFrame(
            [{"shard": int(r.cube_id), "slot": h * nl + lv,
              "window": int(r.window), "lat": float(r.lat_center),
              "lon": float(r.lon_center)}
             for r in man.itertuples() for h in hours
             for lv in range(nl)])
    else:
        rows = pd.DataFrame(
            [{"shard": int(r.cube_id), "slot": h, "window": int(r.window),
              "lat": float(r.lat_center), "lon": float(r.lon_center)}
             for r in man.itertuples() for h in hours])

    os.makedirs(args.out, exist_ok=True)
    rows.to_parquet(os.path.join(args.out, "index.parquet"))
    manifest = {
        "kind": "era5",
        "var": args.var,
        "zarr": os.path.relpath(
            os.path.join(args.zarr_dir, "cubes.zarr"), args.out),
        "chip_px": int(a.shape[-1]),
        "bands": int(a.shape[1]) if a.ndim == 5 else 1,
        "stride_h": args.stride_h,
        "n_cubes": int(man.cube_id.max()) + 1,
    }
    if args.single_level:
        # the moments bake into the store's output (model space): record
        # them so the mapping is part of the store's identity
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        from georvq.sources.era5 import _STATS_PATH
        with open(_STATS_PATH) as f:
            stats = json.load(f)[args.var]
        manifest.update(single_level=True, bands=1,
                        level_mean=stats["mean"], level_std=stats["std"])
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{args.out}: {len(rows)} slices of {args.var!r} over "
          f"{len(man)} cubes, stride {args.stride_h} h")


if __name__ == "__main__":
    main()
