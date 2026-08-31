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
    args = ap.parse_args()

    root = zarr.open(os.path.join(args.zarr_dir, "cubes.zarr"), mode="r")
    a = root[args.var]
    man = pd.read_parquet(os.path.join(args.zarr_dir, "manifest.parquet"))
    hours = list(range(0, a.shape[-3], args.stride_h))
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
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{args.out}: {len(rows)} slices of {args.var!r} over "
          f"{len(man)} cubes, stride {args.stride_h} h")


if __name__ == "__main__":
    main()
