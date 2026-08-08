"""Build a Tessera downstream-eval dataset at an existing AE eval
dataset's label points — same locations, same labels, same geo-block
split groups (the AE tile `fid` rides along), so the AE and Tessera
codecs are probed on directly comparable data.

Example:
  python scripts/eval_build_dataset_tessera.py \
      --ae-dataset /data/AE-samples/eval/lucas_lc \
      --out-dir /data/tessera-samples/eval/lucas_lc --max-tiles 600
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from georvq.eval.tessera_extract import extract_points_tessera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-dataset", required=True,
                    help="built AE eval dataset dir (targets.parquet + meta.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--max-tiles", type=int, default=600,
                    help="tessera tile download budget (densest-first); "
                         "0 = no cap")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-valid-frac", type=float, default=0.5)
    ap.add_argument("--store-index",
                    default="/data/tessera-samples/chips_512/index.parquet")
    args = ap.parse_args()

    ae_meta = json.load(open(os.path.join(args.ae_dataset, "meta.json")))
    t = pd.read_parquet(os.path.join(args.ae_dataset, "targets.parquet"))
    print(f"AE dataset {ae_meta['benchmark']}: {len(t)} targets", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, "_scratch")
    os.makedirs(cache, exist_ok=True)
    windows, targets, stats = extract_points_tessera(
        t, cache, crop_sz=args.crop, max_tiles=args.max_tiles or None,
        workers=args.workers, min_valid_frac=args.min_valid_frac,
        store_index=args.store_index)
    shutil.rmtree(cache, ignore_errors=True)

    np.save(os.path.join(args.out_dir, "windows.npy"), windows)
    targets.to_parquet(os.path.join(args.out_dir, "targets.parquet"))
    meta = {
        "benchmark": ae_meta["benchmark"], "task": ae_meta["task"],
        "metric": ae_meta["metric"], "crop": args.crop,
        "source": "tessera", "raw_bands": 130, "bands": 128,
        "dequant": "x = q * scale (fp16 scale in 2 trailing byte-bands); "
                   "L2-normed at probe time",
        "ae_dataset": args.ae_dataset,
        "n_targets": int(len(targets)), "n_crops": int(len(windows)),
        "n_tiles": int(targets.groupby(["tile_lon", "tile_lat"]).ngroups),
        "extraction_stats": stats,
        "builder_config": {k: v for k, v in vars(args).items()},
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"),
              indent=1)
    gb = windows.nbytes / 2**30
    print(f"\nwrote {args.out_dir}: {len(targets)} targets / "
          f"{len(windows)} crops / {meta['n_tiles']} tiles "
          f"({gb:.1f} GiB)", flush=True)


if __name__ == "__main__":
    main()
