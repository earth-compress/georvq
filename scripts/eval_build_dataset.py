"""Build a contamination-free downstream-eval dataset for one benchmark.

Samples held-out AEF tiles (never seen in training), reads one valid block each
from S3, carves context crops, samples target pixels, and joins the benchmark's
label. Writes a self-contained dataset that eval_rate_sweep.py consumes:

  <out>/windows.npy     (M, 64, crop, crop) int8   de-duplicated context crops
  <out>/targets.parquet  one row per labelled pixel (window_idx, r, c, lon/lat,
                         tile fid/path/year, label)
  <out>/meta.json       benchmark, task, metric, crop size, counts, provenance

Example (a first ASTER GED sample):
  python scripts/eval_build_dataset.py --benchmark aster_ged \
      --n-targets 20000 --crop 128 --stride 10 \
      --out-dir /data/AE-samples/eval/aster_ged
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from georvq.eval import extract
from georvq.eval.benchmarks import BENCHMARKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(BENCHMARKS))
    ap.add_argument("--aef-index", default="/data/AE-samples/aef_index.parquet")
    ap.add_argument("--chips-index",
                    default="/data/AE-samples/chips_1024/index.parquet")
    ap.add_argument("--tiles-selection",
                    default="/data/AE-samples/tiles_full/selection.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=None,
                    help="label tile cache (default <out>/label_cache)")
    ap.add_argument("--n-targets", type=int, default=20000)
    ap.add_argument("--crop", type=int, default=128,
                    help="context-crop size; must be >= the largest codec "
                         "window you intend to evaluate (the 128px codec).")
    ap.add_argument("--stride", type=int, default=10,
                    help="target-pixel spacing in px (10 ~= 100 m, ASTER res)")
    ap.add_argument("--max-per-window", type=int, default=20)
    ap.add_argument("--max-crops-per-tile", type=int, default=2,
                    help="crops kept per tile; small => targets spread across "
                         "many tiles (geo-diversity + low storage)")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="default: ~3x the tiles needed for --n-targets")
    ap.add_argument("--max-blocks-per-tile", type=int, default=2,
                    help="points mode: read this many densest 1024-blocks per "
                         "tile (keeps all their points). 1-2 => max geo-"
                         "diversity + amortized reads for dense surveys (LUCAS)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.out_dir, "label_cache")
    bench = BENCHMARKS[args.benchmark]

    exclude = extract.training_tile_paths(args.chips_index, args.tiles_selection)
    print(f"excluded {len(exclude)} training tiles; benchmark mode="
          f"{bench.mode}; coverage={bench.coverage_bbox}", flush=True)

    if bench.mode == "points":
        points = bench.load_points(cache_dir)
        windows, targets = extract.extract_points(
            points, args.aef_index, exclude, bench.coverage_bbox,
            bench.survey_year, crop_sz=args.crop, n_targets=args.n_targets,
            workers=args.workers, seed=args.seed,
            max_blocks_per_tile=args.max_blocks_per_tile)
    else:
        pool = extract.held_out_tile_pool(
            args.aef_index, exclude, coverage_bbox=bench.coverage_bbox,
            years=bench.years)
        print(f"held-out tile pool: {len(pool)} full tiles", flush=True)
        if len(pool) == 0:
            raise SystemExit("empty tile pool -- check coverage / exclusions")
        windows, targets = extract.build_windows(
            pool, args.n_targets, crop_sz=args.crop, stride=args.stride,
            max_per_window=args.max_per_window, workers=args.workers,
            seed=args.seed, max_tiles=args.max_tiles,
            max_crops_per_tile=args.max_crops_per_tile)
        targets = bench.attach_labels(targets, cache_dir)
    print(f"collected {len(windows)} crops, {len(targets)} labelled targets",
          flush=True)
    if len(targets) == 0:
        raise SystemExit("no labelled targets -- check label source/coverage")

    # prune crops no surviving target references; remap window_idx
    used = np.sort(targets.window_idx.unique())
    remap = {int(o): i for i, o in enumerate(used)}
    windows = windows[used]
    targets = targets.copy()
    targets["window_idx"] = targets.window_idx.map(remap).astype(int)

    np.save(os.path.join(args.out_dir, "windows.npy"), windows)
    targets.to_parquet(os.path.join(args.out_dir, "targets.parquet"))
    meta = {
        "benchmark": bench.name, "task": bench.task, "metric": bench.metric,
        "crop": args.crop, "stride": args.stride, "seed": args.seed,
        "n_targets": int(len(targets)), "n_crops": int(len(windows)),
        "n_tiles": int(targets.fid.nunique()),
        "coverage_bbox": bench.coverage_bbox,
        "dequant": "(s/127.5)**2*sign(s); L2-normed at probe time",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nwrote dataset -> {args.out_dir}", flush=True)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
