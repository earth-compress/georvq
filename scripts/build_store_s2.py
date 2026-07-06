"""Build a Sentinel-2 multi-group chip store from Major-TOM scene
directories (one GeoTIFF per band: B01.tif .. B12.tif).

Each chip is stored at native resolutions as three groups (10m/20m/60m),
uint16 reflectance. Chip size is given in 10m pixels and must be divisible
by 6 so every chip lands exactly on all three grids.

  python scripts/build_store_s2.py --scenes-root /data/Major-TOM/Core-S2L1C \
      --out-dir /data/s2-store --chip-px 1020 --max-scenes 200
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from georvq.data.build import ShardWriter
from georvq.sources.s2 import BANDS_10M, BANDS_20M, BANDS_60M

GROUP_BANDS = {"10m": BANDS_10M, "20m": BANDS_20M, "60m": BANDS_60M}
GROUP_RATIO = {"10m": 1, "20m": 2, "60m": 6}


def discover_scenes(root):
    """Scene dirs two levels below the root (grid-cell / scene layout)."""
    hits = sorted(glob.glob(os.path.join(root, "*", "*", "*", "B02.tif")))
    if not hits:
        hits = sorted(glob.glob(os.path.join(root, "*", "*", "B02.tif")))
    return [os.path.dirname(h) for h in hits]


def load_scene(scene_path):
    import rasterio
    arrays = {}
    for group, bands in GROUP_BANDS.items():
        stack = []
        for b in bands:
            with rasterio.open(os.path.join(scene_path, f"{b}.tif")) as ds:
                stack.append(ds.read(1))
        arrays[group] = np.stack(stack)
    return arrays


def scene_meta(scene_path):
    import rasterio
    with rasterio.open(os.path.join(scene_path, "B02.tif")) as ds:
        crs = str(ds.crs)
        lon, lat = ds.lnglat()
    return crs, lon, lat


def chip_scene(arrays, chip_px):
    """Aligned chip grid over the scene; drops the ragged border."""
    h10 = arrays["10m"].shape[1]
    n = h10 // chip_px
    for r in range(n):
        for c in range(n):
            chip = {}
            ok = True
            for g, ratio in GROUP_RATIO.items():
                p = chip_px // ratio
                a = arrays[g][:, r * p:(r + 1) * p, c * p:(c + 1) * p]
                if a.shape[1] != p or a.shape[2] != p:
                    ok = False
                    break
                chip[g] = np.ascontiguousarray(a)
            if ok:
                yield r, c, chip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chip-px", type=int, default=1020,
                    help="chip size in 10m pixels (divisible by 6)")
    ap.add_argument("--shard-size", type=int, default=64)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    assert args.chip_px % 6 == 0, "--chip-px must be divisible by 6"

    scenes = discover_scenes(args.scenes_root)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    print(f"{len(scenes)} scenes under {args.scenes_root}", flush=True)

    groups = {g: (len(b), args.chip_px // GROUP_RATIO[g])
              for g, b in GROUP_BANDS.items()}
    writer = ShardWriter(args.out_dir, args.shard_size, dtype=np.uint16,
                         groups=groups, append=args.append)
    done_scenes = set()
    prior = writer.prior_index
    if len(prior):
        done_scenes = set(prior.scene_id)

    for sp in scenes:
        scene_id = "/".join(sp.rstrip("/").split("/")[-2:])
        if scene_id in done_scenes:
            continue
        arrays = load_scene(sp)
        crs, lon, lat = scene_meta(sp)
        for r, c, chip in chip_scene(arrays, args.chip_px):
            writer.add(chip, {"scene_id": scene_id, "patch_row": r,
                              "patch_col": c, "crs": crs,
                              "lon": lon, "lat": lat, "year": 0},
                       log_every=200)

    manifest = writer.finalize(
        "s2", nodata=0, grid_period=6,
        extra={"bands_by_group": GROUP_BANDS,
               "units": "uint16 reflectance x10000",
               "builder_config": vars(args)})
    print(f"done: {manifest['n_chips']} chips -> {args.out_dir}")


if __name__ == "__main__":
    main()
