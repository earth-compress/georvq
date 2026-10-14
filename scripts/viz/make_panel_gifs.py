"""Per-place 1024px 'AlphaEarth Variable Compression' panel GIFs.

One large 1024px slice per GIF, laid out as: column ORIGINAL (1st- & 2nd-order
PCA stacked), column RECONSTRUCTION (same two), column COSINE SIMILARITY (one
all-channel quality map). Reconstruction tiles the slice into each codec's
native window; the merged rate sweep animates over the styled compression bar.
Slices are read from AEF S3 tiles at 10 curated places (pass a place name to
render just one).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from georvq.config import TierSpec, parse_tier
from georvq.eval import codec_ab
from georvq.eval.extract import dequant_np
from georvq.sources.ae import NODATA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panels  # noqa: E402

DEFAULT_CKPTS = ["checkpoints/ae_pyr128_deep/model_best.pt",
                 "checkpoints/ae_pyr64_divfull/model_best.pt"]
DEFAULT_TIERS = ["1:256:2048:8", "4:192:8192:8", "16:64:16384:8"]

# curated distinctive 1024px (10.24 km) slices: (name, lon, lat)
PLACES = [
    ("london", -0.128, 51.508),            # urban + Thames
    ("richat", -11.401, 21.124),           # Eye of the Sahara
    ("sundarbans", 89.18, 21.95),          # Ganges delta / mangroves
    ("grand_canyon", -112.11, 36.10),      # plateau / cliffs
    ("everest", 86.92, 27.99),             # Himalaya
    ("khayelitsha", 18.68, -34.04),        # informal settlement, Cape Town
    ("namib_dunes", 15.30, -24.73),        # Sossusvlei dune field
    ("pivot_irrigation", 38.45, 30.05),    # Saudi center-pivot circles
    ("okavango", 22.83, -19.28),           # inland delta
    ("amazon", -73.40, -4.50),             # rainforest meanders
]


def download_block(full, lon, lat, prefer_year=2024):
    """Read the 1024px block containing (lon, lat) from the most recent AEF tile
    (<= prefer_year). Returns (chip (64,1024,1024) int8, year) or (None, None)."""
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window
    h = full[(full.wgs84_west <= lon) & (lon <= full.wgs84_east)
             & (full.wgs84_south <= lat) & (lat <= full.wgs84_north)]
    h = h[h.year <= prefer_year]
    if not len(h):
        return None, None
    row = h.loc[h.year.idxmax()]
    with rasterio.open(row.path.replace("s3://", "/vsis3/")) as ds:
        bottom_up = ds.transform.e > 0
        xs, ys = warp_transform("EPSG:4326", row.crs, [lon], [lat])
        col_px = int((xs[0] - row.utm_west) / 10)
        row_px = int((row.utm_north - ys[0]) / 10)
        br, bc = row_px // 1024, col_px // 1024
        r_tiff = (8192 - (br + 1)*1024) if bottom_up else br*1024
        a = ds.read(window=Window(bc*1024, r_tiff, 1024, 1024))
        if bottom_up:
            a = a[:, ::-1, :]
    return np.ascontiguousarray(a), int(row.year)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="?", default=None,
                    help="render only this place (default: all)")
    ap.add_argument("--ckpt", nargs="+", default=DEFAULT_CKPTS,
                    help="checkpoints, coarsest-window first (its range leads "
                         "the sweep; later models add the high-bitrate tail)")
    ap.add_argument("--aef-index", default="/data/AE-samples/aef_index.parquet")
    ap.add_argument("--windows",
                    default="/data/AE-samples/eval/aster_ged/windows.npy",
                    help="eval windows array the global PCA is fit on")
    ap.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS,
                    help="tier specs grid:dim:codes:depth (shared by ckpts)")
    ap.add_argument("--min-depths", default="6,1,0")
    ap.add_argument("--n-rates", type=int, default=14)
    ap.add_argument("--prefer-year", type=int, default=2024)
    ap.add_argument("--out-dir", default="plots/ae_gifs/panels")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    specs = [TierSpec(**parse_tier(s)) for s in args.tiers]
    mins = [int(x) for x in args.min_depths.split(",")]
    tuples = codec_ab.select_front_tuples(specs, mins, args.n_rates)
    models = [(os.path.basename(os.path.dirname(p)) or "ckpt", p)
              for p in args.ckpt]
    gpca, glohi = panels.global_pca(args.windows)
    idx = pd.read_parquet(args.aef_index,
                          columns=["path", "year", "crs", "utm_west",
                                   "utm_south", "utm_east", "utm_north",
                                   "wgs84_west", "wgs84_south", "wgs84_east",
                                   "wgs84_north"])
    FULL = 8192 * 10.0
    full = idx[((idx.utm_east - idx.utm_west) == FULL)
               & ((idx.utm_north - idx.utm_south) == FULL)]
    os.makedirs(args.out_dir, exist_ok=True)
    for name, lon, lat in PLACES:
        if args.only and args.only != name:
            continue
        chip, year = download_block(full, lon, lat, args.prefer_year)
        if chip is None:
            print(f"{name}: no tile found", flush=True)
            continue
        nod = (chip[0] == NODATA).mean()
        ds_ = dequant_np(chip).reshape(chip.shape[0], 32, 32, 32,
                                       32).mean(axis=(2, 4))
        print(f"{name}: year {year} nodata {nod:.3f} struct "
              f"{ds_.std(axis=(1, 2)).mean():.4f}", flush=True)
        if nod > 0.25:
            print(f"  {name}: high nodata, skipping", flush=True)
            continue
        orig_rgb, to_rgb = panels.panel_mapper(gpca, glohi, chip)
        build_out = os.path.join(args.out_dir, f"ae_vc_{name}.gif")
        panels.build_panel(chip, orig_rgb, to_rgb, specs, tuples, models,
                           build_out, args.device)


if __name__ == "__main__":
    main()
