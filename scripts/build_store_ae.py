"""Build an AlphaEarth chip store by equal-area random sampling of
1024x1024 blocks from the AEF cloud tiles.

Downloads the rate-optimal unit: COG blocks are 1024^2 band-interleaved, so
a block-aligned 1024^2 all-band window is the largest read with zero waste
(~30 MB compressed transfer -> 64 MiB raw int8 kept).

Sampling: lon ~ U(-180,180), lat ~ arcsin(U(-1,1)) (uniform on the sphere),
point -> containing tile via the GeoParquet index (random year among
matches), point -> containing 1024-block. Ocean auto-rejects (no tile /
NoData). Chips are flipped to north-up on read (tiffs are stored bottom-up)
and stored RAW int8 (de-quantise with the signed-square mapping at load).

Usage:
  python scripts/build_store_ae.py --n-chips 10000        # ~670 GB store
  python scripts/build_store_ae.py --n-chips 8 --out-dir /tmp/smoke
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_NUM_THREADS", "16")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

from georvq.data.build import ShardWriter

NODATA = -128
TILE = 8192
BLOCK = 1024
BLOCKS = TILE // BLOCK  # 8x8 blocks per tile


class Sampler:
    """Draw (tile_row, block) targets uniformly on the sphere."""

    def __init__(self, index_path, exclude_paths, seed, seen=None):
        cols = ["path", "year", "utm_zone", "crs",
                "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north",
                "utm_west", "utm_south", "utm_east", "utm_north"]
        df = pd.read_parquet(index_path, columns=cols)
        df = df[~df.path.isin(exclude_paths)].reset_index(drop=True)
        # drop antimeridian-wrapping tiles (west > east), a negligible sliver
        df = df[df.wgs84_west < df.wgs84_east].reset_index(drop=True)
        self.df = df
        self.w = df.wgs84_west.to_numpy()
        self.e = df.wgs84_east.to_numpy()
        self.s = df.wgs84_south.to_numpy()
        self.n = df.wgs84_north.to_numpy()
        self.rng = np.random.default_rng(seed)
        self.seen = seen or set()

    def draw(self):
        """One accepted (row, block_row, block_col, lon, lat) target or None."""
        lon = float(self.rng.uniform(-180, 180))
        lat = float(np.degrees(np.arcsin(self.rng.uniform(-1, 1))))
        hits = np.nonzero((self.w <= lon) & (lon <= self.e)
                          & (self.s <= lat) & (lat <= self.n))[0]
        if not len(hits):
            return None                       # ocean / no coverage
        row = self.df.iloc[int(self.rng.choice(hits))]
        xs, ys = warp_transform("EPSG:4326", row.crs, [lon], [lat])
        x, y = xs[0], ys[0]
        if not (row.utm_west <= x < row.utm_east
                and row.utm_south <= y < row.utm_north):
            return None                       # bbox hit but outside raster
        col_px = (x - row.utm_west) / 10.0
        row_px = (row.utm_north - y) / 10.0   # north-up row
        br, bc = int(row_px) // BLOCK, int(col_px) // BLOCK
        key = (row.path, br, bc)
        if key in self.seen:
            return None
        self.seen.add(key)
        return row, br, bc, lon, lat


def fetch_chip(target, max_nodata):
    """Read one north-up 1024^2 all-band block. Returns (array, meta) or None."""
    row, br, bc, lon, lat = target
    vsis3 = row.path.replace("s3://", "/vsis3/")
    try:
        with rasterio.open(vsis3) as ds:
            bottom_up = ds.transform.e > 0
            r_tiff = (TILE - (br + 1) * BLOCK) if bottom_up else br * BLOCK
            win = Window(bc * BLOCK, r_tiff, BLOCK, BLOCK)
            # cheap pre-screen on a decimated band-1 read (overviews):
            # reject only fully-empty blocks (keep coastal/partial chips)
            b1 = ds.read(1, window=win, out_shape=(32, 32))
            if (b1 == NODATA).mean() >= max_nodata:
                return None
            a = ds.read(window=win)
            if bottom_up:
                a = a[:, ::-1, :]
            nd = float((a[0] == NODATA).mean())
            if nd >= max_nodata:
                return None
            return np.ascontiguousarray(a), {
                "path": row.path, "year": int(row.year),
                "utm_zone": row.utm_zone, "crs": row.crs,
                "block_row": br, "block_col": bc,
                "lon": lon, "lat": lat, "nodata_frac": nd,
                "flipped_from_bottom_up": bool(bottom_up),
            }
    except Exception as e:  # noqa: BLE001 - transient HTTP errors: skip target
        print(f"  ! fetch failed ({type(e).__name__}): {row.path} "
              f"b({br},{bc}): {str(e)[:120]}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/data/AE-samples/aef_index.parquet")
    ap.add_argument("--out-dir", default="/data/AE-samples/chips_1024")
    ap.add_argument("--exclude-selection",
                    default="/data/AE-samples/tiles_full/selection.csv",
                    help="CSV with a 'path' column of tiles to exclude")
    ap.add_argument("--n-chips", type=int, default=10000)
    ap.add_argument("--shard-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-nodata", type=float, default=1.0,
                    help="reject chips with NoData fraction >= this; default "
                         "1.0 = reject only fully-empty blocks (keep coastal)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--append", action="store_true",
                    help="continue an existing store: resume shard/chip "
                         "numbering, dedupe against its index")
    args = ap.parse_args()

    exclude = set()
    if args.exclude_selection and os.path.exists(args.exclude_selection):
        exclude = set(pd.read_csv(args.exclude_selection).path)

    writer = ShardWriter(args.out_dir, args.shard_size, bands=64,
                         chip_px=BLOCK, dtype=np.int8, append=args.append)
    seen0 = set()
    prior = writer.prior_index
    if len(prior):
        seen0 = set(zip(prior.path, prior.block_row, prior.block_col))
    sampler = Sampler(args.index, exclude, args.seed, seen=seen0)
    print(f"index: {len(sampler.df)} tiles after exclusions; "
          f"target {args.n_chips} new chips -> {args.out_dir}", flush=True)

    target = args.n_chips + writer.n_start
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pending = set()
        while writer.n_done < target:
            while len(pending) < args.workers * 2 and writer.n_done < target:
                t = sampler.draw()
                if t is not None:
                    pending.add(ex.submit(fetch_chip, t, args.max_nodata))
            if not pending:
                continue
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                res = f.result()
                if res is not None and writer.n_done < target:
                    writer.add(*res, target=args.n_chips)

    manifest = writer.finalize(
        "ae", nodata=NODATA, grid_period=2,
        extra={
            "orientation": "north-up (flipped from bottom-up source on read)",
            "dequant": "x = (s/127.5)**2 * sign(s); unit-norm 64-d per pixel",
            "attribution": ("The AlphaEarth Foundations Satellite Embedding "
                            "dataset is produced by Google and Google "
                            "DeepMind."),
            "license": "CC-BY 4.0",
            "builder_config": vars(args) | {"out_dir": str(args.out_dir)},
        })
    el = time.time() - t0
    print(f"\ndone: {manifest['n_chips']} chips in {el/3600:.2f}h; "
          f"store -> {args.out_dir} "
          f"({manifest['n_chips'] * 64 * BLOCK * BLOCK / 2**30:.0f} GiB)")


if __name__ == "__main__":
    main()
