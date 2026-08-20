"""Build a Tessera chip store by equal-area random sampling of 512x512
crops from the 0.1-degree Tessera embedding tiles (github.com/ucam-eo/
geotessera).

Tessera tiles are UTM 10 m rasters of ~1110 rows x ~1110*cos(lat) cols x
128 bands, shipped as per-pixel max-abs-quantized int8 plus a per-pixel
float32 scale (dequant: x = q * scale). A whole tile (~110 MB) is the
download quantum, so each fetched tile yields up to --crops-per-tile
non-overlapping 512^2 crops at a random grid phase. Chips are stored RAW
int8 (the unit direction of each vector is scale-free; normalize at load,
see georvq/sources/tessera.py); the scale planes go to float16
scales_XXXX.npy sidecars with the same shard/slot layout so amplitude
stays available without entering chip-store v1.

Sampling: lon ~ U(-180,180), lat ~ arcsin(U(-1,1)) (uniform on the
sphere), point -> containing 0.1-degree tile, random year among the years
the registry has for that tile. Ocean auto-rejects (no tile). Nodata is
the all-bands-zero vector (matches scale == 0); crops that are fully
nodata are rejected, coastal partials kept.

Usage:
  python scripts/build_store_tessera.py --n-chips 16384   # ~512 GiB store
  python scripts/build_store_tessera.py --n-chips 8 --out-dir /tmp/smoke
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

from georvq.data.build import ShardWriter

PX = 512


def utm_epsg(lon: float, lat: float) -> str:
    """Standard UTM zone for a tile center (provenance only; Tessera derives
    the exact CRS from its landmask tiles, which may differ in the Norway/
    Svalbard exception zones)."""
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def tile_key(lon: float, lat: float) -> tuple[int, int]:
    """Containing-tile center in integer 1/100 degrees (tiles are 0.1-degree
    squares centered at 0.05 offsets)."""
    return (int(np.floor(lon * 10)) * 10 + 5,
            int(np.floor(lat * 10)) * 10 + 5)


class Sampler:
    """Draw (year, tile_lon, tile_lat) targets uniformly on the sphere."""

    def __init__(self, available, seed, seen=None):
        """available: list of (year, lon, lat) tile-center tuples from
        geotessera's registry."""
        self.years = {}      # tile_key -> [years]
        for year, lon, lat in available:
            key = (int(round(lon * 100)), int(round(lat * 100)))
            self.years.setdefault(key, []).append(year)
        self.rng = np.random.default_rng(seed)
        self.seen = seen or set()

    def draw(self):
        """One accepted (year, lon, lat) tile target or None."""
        lon = float(self.rng.uniform(-180, 180))
        lat = float(np.degrees(np.arcsin(self.rng.uniform(-1, 1))))
        key = tile_key(lon, lat)
        years = self.years.get(key)
        if not years:
            return None                       # ocean / no coverage
        year = int(self.rng.choice(years))
        if (year, *key) in self.seen:
            return None
        self.seen.add((year, *key))
        return year, key[0] / 100.0, key[1] / 100.0


class ScaleSidecar:
    """float16 scales_XXXX.npy planes, same (shard, slot) layout as the
    chip shards; skipped entirely by ChipStore's shard_????.npy glob."""

    def __init__(self, out_dir, shard_size, px):
        self.out_dir, self.shard_size, self.px = out_dir, shard_size, px
        self.sh, self.mm = None, None

    def write(self, chip_id, plane):
        sh, slot = divmod(chip_id, self.shard_size)
        if sh != self.sh:
            path = os.path.join(self.out_dir, f"scales_{sh:04d}.npy")
            if os.path.exists(path):
                self.mm = np.lib.format.open_memmap(path, mode="r+")
            else:
                self.mm = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float16,
                    shape=(self.shard_size, self.px, self.px))
            self.sh = sh
        self.mm[slot] = plane


def fetch_tile(gt_local, target, crops_per_tile, max_nodata, rng):
    """Fetch one tile, return a list of (chip_array, scales, meta) crops."""
    year, lon, lat = target
    gt = gt_local.get()
    try:
        qp = gt.registry.fetch(year=year, lon=lon, lat=lat)
        sp = gt.registry.fetch(year=year, lon=lon, lat=lat, is_scales=True)
        q = np.load(qp)                       # (H, W, 128) int8
        s = np.load(sp)                       # (H, W) float32
        shutil.rmtree(os.path.dirname(qp), ignore_errors=True)
    except Exception as e:  # noqa: BLE001 - transient HTTP errors: skip
        print(f"  ! fetch failed ({type(e).__name__}): {year} "
              f"({lon},{lat}): {str(e)[:120]}", flush=True)
        return []
    h, w = q.shape[:2]
    nr, nc = h // PX, w // PX
    if nr == 0 or nc == 0:
        return []                             # high-latitude sliver tile
    r0 = int(rng.integers(0, h - nr * PX + 1))
    c0 = int(rng.integers(0, w - nc * PX + 1))
    cells = [(r, c) for r in range(nr) for c in range(nc)]
    rng.shuffle(cells)
    out = []
    for r, c in cells[:crops_per_tile]:
        ro, co = r0 + r * PX, c0 + c * PX
        crop = q[ro:ro + PX, co:co + PX]
        nd = float((crop == 0).all(axis=-1).mean())
        if nd >= max_nodata:
            continue
        arr = np.ascontiguousarray(crop.transpose(2, 0, 1))   # (128, px, px)
        # crop-center lat/lon: linear across the tile's 0.1-degree footprint
        out.append((arr, s[ro:ro + PX, co:co + PX], {
            "lat": lat + 0.05 - (ro + PX / 2) / h * 0.1,
            "lon": lon - 0.05 + (co + PX / 2) / w * 0.1,
            "crs": utm_epsg(lon, lat), "year": year,
            "tile_lon": lon, "tile_lat": lat,
            "row_off": ro, "col_off": co, "nodata_frac": nd,
        }))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/tessera-samples/chips_512")
    ap.add_argument("--n-chips", type=int, default=16384)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--crops-per-tile", type=int, default=2,
                    help="non-overlapping 512^2 crops kept per fetched tile "
                         "(a tile is the download quantum; ~2 fit)")
    ap.add_argument("--max-nodata", type=float, default=1.0,
                    help="reject crops with nodata fraction >= this; default "
                         "1.0 = reject only fully-empty crops (keep coastal)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--append", action="store_true",
                    help="continue an existing store: resume shard/chip "
                         "numbering, dedupe against its index")
    args = ap.parse_args()

    import threading

    from geotessera import GeoTessera

    scratch = os.path.join(args.out_dir, "_scratch")
    os.makedirs(scratch, exist_ok=True)

    class GTLocal(threading.local):
        def get(self):
            if not hasattr(self, "gt"):
                self.gt = GeoTessera(embeddings_dir=scratch)
            return self.gt

    gt_local = GTLocal()

    writer = ShardWriter(args.out_dir, args.shard_size, bands=128,
                         chip_px=PX, dtype=np.int8, append=args.append)
    scales = ScaleSidecar(args.out_dir, args.shard_size, PX)
    seen0 = set()
    prior = writer.prior_index
    if len(prior):
        seen0 = set(zip(prior.year,
                        (prior.tile_lon * 100).round().astype(int),
                        (prior.tile_lat * 100).round().astype(int)))
    available = gt_local.get().registry.get_available_embeddings()
    sampler = Sampler(available, args.seed, seen=seen0)
    print(f"registry: {len(available)} (year,tile) entries, "
          f"{len(sampler.years)} distinct tiles; "
          f"target {args.n_chips} new chips -> {args.out_dir}", flush=True)

    target = args.n_chips + writer.n_start
    crop_rng = np.random.default_rng(args.seed + 1)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pending = set()
        while writer.n_done < target:
            while len(pending) < args.workers * 2 and writer.n_done < target:
                t = sampler.draw()
                if t is not None:
                    rng = np.random.default_rng(crop_rng.integers(2**63))
                    pending.add(ex.submit(fetch_tile, gt_local, t,
                                          args.crops_per_tile,
                                          args.max_nodata, rng))
            if not pending:
                continue
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                for arr, sc, meta in f.result():
                    if writer.n_done >= target:
                        break
                    # finite fp16 only: rare corrupt source scales overflow
                    # to inf on cast (TesseraSource masks them via SCALE_MAX)
                    scales.write(writer.n_done, np.nan_to_num(
                        sc, nan=0.0, posinf=65504, neginf=0.0
                    ).astype(np.float16))
                    writer.add(arr, meta, target=args.n_chips)

    manifest = writer.finalize(
        "tessera", nodata=None, grid_period=1,
        extra={
            "nodata_convention": "all-bands-zero vector (scale == 0)",
            "dequant": ("x = q * scale (per-pixel float scale in "
                        "scales_XXXX.npy float16 sidecars, same shard/slot "
                        "layout); unit direction is scale-free: "
                        "normalize(q) as in georvq/sources/tessera.py"),
            "attribution": ("Tessera embeddings by the University of "
                            "Cambridge EO group (github.com/ucam-eo/"
                            "geotessera)."),
            "builder_config": vars(args) | {"out_dir": str(args.out_dir)},
        })
    shutil.rmtree(scratch, ignore_errors=True)
    el = time.time() - t0
    print(f"\ndone: {manifest['n_chips']} chips in {el/3600:.2f}h; "
          f"store -> {args.out_dir} "
          f"({manifest['n_chips'] * 128 * PX * PX / 2**30:.0f} GiB)")


if __name__ == "__main__":
    main()
