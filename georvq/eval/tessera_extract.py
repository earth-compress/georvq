"""Tessera extraction at existing AE-eval label points.

Reuses an already-built AlphaEarth eval dataset's `targets.parquet` (the
label points: lon/lat + label columns + the AE tile `fid` used for the
geo-block split) and extracts, for each point, a `crop_sz` context crop
from the Tessera embedding tile (0.1-degree, UTM 10 m) containing it —
so the downstream probes run over the SAME locations and labels as the
AE study.

Windows are stored ChipStore-style raw: (M, 130, crop, crop) int8 = 128
max-abs-quantized bands + the per-pixel float16 dequant scale as 2
trailing byte-bands (exactly what `TesseraSource.prep` expects). The
original (uncompressed) embedding at a target pixel is `q * scale`,
L2-normed at probe time — matching the AE protocol, where probes also
run on L2-normed vectors.

Contamination guard: any 0.1-degree tile that contributed chips to the
training store (any year — embeddings of one tile correlate across
years) is excluded, and its points dropped.

A tile (~110 MB) is the download quantum, so tiles are processed densest
-first up to `max_tiles`; per tile we pick the year the AE targets there
used (modal), falling back to the nearest year the registry has.
"""
from __future__ import annotations

import os
import shutil
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

BANDS = 128
RAW_BANDS = 130                       # + 2 scale bytes (fp16 little-endian)
SCALE_MAX = 1.0                       # match TesseraSource.SCALE_MAX


def tile_key(lon: float, lat: float) -> tuple[int, int]:
    """Containing-tile center in integer 1/100 degrees (0.1-degree tiles
    centered at 0.05 offsets) — same convention as the store builder."""
    return (int(np.floor(lon * 10)) * 10 + 5,
            int(np.floor(lat * 10)) * 10 + 5)


def training_tile_keys(store_index="/data/tessera-samples/chips_512/index.parquet"):
    """Tile keys (any year) that contributed training chips."""
    if not os.path.exists(store_index):
        return set()
    df = pd.read_parquet(store_index, columns=["tile_lon", "tile_lat"])
    return set(zip((df.tile_lon * 100).round().astype(int),
                   (df.tile_lat * 100).round().astype(int)))


def registry_years(gt):
    """tile_key -> sorted available years, from geotessera's registry."""
    years = defaultdict(set)
    for year, lon, lat in gt.registry.get_available_embeddings():
        years[(int(round(lon * 100)), int(round(lat * 100)))].add(int(year))
    return {k: sorted(v) for k, v in years.items()}


def plan_tiles(targets: pd.DataFrame, avail: dict, exclude: set,
               max_tiles: int | None):
    """Group AE target rows by containing Tessera tile; drop excluded /
    uncovered tiles; pick each tile's year (modal AE target year, else the
    nearest year the registry has). Returns a list of
    (tile_lon, tile_lat, year, row_indices) densest-first, and counters."""
    by_tile = defaultdict(list)
    lons, lats = targets.lon.to_numpy(), targets.lat.to_numpy()
    for i in range(len(targets)):
        by_tile[tile_key(lons[i], lats[i])].append(i)
    n_train, n_nocov = 0, 0
    plans = []
    yr_tgt = targets.year.to_numpy() if "year" in targets else \
        np.full(len(targets), 2024)
    for key, rows in by_tile.items():
        if key in exclude:
            n_train += len(rows)
            continue
        ys = avail.get(key)
        if not ys:
            n_nocov += len(rows)
            continue
        want = Counter(int(yr_tgt[i]) for i in rows).most_common(1)[0][0]
        year = min(ys, key=lambda y: abs(y - want))
        plans.append((key[0] / 100.0, key[1] / 100.0, year, np.array(rows)))
    plans.sort(key=lambda p: -len(p[3]))
    dropped_cap = sum(len(p[3]) for p in plans[max_tiles:]) if max_tiles else 0
    if max_tiles:
        plans = plans[:max_tiles]
    return plans, {"train_tile": n_train, "no_coverage": n_nocov,
                   "tile_cap": dropped_cap}


def pack_scale_bytes(scale_f16: np.ndarray) -> np.ndarray:
    """(H, W) float16 -> (2, H, W) int8 byte-planes (little-endian),
    matching the ChipStore aux_bytes layout TesseraSource.prep unpacks."""
    b = scale_f16.view(np.uint8).reshape(*scale_f16.shape, 2)
    return b.transpose(2, 0, 1).astype(np.int8, copy=False)


def fetch_tile_crops(gt, tile_lon, tile_lat, year, sub: pd.DataFrame,
                     label_cols, crop_sz, min_valid_frac=0.5):
    """Download one Tessera tile (q + scales + landmask georeferencing),
    slice a grid-aligned `crop_sz` context crop around every point, keep
    points whose own pixel is valid. Returns (crops, target rows)."""
    from rasterio.warp import transform as warp_transform
    try:
        qp = gt.registry.fetch(year=year, lon=tile_lon, lat=tile_lat)
        sp = gt.registry.fetch(year=year, lon=tile_lon, lat=tile_lat,
                               is_scales=True)
        q = np.load(qp)                               # (H, W, 128) int8
        s = np.load(sp)                               # (H, W) float32
        shutil.rmtree(os.path.dirname(qp), ignore_errors=True)
        crs, transform = gt._get_utm_projection_from_landmask(
            tile_lon, tile_lat, False)
    except Exception as e:  # noqa: BLE001 - transient HTTP errors: skip tile
        print(f"  ! fetch failed ({type(e).__name__}): {year} "
              f"({tile_lon},{tile_lat}): {str(e)[:120]}", flush=True)
        return [], []
    h, w = q.shape[:2]
    if h < crop_sz or w < crop_sz:
        return [], []                                 # high-latitude sliver
    # finite fp16 only (rare corrupt source scales; SCALE_MAX masks them)
    s16 = np.nan_to_num(s, nan=0.0, posinf=65504,
                        neginf=0.0).astype(np.float16)
    xs, ys = warp_transform("EPSG:4326", crs, sub.lon.tolist(),
                            sub.lat.tolist())
    inv = ~transform
    crops, rows, crop_id = [], [], {}
    lab = {c: sub[c].to_numpy() for c in label_cols}
    fid = sub.fid.to_numpy()
    ae_idx = sub.index.to_numpy()
    for j in range(len(sub)):
        fc, fr = inv * (xs[j], ys[j])
        r, c = int(np.floor(fr)), int(np.floor(fc))
        if not (0 <= r < h and 0 <= c < w):
            continue
        if not (0 < float(s16[r, c]) < SCALE_MAX) or not q[r, c].any():
            continue                                  # nodata/corrupt pixel
        cr, cc = r // crop_sz, c // crop_sz
        if (cr + 1) * crop_sz > h or (cc + 1) * crop_sz > w:
            continue                                  # edge sliver crop
        if (cr, cc) not in crop_id:
            qc = q[cr * crop_sz:(cr + 1) * crop_sz,
                   cc * crop_sz:(cc + 1) * crop_sz]
            sc = s16[cr * crop_sz:(cr + 1) * crop_sz,
                     cc * crop_sz:(cc + 1) * crop_sz]
            valid = qc.any(axis=-1) & (sc > 0) & (sc < SCALE_MAX)
            if valid.mean() < min_valid_frac:
                continue                              # mostly-nodata crop
            arr = np.empty((RAW_BANDS, crop_sz, crop_sz), np.int8)
            arr[:BANDS] = qc.transpose(2, 0, 1)
            arr[BANDS:] = pack_scale_bytes(sc)
            crop_id[(cr, cc)] = len(crops)
            crops.append(arr)
        rows.append({"window_idx": crop_id[(cr, cc)],
                     "r": r - cr * crop_sz, "c": c - cc * crop_sz,
                     "lon": float(sub.lon.iloc[j]),
                     "lat": float(sub.lat.iloc[j]),
                     **{k: v[j] for k, v in lab.items()},
                     "fid": int(fid[j]), "ae_idx": int(ae_idx[j]),
                     "tile_lon": tile_lon, "tile_lat": tile_lat,
                     "year": int(year)})
    return crops, rows


def extract_points_tessera(targets: pd.DataFrame, cache_dir, crop_sz=128,
                           max_tiles=600, workers=12, min_valid_frac=0.5,
                           store_index="/data/tessera-samples/chips_512/index.parquet"):
    """Extract Tessera context crops at every AE target point. Returns
    (windows (M,130,crop,crop) int8, targets DataFrame, stats dict)."""
    from geotessera import GeoTessera

    class GTLocal(threading.local):
        def get(self):
            if not hasattr(self, "gt"):
                self.gt = GeoTessera(embeddings_dir=cache_dir)
            return self.gt

    gt_local = GTLocal()
    avail = registry_years(gt_local.get())
    exclude = training_tile_keys(store_index)
    label_cols = [c for c in targets.columns
                  if c == "label" or c.startswith("label_")]
    plans, dropped = plan_tiles(targets, avail, exclude, max_tiles)
    n_planned = sum(len(p[3]) for p in plans)
    print(f"  {len(plans)} tessera tiles planned, {n_planned}/{len(targets)} "
          f"targets reachable (dropped: {dropped})", flush=True)

    crops, rows, base = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_tile_crops, gt_local.get(), lo, la, yr,
                          targets.iloc[idx], label_cols, crop_sz,
                          min_valid_frac): (lo, la)
                for lo, la, yr, idx in plans}
        done = 0
        for fut in as_completed(futs):
            c, r = fut.result()
            done += 1
            for row in r:
                row["window_idx"] += base
            base += len(c)
            crops.extend(c)
            rows.extend(r)
            if done % 25 == 0 or done == len(futs):
                print(f"  tiles {done}/{len(futs)} | crops {base} | "
                      f"targets {len(rows)}", flush=True)
    if not crops:
        raise SystemExit("no valid tessera crops -- check coverage/registry")
    windows = np.stack(crops)
    tdf = pd.DataFrame(rows)
    stats = {"planned_tiles": len(plans), "planned_targets": n_planned,
             "kept_targets": len(tdf), "kept_crops": len(windows),
             "dropped": dropped}
    return windows, tdf, stats
