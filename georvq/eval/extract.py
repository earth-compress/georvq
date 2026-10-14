"""AlphaEarth (AEF) held-out extraction: contamination-aware eval windows.

Eval pixels are drawn ONLY from AEF source tiles that contributed zero training
chips (exclude the `path`s in chips_1024/index.parquet + tiles_full/selection.csv),
so the compression model has never seen them. For each sampled full 8192^2 tile
we read one valid north-up 1024^2 block from S3, carve it into `crop x crop`
context windows, and sample target pixels on a stride grid inside fully-valid
crops.

`crop` is the *largest* codec window we want to evaluate (default 128, to cover
the 128px codec); the run-side sub-tiles each stored crop into the checkpoint's
native window (64, 32, ...) so one eval dataset faithfully serves every codec.
The output dataset keeps, per target pixel: the index of its crop (so the codec
can encode->decode real spatial context), the (row, col) offset of the pixel
within that crop, and its lon/lat (to join a downstream label). Crops are
de-duplicated -- many targets share one crop.
"""
from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

NODATA = -128
TILE = 8192
BLOCK = 1024
BLOCKS = TILE // BLOCK
PX_M = 10.0
FULL_M = TILE * PX_M

# Unsigned public-bucket reads of the AEF COGs. Applied by the S3 readers
# rather than at import, and only where the caller has not set a value.
_S3_ENV = {"AWS_NO_SIGN_REQUEST": "YES",
           "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
           "GDAL_NUM_THREADS": "16",
           "GDAL_HTTP_MAX_RETRY": "4",
           "GDAL_HTTP_RETRY_DELAY": "2"}


def _s3_env():
    for k, v in _S3_ENV.items():
        os.environ.setdefault(k, v)


TILE_COLS = ["fid", "path", "year", "utm_zone", "crs",
             "utm_west", "utm_south", "utm_east", "utm_north",
             "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]


def dequant_np(s: np.ndarray) -> np.ndarray:
    """int8 -> float, signed-square (matches sources.ae.dequant; NOT renormed)."""
    x = s.astype(np.float32) / 127.5
    return np.square(x) * np.sign(x)


def l2norm_np(x: np.ndarray, axis=-1, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def training_tile_paths(chips_index, tiles_selection):
    """Every AEF source tile the compressor could have trained or eyeballed on:
    the `path`s in the training store's chip index and in the full-tile
    selection. Both files must exist, because an empty exclusion set would
    let training tiles into the eval set."""
    for p in (chips_index, tiles_selection):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"exclusion file {p} not found; the held-out eval set cannot "
                f"be built without the training tiles to exclude")
    return (set(pd.read_parquet(chips_index, columns=["path"]).path)
            | set(pd.read_csv(tiles_selection).path))


def held_out_tile_pool(index_path, exclude_paths, coverage_bbox=None,
                       years=None, full_only=True):
    """Full 8192^2 AEF tiles minus the training tiles, optionally clipped to a
    benchmark's WGS84 coverage bbox (w, s, e, n) and a set of years."""
    df = pd.read_parquet(index_path, columns=TILE_COLS)
    df = df[~df.path.isin(exclude_paths)]
    df = df[df.wgs84_west < df.wgs84_east]                 # drop antimeridian
    if full_only:
        df = df[((df.utm_east - df.utm_west) == FULL_M)
                & ((df.utm_north - df.utm_south) == FULL_M)]
    if coverage_bbox is not None:
        w, s, e, n = coverage_bbox
        df = df[(df.wgs84_east > w) & (df.wgs84_west < e)
                & (df.wgs84_north > s) & (df.wgs84_south < n)]
    if years is not None:
        df = df[df.year.isin(list(years))]
    return df.reset_index(drop=True)


def fetch_block(row, rng, max_nodata=0.0, tries=4):
    """Read one north-up 1024^2 all-band block from a random block of `row`'s
    tile. Returns (arr (64,1024,1024) int8, meta) or None. `max_nodata`=0.0
    means require a fully-valid block (windows are validated again on carve)."""
    _s3_env()
    vsis3 = row.path.replace("s3://", "/vsis3/")
    try:
        with rasterio.open(vsis3) as ds:
            bottom_up = ds.transform.e > 0
            order = rng.permutation(BLOCKS * BLOCKS)[:tries]
            for k in order:
                br, bc = int(k) // BLOCKS, int(k) % BLOCKS
                r_tiff = (TILE - (br + 1) * BLOCK) if bottom_up else br * BLOCK
                win = Window(bc * BLOCK, r_tiff, BLOCK, BLOCK)
                b1 = ds.read(1, window=win, out_shape=(64, 64))
                if (b1 == NODATA).mean() > max_nodata:
                    continue
                a = ds.read(window=win)
                if bottom_up:
                    a = a[:, ::-1, :]
                if (a[0] == NODATA).mean() > max_nodata:
                    continue
                return np.ascontiguousarray(a), {
                    "tile_path": row.path, "fid": int(row.fid),
                    "year": int(row.year), "crs": row.crs,
                    "utm_west": float(row.utm_west),
                    "utm_north": float(row.utm_north),
                    "block_row": br, "block_col": bc,
                }
    except Exception as e:  # noqa: BLE001 - transient HTTP/COG errors: skip tile
        print(f"  ! fetch failed ({type(e).__name__}): {row.path}: "
              f"{str(e)[:120]}", flush=True)
    return None


def carve_windows(block, meta, crop_sz, stride, max_per_window, rng,
                  max_crops=2):
    """Carve a north-up block into fully-valid `crop_sz`^2 context crops; sample
    target pixels on a `stride` grid inside each. Keeps at most `max_crops` crops
    per block so targets spread across many tiles (diversity + low storage).
    Returns (crops list of (64,crop_sz,crop_sz) int8, targets DataFrame with
    crop-local index/offsets, lon/lat, provenance)."""
    crops, rows = [], []
    grid = np.arange(0, crop_sz, stride)
    nwin = BLOCK // crop_sz
    order = rng.permutation(nwin * nwin)
    for k in order:
        if max_crops and len(crops) >= max_crops:
            break
        wr, wc = int(k) // nwin, int(k) % nwin
        crop = block[:, wr * crop_sz:(wr + 1) * crop_sz,
                     wc * crop_sz:(wc + 1) * crop_sz]
        if (crop[0] == NODATA).any():
            continue                                   # need a clean crop
        rr, cc = np.meshgrid(grid, grid, indexing="ij")
        rr, cc = rr.ravel(), cc.ravel()
        if max_per_window and len(rr) > max_per_window:
            sel = rng.permutation(len(rr))[:max_per_window]
            rr, cc = rr[sel], cc[sel]
        wi = len(crops)
        crops.append(np.ascontiguousarray(crop))
        # global north-up pixel -> UTM centre
        g_row = meta["block_row"] * BLOCK + wr * crop_sz + rr
        g_col = meta["block_col"] * BLOCK + wc * crop_sz + cc
        xs = meta["utm_west"] + (g_col + 0.5) * PX_M
        ys = meta["utm_north"] - (g_row + 0.5) * PX_M
        for i in range(len(rr)):
            rows.append({"window_idx": wi, "r": int(rr[i]), "c": int(cc[i]),
                         "utm_x": float(xs[i]), "utm_y": float(ys[i]),
                         **meta})
    if not crops:
        return [], pd.DataFrame()
    tdf = pd.DataFrame(rows)
    lon, lat = warp_transform(meta["crs"], "EPSG:4326",
                              tdf.utm_x.tolist(), tdf.utm_y.tolist())
    tdf["lon"], tdf["lat"] = lon, lat
    return crops, tdf


def build_windows(pool, n_targets, crop_sz=128, stride=10, max_per_window=16,
                  workers=12, seed=0, max_tiles=None, max_crops_per_tile=2):
    """Sample held-out tiles until ~n_targets pixels collected. Returns
    (windows (M,64,crop_sz,crop_sz) int8, targets DataFrame with a global
    `window_idx` indexing the crop that holds each target pixel)."""
    rng = np.random.default_rng(seed)
    if max_tiles is None:                          # ~3x for nodata/ocean/no-label drops
        per_tile = max(1, max_crops_per_tile * max_per_window)
        max_tiles = int(n_targets / per_tile * 3) + 20
    n_tiles = min(len(pool), max_tiles)
    tile_order = rng.permutation(len(pool))[:n_tiles]
    seeds = rng.integers(0, 2**31, size=len(tile_order))

    all_crops, all_tdf, base = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_block, pool.iloc[int(ti)],
                          np.random.default_rng(int(sd))): int(ti)
                for ti, sd in zip(tile_order, seeds)}
        n_done = 0
        for fut in as_completed(futs):
            res = fut.result()
            n_done += 1
            if res is None:
                continue
            block, meta = res
            crops, tdf = carve_windows(block, meta, crop_sz, stride,
                                       max_per_window,
                                       np.random.default_rng(meta["fid"]),
                                       max_crops=max_crops_per_tile)
            if not crops:
                continue
            tdf["window_idx"] = tdf["window_idx"] + base
            base += len(crops)
            all_crops.extend(crops)
            all_tdf.append(tdf)
            got = sum(len(t) for t in all_tdf)
            if n_done % 25 == 0 or got >= n_targets:
                print(f"  tiles {n_done}/{len(futs)} | windows {base} | "
                      f"targets {got}", flush=True)
            if got >= n_targets:
                for f in futs:
                    f.cancel()
                break
    if not all_crops:
        raise RuntimeError("no valid windows collected -- widen the pool/years")
    windows = np.stack(all_crops).astype(np.int8)
    targets = pd.concat(all_tdf, ignore_index=True)
    return windows, targets


# --------------------------------------------------------------------------- #
#  Point-driven extraction (for survey-point benchmarks: LUCAS, LCMAP)        #
#  The label points ARE the targets; we extract a 128px context crop around   #
#  each, dropping any point whose tile was used in training.                  #
# --------------------------------------------------------------------------- #
def _tile_lookup(aef_index_path, exclude_paths, coverage_bbox):
    """Held-out tiles in coverage, bucketed by floor(SW corner) for O(1) point
    containment lookup."""
    df = pd.read_parquet(aef_index_path, columns=TILE_COLS)
    df = df[~df.path.isin(exclude_paths)]
    df = df[df.wgs84_west < df.wgs84_east]
    df = df[((df.utm_east - df.utm_west) == FULL_M)        # full 8192^2 only:
            & ((df.utm_north - df.utm_south) == FULL_M)]   # no clipped reads
    if coverage_bbox is not None:
        w, s, e, n = coverage_bbox
        df = df[(df.wgs84_east > w) & (df.wgs84_west < e)
                & (df.wgs84_north > s) & (df.wgs84_south < n)]
    df = df.reset_index(drop=True)
    buckets = defaultdict(list)
    wv, sv = df.wgs84_west.to_numpy(), df.wgs84_south.to_numpy()
    for i in range(len(df)):
        buckets[(int(np.floor(wv[i])), int(np.floor(sv[i])))].append(i)
    return df, {k: np.array(v) for k, v in buckets.items()}


def _match_tile(lon, lat, prefer_year, df, buckets, W, E, S, N, YR):
    """Index of a held-out tile containing (lon, lat); prefer `prefer_year`."""
    flo, fla = int(np.floor(lon)), int(np.floor(lat))
    cand = [buckets[(flo + dx, fla + dy)] for dx in (-1, 0) for dy in (-1, 0)
            if (flo + dx, fla + dy) in buckets]
    if not cand:
        return -1
    c = np.concatenate(cand)
    hit = c[(W[c] <= lon) & (lon <= E[c]) & (S[c] <= lat) & (lat <= N[c])]
    if not len(hit):
        return -1
    pref = hit[YR[hit] == prefer_year]
    return int(pref[0]) if len(pref) else int(hit[0])


def _fetch_point_crops(row, members, lons, lats, labels, crop_sz, max_blocks=2):
    """Open one tile, read the `max_blocks` densest 1024-blocks holding matched
    points (COGs are 1024^2-tiled, so a lone crop read fetches a whole block --
    so we read the block ONCE and keep ALL its points), and slice every point's
    `crop_sz` context crop. Returns (crops, target rows)."""
    _s3_env()
    vsis3 = row.path.replace("s3://", "/vsis3/")
    out_crops, out_rows = [], []
    cpb = BLOCK // crop_sz                              # crops per block side
    try:
        with rasterio.open(vsis3) as ds:
            H, Wd = ds.height, ds.width
            bottom_up = ds.transform.e > 0
            xs, ys = warp_transform("EPSG:4326", row.crs, list(lons),
                                    list(lats))
            xs, ys = np.asarray(xs), np.asarray(ys)
            col_px = ((xs - row.utm_west) / PX_M).astype(int)
            row_px = ((row.utm_north - ys) / PX_M).astype(int)   # north-up
            blocks = {}
            for j in range(len(members)):
                rp, cp = row_px[j], col_px[j]
                br, bc = rp // BLOCK, cp // BLOCK
                if br * BLOCK + BLOCK > H or bc * BLOCK + BLOCK > Wd:
                    continue                           # block off the raster
                blocks.setdefault((br, bc), []).append(j)
            # densest blocks first => most points kept per (expensive) read
            chosen = sorted(blocks.items(), key=lambda kv: -len(kv[1]))
            if max_blocks:
                chosen = chosen[:max_blocks]
            for (br, bc), js in chosen:
                r_tiff = (H - (br + 1) * BLOCK) if bottom_up else br * BLOCK
                blk = ds.read(window=Window(bc * BLOCK, r_tiff, BLOCK, BLOCK))
                if bottom_up:
                    blk = blk[:, ::-1, :]
                crop_id = {}
                for j in js:
                    lcr = (row_px[j] // crop_sz) - br * cpb   # crop within block
                    lcc = (col_px[j] // crop_sz) - bc * cpb
                    if not (0 <= lcr < cpb and 0 <= lcc < cpb):
                        continue
                    crop = blk[:, lcr * crop_sz:(lcr + 1) * crop_sz,
                               lcc * crop_sz:(lcc + 1) * crop_sz]
                    if crop.shape[1:] != (crop_sz, crop_sz):
                        continue                       # clipped read at edge
                    if (crop[0] == NODATA).any():
                        continue                       # need a clean crop
                    if (lcr, lcc) not in crop_id:
                        crop_id[(lcr, lcc)] = len(out_crops)
                        out_crops.append(np.ascontiguousarray(crop))
                    out_rows.append({
                        "window_idx": crop_id[(lcr, lcc)],
                        "r": int(row_px[j] - (br * cpb + lcr) * crop_sz),
                        "c": int(col_px[j] - (bc * cpb + lcc) * crop_sz),
                        "lon": float(lons[j]), "lat": float(lats[j]),
                        **{k: v[j] for k, v in labels.items()},
                        "fid": int(row.fid), "tile_path": row.path,
                        "year": int(row.year)})
    except Exception as e:  # noqa: BLE001
        print(f"  ! point fetch failed ({type(e).__name__}): {row.path}: "
              f"{str(e)[:120]}", flush=True)
    return out_crops, out_rows


def extract_points(points, aef_index_path, exclude_paths, coverage_bbox,
                   prefer_year, crop_sz=128, n_targets=None, workers=16, seed=0,
                   max_blocks_per_tile=2, max_match=250000):
    """Extract a `crop_sz` context crop around each survey point from a held-out
    tile. `points` has columns lon, lat, label. Points whose containing tile was
    used in training (excluded) or fall outside coverage are dropped
    (contamination guard). Each tile reads only its `max_blocks_per_tile` densest
    1024-blocks and keeps ALL their points -- one ~30 MB block read yields many
    targets (e.g. ~26 LUCAS points/block on the 2 km grid) and spreads targets
    across many tiles. Returns (windows, targets) like build_windows."""
    rng = np.random.default_rng(seed)
    pts = points.dropna(subset=["lon", "lat", "label"]).reset_index(drop=True)
    if len(pts) > max_match:                           # cap matching cost
        pts = pts.iloc[rng.permutation(len(pts))[:max_match]] \
            .reset_index(drop=True)
    df, buckets = _tile_lookup(aef_index_path, exclude_paths, coverage_bbox)
    W, E = df.wgs84_west.to_numpy(), df.wgs84_east.to_numpy()
    S, N = df.wgs84_south.to_numpy(), df.wgs84_north.to_numpy()
    YR = df.year.to_numpy()
    tidx = np.array([_match_tile(lo, la, prefer_year, df, buckets, W, E, S, N,
                                 YR) for lo, la in zip(pts.lon, pts.lat)])
    ok = tidx >= 0
    print(f"  matched {ok.sum()}/{len(pts)} points to held-out tiles "
          f"(dropped {(~ok).sum()}: training-tile / outside coverage)",
          flush=True)
    pts, tidx = pts[ok].reset_index(drop=True), tidx[ok]

    by_tile = defaultdict(list)
    for i, ti in enumerate(tidx):
        by_tile[int(ti)].append(i)
    tile_ids = list(by_tile)
    rng.shuffle(tile_ids)
    lon_a, lat_a = pts.lon.to_numpy(), pts.lat.to_numpy()
    # carry through every label column (label, label_lu, ...) -> one extraction
    # can serve several downstream tasks at the same points (pick via --label-col)
    label_cols = [c for c in pts.columns if c == "label" or c.startswith("label_")]
    lab_arr = {c: pts[c].to_numpy() for c in label_cols}

    crops, rows, base = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for ti in tile_ids:
            mem = np.array(by_tile[ti])
            futs[ex.submit(_fetch_point_crops, df.iloc[ti], mem, lon_a[mem],
                           lat_a[mem], {c: a[mem] for c, a in lab_arr.items()},
                           crop_sz, max_blocks_per_tile)] = ti
        done = 0
        for fut in as_completed(futs):
            c, r = fut.result()
            done += 1
            for row in r:
                row["window_idx"] += base
            base += len(c)
            crops.extend(c)
            rows.extend(r)
            if done % 50 == 0 or (n_targets and len(rows) >= n_targets):
                print(f"  tiles {done}/{len(futs)} | crops {base} | "
                      f"targets {len(rows)}", flush=True)
            if n_targets and len(rows) >= n_targets:
                for f in futs:
                    f.cancel()
                break
    if not crops:
        raise RuntimeError("no valid point crops -- check coverage / year")
    windows = np.stack(crops).astype(np.int8)
    targets = pd.DataFrame(rows)
    return windows, targets
