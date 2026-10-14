"""Build an ERA5 cube store by seeded space-time sampling of the ARCO-ERA5
0.25-degree hourly archive (gs://gcp-public-data-arco-era5, anonymous).

A cube is 128x128 px (32 deg) x 96 h (4 days) x 9 variables: t, z, u, v, q
on all 37 pressure levels plus msl, t2m, u10, v10 at the surface (189
channels). The ARCO archive is chunked as one full-globe field per hour,
so one hour costs the same transfer no matter how little of it is kept.
The sampler therefore draws 4-day time windows first and several
well-separated cubes per window, and the downloader streams each window
once, slicing all of its cubes per fetched hour.

Sampling is a pure function of (seed, split, window index, cube index)
via per-index Philox streams: window -> year uniform over the split's
years, start hour uniform within the year; cube -> lon ~ U(0, 360),
lat ~ arcsin(U) (equal-area) with centers within +/-66 deg and a minimum
great-circle separation inside the window. The store grows by extending
either index without disturbing anything already downloaded, and new
variables can be back-filled against the same manifest.

Values are stored as int16 with a per-(cube, level) affine, matching the
16-bit packing of the native ECMWF GRIB distribution:

    x = (q.astype(i32) + 32768) * scale + offset

Layout: one Zarr group per split. Per variable an int16 array of shape
(cube, level, time, y, x) - surface variables drop the level axis - plus
float64 <var>_scale / <var>_offset of shape (cube, level) or (cube,), and
per-cube coordinate arrays. manifest.parquet mirrors the cube axis and is
the resume point: a window is committed only once all its cubes are
written, so an interrupted run re-does at most one window.

Usage:
  python scripts/build_store_era5.py --split train --n-windows 200   # ~1.9 TB
  python scripts/build_store_era5.py --split val --n-windows 20
  python scripts/build_store_era5.py --split train --n-windows 1 \
      --cubes-per-window 2 --hours 12 --out-dir /data/ERA5-samples/_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd

ARCO_URL = ("gs://gcp-public-data-arco-era5/ar/"
            "full_37-1h-0p25deg-chunk-1.zarr-v3")

PL_VARS = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
}
SFC_VARS = {
    "msl": "mean_sea_level_pressure",
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}

# 2019 and 2024 are withheld from train as whole years so hourly
# autocorrelation cannot leak across the split.
SPLITS = {
    "train": [y for y in range(1979, 2024) if y != 2019],
    "val": [2019, 2024],
}
SPLIT_ID = {"train": 0, "val": 1}

NLAT, NLON = 721, 1440
DEG = 0.25


def draw_window(seed, split, w):
    """Window index -> start time, uniform over the split's years."""
    rng = np.random.default_rng([seed, SPLIT_ID[split], 0, w])
    year = int(rng.choice(SPLITS[split]))
    y0 = np.datetime64(f"{year}-01-01T00")
    n_hours = int((np.datetime64(f"{year + 1}-01-01T00") - y0)
                  / np.timedelta64(1, "h"))
    return y0 + np.timedelta64(int(rng.integers(0, n_hours - 96)), "h")


def draw_cubes(seed, split, w, k, px, lat_max, min_sep):
    """Cube targets for one window: equal-area centers, pairwise separated
    by at least min_sep degrees great-circle. Deterministic given the
    stream indices and the draw order j = 0..k-1."""
    half = px * DEG / 2
    out = []
    for j in range(k):
        rng = np.random.default_rng([seed, SPLIT_ID[split], 1, w, j])
        for _ in range(200):
            lon_c = float(rng.uniform(0, 360))
            s = np.sin(np.radians(lat_max))
            lat_c = float(np.degrees(np.arcsin(rng.uniform(-s, s))))
            if all(_gc_deg(lat_c, lon_c, a, b) >= min_sep
                   for a, b, _, _ in out):
                break
        else:
            raise RuntimeError(
                f"window {w}: no room for cube {j} at min_sep={min_sep}")
        lat_row = int(np.clip(round((90 - lat_c - half) / DEG),
                              0, NLAT - px))
        lon_col = int(round(lon_c / DEG)) % NLON
        out.append((lat_c, lon_c, lat_row, lon_col))
    return out


def _gc_deg(lat1, lon1, lat2, lon2):
    p1, p2, dl = map(np.radians, (lat1, lat2, lon2 - lon1))
    c = (np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def open_arco():
    import xarray as xr
    ds = xr.open_zarr(ARCO_URL, chunks=None, storage_options={"token": "anon"})
    stop = np.datetime64(ds.attrs["valid_time_stop"])
    return ds, stop


def open_store(out_dir, px, hours, levels, args):
    """Open (creating on first use) the split's zarr group. Every array is
    appendable along axis 0; geometry is fixed at creation."""
    import zarr
    root = zarr.open_group(os.path.join(out_dir, "cubes.zarr"), mode="a")
    if "t" not in root:
        for name in PL_VARS:
            root.create_array(name, shape=(0, len(levels), hours, px, px),
                              chunks=(1, len(levels), min(hours, 24), px, px),
                              dtype="int16")
            for s in ("_scale", "_offset"):
                root.create_array(name + s, shape=(0, len(levels)),
                                  chunks=(4096, len(levels)), dtype="float64")
        for name in SFC_VARS:
            root.create_array(name, shape=(0, hours, px, px),
                              chunks=(1, min(hours, 24), px, px),
                              dtype="int16")
            for s in ("_scale", "_offset"):
                root.create_array(name + s, shape=(0,), chunks=(4096,),
                                  dtype="float64")
        lv = root.create_array("level", shape=(len(levels),), dtype="int16")
        lv[:] = np.asarray(levels, dtype=np.int16)
        root.attrs.update({
            "store_version": 1,
            "source": ARCO_URL,
            "dequant": "x = (q.astype(int32) + 32768) * scale + offset, "
                       "scale/offset per (cube, level)",
            "grid": f"0.25 deg regular lat-lon crop, {px}x{px} px, "
                    "row 0 is the northern edge",
            "hours": hours, "px": px,
            "pl_vars": list(PL_VARS), "sfc_vars": list(SFC_VARS),
            "arco_names": {**PL_VARS, **SFC_VARS},
            "builder_config": {k: v for k, v in vars(args).items()},
        })
    return root


def fetch_var(ds, arco_name, ti0, hours, cubes, px, workers,
              stall_timeout=180):
    """One variable, one window: fetch `hours` global fields concurrently
    and slice every cube from each. Returns fp32 (k, [L,] hours, px, px).

    A remote read can hang without raising, which the per-attempt retry
    cannot see. When no fetch completes for stall_timeout seconds, the
    outstanding hours are resubmitted on fresh threads and the stuck
    futures abandoned; the executor is shut down without joining so an
    abandoned thread cannot block the window (a duplicate late write of
    the same hour is harmless)."""
    da = ds[arco_name]
    has_levels = "level" in da.dims
    k = len(cubes)
    shape = ((k, da.sizes["level"], hours, px, px) if has_levels
             else (k, hours, px, px))
    buf = np.empty(shape, dtype=np.float32)

    def one_hour(h):
        for attempt in range(4):
            try:
                field = da[ti0 + h].values     # ([L,] 721, 1440)
                break
            except Exception:  # noqa: BLE001 - transient GCS reads
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 5)
        for ci, (_, _, r0, c0) in enumerate(cubes):
            cols = (c0 + np.arange(px)) % NLON
            crop = field[..., r0:r0 + px, :].take(cols, axis=-1)
            if has_levels:
                buf[ci, :, h] = crop
            else:
                buf[ci, h] = crop

    ex = ThreadPoolExecutor(max_workers=workers)
    tries = dict.fromkeys(range(hours), 1)
    futs = {ex.submit(one_hour, h): h for h in range(hours)}
    try:
        while futs:
            done, _ = wait(futs, timeout=stall_timeout,
                           return_when=FIRST_COMPLETED)
            if not done:
                for f, h in list(futs.items()):
                    if tries[h] >= 3:
                        raise RuntimeError(
                            f"{arco_name}: hour {h} stalled "
                            f"{tries[h]} times")
                    tries[h] += 1
                    del futs[f]
                    print(f"  ! {arco_name} hour {h} stalled "
                          f">{stall_timeout}s, resubmitting", flush=True)
                    futs[ex.submit(one_hour, h)] = h
                continue
            for f in done:
                futs.pop(f)
                f.result()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return buf


def quantize(buf):
    """(k, [L,] T, y, x) fp32 -> int16 plus per-(cube, level) affine."""
    flat = buf.reshape(buf.shape[:-3] + (-1,))
    lo = flat.min(axis=-1)                                     # float32
    hi = flat.max(axis=-1)
    scale = np.maximum((hi.astype(np.float64) - lo) / 65535.0,
                       np.finfo(np.float32).tiny).astype(np.float32)
    # in place in fp32: the fetch buffer for one pressure-level variable is
    # ~4 GiB and a fp64 broadcast version would peak at several times that.
    # The stored affine is exactly the fp32 one applied here, so the decode
    # error is scale/2 plus fp32 rounding of a quotient <= 65535.
    buf -= lo[..., None, None, None]
    buf /= scale[..., None, None, None]
    np.rint(buf, out=buf)
    np.clip(buf, 0, 65535, out=buf)
    q = (buf.astype(np.int32) - 32768).astype(np.int16)
    return q, scale.astype(np.float64), lo.astype(np.float64)


def verify(root, ds, cube_id, manifest_row, rng):
    """Re-fetch one random field of a stored cube and bound the quant
    error against the recorded scale."""
    r = manifest_row
    name = rng.choice(list(PL_VARS) + list(SFC_VARS))
    h = int(rng.integers(root.attrs["hours"]))
    ti = int((np.datetime64(r["t0"], "h") - ds.time.values[0].astype("datetime64[h]"))
             / np.timedelta64(1, "h")) + h
    cubes = [(r["lat_center"], r["lon_center"], r["lat_row"], r["lon_col"])]
    px = root.attrs["px"]
    ref = fetch_var(ds, {**PL_VARS, **SFC_VARS}[name], ti, 1, cubes, px, 1)[0]
    q = root[name][cube_id]
    sc, off = root[name + "_scale"][cube_id], root[name + "_offset"][cube_id]
    if q.ndim == 4:            # (L, T, y, x)
        dec = (q[:, h].astype(np.int32) + 32768) * sc[:, None, None] \
            + off[:, None, None]
        err = np.abs(dec - ref[:, 0])
        # 0.505: half a step plus fp32 rounding of the quotient (<= 65535,
        # so at most ~0.004 steps)
        bound = (sc * 0.505 + 1e-9)[:, None, None]
    else:
        dec = (q[h].astype(np.int32) + 32768) * sc + off
        err = np.abs(dec - ref[0])
        bound = sc * 0.505 + 1e-9
    ok = bool((err <= bound).all())
    print(f"  verify cube {cube_id} {name} h={h}: max|err|={err.max():.3e} "
          f"max scale/2={float(np.max(bound)):.3e} -> "
          f"{'ok' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="default /data/ERA5-samples/<split>")
    ap.add_argument("--split", choices=list(SPLITS), default="train")
    ap.add_argument("--n-windows", type=int, default=200)
    ap.add_argument("--cubes-per-window", type=int, default=16)
    ap.add_argument("--px", type=int, default=128)
    ap.add_argument("--hours", type=int, default=96)
    ap.add_argument("--lat-max", type=float, default=66.0,
                    help="max |lat| of a cube center")
    ap.add_argument("--min-sep", type=float, default=25.0,
                    help="min great-circle degrees between centers in a window")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verify", action="store_true",
                    help="re-fetch one random field per window and check the "
                         "dequantized values against the archive")
    args = ap.parse_args()
    out_dir = args.out_dir or f"/data/ERA5-samples/{args.split}"
    os.makedirs(out_dir, exist_ok=True)

    ds, valid_stop = open_arco()
    t_axis0 = ds.time.values[0].astype("datetime64[h]")
    for y in SPLITS[args.split]:
        assert np.datetime64(f"{y + 1}-01-01") <= valid_stop + 1, \
            f"split year {y} runs past the archive's valid_time_stop"

    root = open_store(out_dir, args.px, args.hours, ds.level.values, args)
    mpath = os.path.join(out_dir, "manifest.parquet")
    records, w_start = [], 0
    if os.path.exists(mpath):
        records = pd.read_parquet(mpath).to_dict("records")
        w_start = max(r["window"] for r in records) + 1 if records else 0
        print(f"resume: {len(records)} cubes, next window {w_start}",
              flush=True)
    n_done = len(records)

    cube_gib = (args.px ** 2 * args.hours *
                (len(PL_VARS) * 37 + len(SFC_VARS)) * 2) / 2 ** 30
    print(f"split={args.split} years={SPLITS[args.split][0]}.."
          f"{SPLITS[args.split][-1]} target={args.n_windows} windows x "
          f"{args.cubes_per_window} cubes ({cube_gib:.2f} GiB/cube "
          f"pre-compression) -> {out_dir}", flush=True)

    vrng = np.random.default_rng(args.seed + 99)
    t_start = time.time()
    for w in range(w_start, args.n_windows):
        t0 = draw_window(args.seed, args.split, w)
        cubes = draw_cubes(args.seed, args.split, w, args.cubes_per_window,
                           args.px, args.lat_max, args.min_sep)
        ti0 = int((t0 - t_axis0) / np.timedelta64(1, "h"))
        k = len(cubes)
        print(f"window {w}/{args.n_windows}: t0={t0} "
              f"centers={[(round(a, 1), round(b, 1)) for a, b, _, _ in cubes]}",
              flush=True)
        tw = time.time()
        for arr in root.array_keys():
            if arr != "level":
                root[arr].resize((n_done + k,) + root[arr].shape[1:])
        for name, arco_name in {**PL_VARS, **SFC_VARS}.items():
            buf = fetch_var(ds, arco_name, ti0, args.hours, cubes, args.px,
                            args.workers)
            q, scale, offset = quantize(buf)
            root[name][n_done:n_done + k] = q
            root[name + "_scale"][n_done:n_done + k] = scale
            root[name + "_offset"][n_done:n_done + k] = offset
            print(f"  {name}: fetched+quantized "
                  f"({time.time() - tw:6.0f}s elapsed)", flush=True)
        for j, (lat_c, lon_c, r0, c0) in enumerate(cubes):
            records.append({
                "cube_id": n_done + j, "split": args.split, "window": w,
                "j": j, "t0": str(t0), "year": int(str(t0)[:4]),
                "lat_center": lat_c, "lon_center": lon_c,
                "lat_row": r0, "lon_col": c0,
            })
        n_done += k
        df = pd.DataFrame(records)
        df.to_parquet(mpath + ".tmp")
        os.replace(mpath + ".tmp", mpath)   # window commit point
        if args.verify:
            cid = int(vrng.integers(n_done - k, n_done))
            if not verify(root, ds, cid, records[cid], vrng):
                raise SystemExit("verification failed, stopping")
        rate = (time.time() - t_start) / max(w - w_start + 1, 1)
        print(f"  window done in {time.time() - tw:.0f}s "
              f"({n_done} cubes, ~{rate * (args.n_windows - w - 1) / 3600:.1f}h"
              f" remaining)", flush=True)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"n_cubes": n_done, "n_windows": args.n_windows,
                   "cube_gib_pre_compression": cube_gib,
                   "builder_config": vars(args),
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  indent=2)
    print(f"done: {n_done} cubes ({n_done * cube_gib / 1024:.2f} TiB "
          f"pre-compression) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
