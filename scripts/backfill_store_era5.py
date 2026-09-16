"""Back-fill variables into an existing ERA5 cube store.

The store's manifest.parquet fixes every cube's window start and crop,
so a new ARCO variable can be fetched at exactly the hours and places
the original layers occupy. Fetching, quantization and layout reuse
build_store_era5.py; this script adds the arrays for the requested
variables, sized to the manifest's cube axis, and fills them window by
window. Progress is kept per variable in backfill_<var>.json so passes
for different variables can run concurrently and any of them can be
resumed. A window is committed only after all its cubes are written.

Variables with missing values (sea surface temperature over land) use a
reserved code: int16 -32768 marks NaN and the affine maps the remaining
65535 codes onto [min, max] of the finite values,

    valid = q != -32768
    x = (q.astype(int32) + 32767) * scale + offset     where valid

The store attrs record which variables use this form under
"nan_coded_vars"; every other variable keeps the original dequant.

Usage:
  python scripts/backfill_store_era5.py --split val --vars sst mtpr mtnlwrf
  python scripts/backfill_store_era5.py --split train --vars ciwc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_store_era5 import (NLON, fetch_var, open_arco,  # noqa: E402
                              quantize)

# ClimateBenchPress equivalents: sst ~ CMIP6 tos, mtpr = avg_tprate
# (NextGEMS pr), mtnlwrf = avg_tnlwrf (NextGEMS rlut), ciwc = IFS ciwc
# (37 pressure levels here, 137 model levels in the benchmark).
BACKFILL_VARS = {
    "sst": ("sea_surface_temperature", False, True),
    "mtpr": ("mean_total_precipitation_rate", False, False),
    "mtnlwrf": ("mean_top_net_long_wave_radiation_flux", False, False),
    "ciwc": ("specific_cloud_ice_water_content", True, False),
    "tp": ("total_precipitation", False, False),
    "ttr": ("top_net_thermal_radiation", False, False),
}
NAN_CODE = -32768


def quantize_nan(buf):
    """(k, [L,] T, y, x) fp32 with NaNs -> int16 with the reserved code."""
    flat = buf.reshape(buf.shape[:-3] + (-1,))
    lo = np.nanmin(flat, axis=-1)
    hi = np.nanmax(flat, axis=-1)
    lo = np.where(np.isfinite(lo), lo, 0.0).astype(np.float32)
    hi = np.where(np.isfinite(hi), hi, 0.0).astype(np.float32)
    scale = np.maximum((hi.astype(np.float64) - lo) / 65534.0,
                       np.finfo(np.float32).tiny).astype(np.float32)
    nan = np.isnan(buf)
    buf -= lo[..., None, None, None]
    buf /= scale[..., None, None, None]
    np.rint(buf, out=buf)
    np.clip(buf, 0, 65534, out=buf)
    buf[nan] = -1.0
    q = (buf.astype(np.int32) - 32767).astype(np.int16)
    return q, scale.astype(np.float64), lo.astype(np.float64)


def ensure_arrays(root, names):
    import zarr  # noqa: F401
    n = root["t"].shape[0]
    L, hours, px = root["t"].shape[1:4]
    created = []
    for name in names:
        if name in root:
            continue
        arco, has_levels, nan_coded = BACKFILL_VARS[name]
        if has_levels:
            root.create_array(name, shape=(n, L, hours, px, px),
                              chunks=(1, L, min(hours, 24), px, px),
                              dtype="int16")
            for s in ("_scale", "_offset"):
                root.create_array(name + s, shape=(n, L), chunks=(4096, L),
                                  dtype="float64")
        else:
            root.create_array(name, shape=(n, hours, px, px),
                              chunks=(1, min(hours, 24), px, px),
                              dtype="int16")
            for s in ("_scale", "_offset"):
                root.create_array(name + s, shape=(n,), chunks=(4096,),
                                  dtype="float64")
        created.append(name)
    if created:
        attrs = dict(root.attrs)
        arco_names = dict(attrs.get("arco_names", {}))
        pl = list(attrs.get("pl_vars", []))
        sfc = list(attrs.get("sfc_vars", []))
        nan_vars = list(attrs.get("nan_coded_vars", []))
        for name in created:
            arco, has_levels, nan_coded = BACKFILL_VARS[name]
            arco_names[name] = arco
            (pl if has_levels else sfc).append(name)
            if nan_coded:
                nan_vars.append(name)
        root.attrs.update({
            "arco_names": arco_names, "pl_vars": pl, "sfc_vars": sfc,
            "nan_coded_vars": nan_vars,
            "dequant_nan": "valid = q != -32768; x = (q.astype(int32) + "
                           "32767) * scale + offset where valid",
            "backfilled_vars": sorted(set(attrs.get("backfilled_vars", []))
                                      | set(created)),
        })
    return created


def state_path(out_dir, name):
    return os.path.join(out_dir, f"backfill_{name}.json")


def load_state(out_dir, name):
    p = state_path(out_dir, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"done_windows": []}


def save_state(out_dir, name, st):
    p = state_path(out_dir, name)
    with open(p + ".tmp", "w") as f:
        json.dump(st, f)
    os.replace(p + ".tmp", p)


def verify_one(root, ds, name, cube_id, row, rng):
    arco, has_levels, nan_coded = BACKFILL_VARS[name]
    hours, px = root.attrs["hours"], root.attrs["px"]
    h = int(rng.integers(hours))
    t_axis0 = ds.time.values[0].astype("datetime64[h]")
    ti = int((np.datetime64(row["t0"], "h") - t_axis0)
             / np.timedelta64(1, "h")) + h
    cubes = [(row["lat_center"], row["lon_center"], row["lat_row"],
              row["lon_col"])]
    ref = fetch_var(ds, arco, ti, 1, cubes, px, 1)[0]
    q = root[name][cube_id]
    sc, off = root[name + "_scale"][cube_id], root[name + "_offset"][cube_id]
    if has_levels:
        qh, refh = q[:, h], ref[:, 0]
        sc_b, off_b = sc[:, None, None], off[:, None, None]
    else:
        qh, refh = q[h], ref[0]
        sc_b, off_b = sc, off
    if nan_coded:
        valid = qh != NAN_CODE
        if not np.array_equal(valid, ~np.isnan(refh)):
            print(f"  verify cube {cube_id} {name}: NaN mask mismatch -> FAIL",
                  flush=True)
            return False
        dec = (qh.astype(np.int32) + 32767) * sc_b + off_b
        err = np.abs(np.where(valid, dec - refh, 0.0))
    else:
        dec = (qh.astype(np.int32) + 32768) * sc_b + off_b
        err = np.abs(dec - refh)
    bound = np.broadcast_to(sc_b * 0.505 + 1e-9, err.shape)
    ok = bool((err <= bound).all())
    print(f"  verify cube {cube_id} {name} h={h}: max|err|={err.max():.3e} "
          f"max scale/2={float(np.max(bound)):.3e} -> "
          f"{'ok' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--vars", nargs="+", required=True,
                    choices=list(BACKFILL_VARS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="stop after this many windows of the manifest")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--init-only", action="store_true",
                    help="create the arrays and exit")
    args = ap.parse_args()
    out_dir = args.out_dir or f"/data/ERA5-samples/{args.split}"

    import zarr
    root = zarr.open_group(os.path.join(out_dir, "cubes.zarr"), mode="a")
    created = ensure_arrays(root, args.vars)
    print(f"arrays present for {args.vars} (created {created})", flush=True)
    if args.init_only:
        return

    man = pd.read_parquet(os.path.join(out_dir, "manifest.parquet"))
    assert len(man) == root["t"].shape[0], "manifest and store disagree"
    windows = sorted(man["window"].unique())
    if args.max_windows is not None:
        windows = windows[:args.max_windows]
    hours, px = root.attrs["hours"], root.attrs["px"]

    ds, _ = open_arco()
    t_axis0 = ds.time.values[0].astype("datetime64[h]")
    vrng = np.random.default_rng(7)

    for name in args.vars:
        arco, has_levels, nan_coded = BACKFILL_VARS[name]
        st = load_state(out_dir, name)
        done = set(st["done_windows"])
        todo = [w for w in windows if w not in done]
        print(f"{name} ({arco}): {len(done)} windows done, {len(todo)} to go",
              flush=True)
        t_start = time.time()
        for i, w in enumerate(todo):
            rows = man[man["window"] == w].sort_values("cube_id")
            ids = rows["cube_id"].to_numpy()
            assert np.all(np.diff(ids) == 1), "cubes of a window not contiguous"
            t0 = np.datetime64(rows["t0"].iloc[0], "h")
            ti0 = int((t0 - t_axis0) / np.timedelta64(1, "h"))
            cubes = [(r.lat_center, r.lon_center, int(r.lat_row),
                      int(r.lon_col)) for r in rows.itertuples()]
            tw = time.time()
            buf = fetch_var(ds, arco, ti0, hours, cubes, px, args.workers)
            if nan_coded:
                q, scale, offset = quantize_nan(buf)
            else:
                if np.isnan(buf).any():
                    raise RuntimeError(f"{name}: NaN in window {w} but the "
                                       "variable is not NaN-coded")
                q, scale, offset = quantize(buf)
            sl = slice(int(ids[0]), int(ids[-1]) + 1)
            root[name][sl] = q
            root[name + "_scale"][sl] = scale
            root[name + "_offset"][sl] = offset
            st["done_windows"] = sorted(done | {int(w)})
            done.add(int(w))
            save_state(out_dir, name, st)
            if args.verify:
                cid = int(vrng.choice(ids))
                row = man.iloc[cid]
                if not verify_one(root, ds, name, cid, row, vrng):
                    raise SystemExit("verification failed, stopping")
            rate = (time.time() - t_start) / (i + 1)
            print(f"  {name} window {w} t0={t0} done in {time.time() - tw:.0f}s "
                  f"(~{rate * (len(todo) - i - 1) / 3600:.1f}h remaining)",
                  flush=True)
        st["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_state(out_dir, name, st)
        print(f"{name}: complete over {len(done)} windows", flush=True)


if __name__ == "__main__":
    main()
