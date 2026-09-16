"""Build a CMIP6 monthly store of air temperature (ta, Amon, 19 pressure
levels) and sea surface temperature (tos, Omon, native ocean grid) from
the Google Cloud CMIP6 zarr mirror (anonymous).

ClimateBenchPress evaluates year 2020 of ssp585 from ACCESS-ESM1-5,
CanESM5 and UKESM1-0-LL, so year 2020 is withheld from every training
store here and written separately as <model>_ssp585_<var>_2020.zarr
where the model has it.

Each model's grid is kept as is, because the grids differ between
models and the benchmark compresses native fields. One zarr per
(model, experiment, variable) holds the float32 field with its
coordinates, chunked one year at a time. A .done marker per store is
the resume point.

Usage:
  python scripts/build_store_cmip6.py --out-dir /data/CMIP6-samples
  python scripts/build_store_cmip6.py --models CanESM5 --vars tos
"""
from __future__ import annotations

import argparse
import json
import os
import time

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

CATALOG = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"

# Benchmark models first. The rest add grid diversity (64x128 up to
# 192x288 for the atmosphere) at a modest download size.
MODELS = ["ACCESS-ESM1-5", "CanESM5", "UKESM1-0-LL", "MRI-ESM2-0",
          "GFDL-ESM4", "MPI-ESM1-2-LR", "MIROC6", "IPSL-CM6A-LR",
          "NorESM2-MM"]
VARS = {"ta": "Amon", "tos": "Omon"}
EXPERIMENTS = {"historical": ("1950", "2014"), "ssp585": ("2015", "2100")}
HOLDOUT_YEAR = "2020"


def pick_store(df, model, exp, var):
    q = df[(df.source_id == model) & (df.experiment_id == exp)
           & (df.variable_id == var) & (df.table_id == VARS[var])]
    if q.empty:
        return None, None
    # prefer the r1 member; otherwise the first member listed
    q = q.sort_values("member_id")
    r1 = q[q.member_id.str.startswith("r1i1")]
    row = (r1 if not r1.empty else q).iloc[0]
    return row.zstore, row.member_id


def open_gcs(zstore):
    url = zstore.replace("gs://", "https://storage.googleapis.com/")
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True)


def write(ds, var, path, chunks):
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        import shutil
        shutil.rmtree(tmp)
    keep = [var] + [c for c in ("lat", "lon", "plev", "latitude", "longitude",
                                "lat_bnds", "lon_bnds", "plev_bnds",
                                "vertices_latitude", "vertices_longitude")
                    if c in ds.variables]
    out = ds[keep]
    out = out.drop_vars([c for c in out.coords if c not in out.dims
                         and c not in keep and c != "time"], errors="ignore")
    out = out.chunk(chunks)
    for v in out.variables:
        out[v].encoding = {}
    out.to_zarr(tmp, mode="w", consolidated=True)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/CMIP6-samples")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--vars", nargs="+", default=list(VARS))
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(CATALOG)
    log = []
    for model in args.models:
        for var in args.vars:
            for exp in args.experiments:
                name = f"{model}_{exp}_{var}"
                path = os.path.join(args.out_dir, name + ".zarr")
                done = path + ".done"
                if os.path.exists(done):
                    print(f"{name}: present", flush=True)
                    continue
                zstore, member = pick_store(df, model, exp, var)
                if zstore is None:
                    print(f"{name}: not in catalog, skipped", flush=True)
                    log.append({"store": name, "status": "missing"})
                    continue
                t0 = time.time()
                ds = open_gcs(zstore)
                y0, y1 = EXPERIMENTS[exp]
                ds = ds.sel(time=slice(y0, y1))
                years = ds.time.dt.year
                tdim = {"time": 12}
                chunks = {**tdim, **{d: -1 for d in ds[var].dims if d != "time"}}
                train = ds.sel(time=(years != int(HOLDOUT_YEAR)))
                write(train, var, path, chunks)
                if exp == "ssp585":
                    hold = ds.sel(time=(years == int(HOLDOUT_YEAR)))
                    if hold.time.size:
                        write(hold, var,
                              os.path.join(args.out_dir,
                                           f"{name}_{HOLDOUT_YEAR}.zarr"),
                              chunks)
                meta = {"store": name, "zstore": zstore, "member": member,
                        "years": [y0, y1], "holdout_year": HOLDOUT_YEAR,
                        "shape": [int(s) for s in train[var].shape],
                        "dims": list(train[var].dims),
                        "gb": float(train[var].nbytes / 1e9),
                        "seconds": round(time.time() - t0)}
                with open(done, "w") as f:
                    json.dump(meta, f, indent=1)
                log.append({**meta, "status": "ok"})
                print(f"{name}: {meta['shape']} {meta['gb']:.2f} GB "
                      f"in {meta['seconds']}s", flush=True)
    with open(os.path.join(args.out_dir, "build_log.json"), "a") as f:
        for rec in log:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
