"""Temporal autocorrelation of ERA5 cube-store variables.

For each variable, sample cubes, remove each pixel's 96 h time mean (so
static geography does not inflate the correlation), and compute the
pooled anomaly autocorrelation at increasing hour lags. This is the
number that will decide the time-chunk length for a multitemporal codec;
the cycle-1 single-slice model does not consume it.
"""
import sys

import numpy as np
import zarr

STORE = "/data/ERA5-samples/val/cubes.zarr"
LAGS = [1, 2, 3, 6, 12, 24, 48]
N_CUBES = 12

root = zarr.open(STORE, mode="r")
rng = np.random.default_rng(0)

for var in ["t", "z", "u", "v", "q", "msl", "t2m"]:
    a = root[var]
    pl = a.ndim == 5
    sc = np.asarray(root[var + "_scale"], dtype=np.float64)
    off = np.asarray(root[var + "_offset"], dtype=np.float64)
    cubes = rng.choice(a.shape[0], N_CUBES, replace=False)
    num = {d: 0.0 for d in LAGS}
    den = 0.0
    n = {d: 0 for d in LAGS}
    nd = 0
    for c in cubes:
        c = int(c)
        q = np.asarray(a[c]).astype(np.int64) + 32768   # (L,T,y,x) or (T,y,x)
        if pl:
            x = q * sc[c][:, None, None, None] + off[c][:, None, None, None]
        else:
            x = (q * sc[c] + off[c])[None]
        x = x.astype(np.float32)
        anom = x - x.mean(axis=1, keepdims=True)
        den += float((anom ** 2).sum())
        nd += anom[:, 0].size * anom.shape[1]
        for d in LAGS:
            prod = anom[:, :-d] * anom[:, d:]
            num[d] += float(prod.sum())
            n[d] += prod.size
    den /= nd
    r = {d: (num[d] / n[d]) / den for d in LAGS}
    line = "  ".join(f"lag{d}h={r[d]:.3f}" for d in LAGS)
    half = next((d for d in LAGS if r[d] < 0.5), None)
    print(f"{var:>4}: {line}   (<0.5 at "
          f"{'>' + str(LAGS[-1]) if half is None else half} h)")
    sys.stdout.flush()
