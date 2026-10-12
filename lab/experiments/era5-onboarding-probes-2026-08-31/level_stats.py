"""Per-(var, level) mean/std over random slices of the train store.

These are candidates for the fixed standardization constants in
georvq/sources/era5.py, so they are computed on the TRAIN split.
"""
import json

import numpy as np
import pandas as pd
import zarr

STORE = "/data/ERA5-samples/train"
N_SLICES = 256

root = zarr.open(STORE + "/cubes.zarr", mode="r")
man = pd.read_parquet(STORE + "/manifest.parquet")
n_cubes = int(man.cube_id.max()) + 1
rng = np.random.default_rng(0)
out = {}
for var in ["t", "z", "u", "v", "q"]:
    a = root[var]
    sc = np.asarray(root[var + "_scale"], dtype=np.float64)
    off = np.asarray(root[var + "_offset"], dtype=np.float64)
    s0 = s1 = None
    n = 0
    for c, t in zip(rng.integers(0, n_cubes, N_SLICES),
                    rng.integers(0, a.shape[2], N_SLICES)):
        c, t = int(c), int(t)
        q = np.asarray(a[c, :, t]).astype(np.int64) + 32768
        x = q * sc[c][:, None, None] + off[c][:, None, None]
        f = x.reshape(x.shape[0], -1)
        if s0 is None:
            s0 = f.sum(1)
            s1 = (f ** 2).sum(1)
        else:
            s0 += f.sum(1)
            s1 += (f ** 2).sum(1)
        n += f.shape[1]
    mu = s0 / n
    sd = np.sqrt(np.maximum(s1 / n - mu ** 2, 0))
    out[var] = {"mean": [float(f"{m:.8g}") for m in mu],
                "std": [float(f"{s:.8g}") for s in sd]}
    print(var, "done: std range", float(sd.min()), float(sd.max()),
          flush=True)
with open("/home/ali/.claude/jobs/0e80fb90/tmp/probes/level_stats.json",
          "w") as f:
    json.dump(out, f, indent=1)
