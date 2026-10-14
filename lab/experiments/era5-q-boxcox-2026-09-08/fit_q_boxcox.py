"""Fit the per-level Box-Cox block for era5:q on the TRAIN store and
write it into georvq/sources/era5_stats.json under q.boxcox
(lam bounded to [0,1] by minimum |skew|; mean/std of the transformed
values). The raw q moments are left untouched."""
import json
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import numpy as np
from scipy import stats as sps

from georvq.data.era5 import Era5Store

EPS = 1e-8
PATH = ("/home/ali/projects/georvq/.claude/worktrees/era5-model/"
        "georvq/sources/era5_stats.json")

st = Era5Store("/data/ERA5-samples/train/store_q")
rng = np.random.default_rng(7)
rows = rng.choice(len(st.index), 256, replace=False)
xs = []
for r in rows:
    row = st.index.iloc[int(r)]
    xs.append(np.asarray(st.read_slice(int(row.shard), int(row.slot)),
                         dtype=np.float64))
x = np.stack(xs)
print("train sample", x.shape)

grid = np.round(np.linspace(0.0, 1.0, 21), 2)
lam, mean, std = [], [], []
for i in range(37):
    v = np.clip(x[:, i], 0, None).ravel() + EPS
    sub = rng.choice(v, 400_000, replace=False)
    best = (np.inf, 1.0)
    for lb in grid:
        y = np.log(sub) if lb == 0 else (sub ** lb - 1) / lb
        best = min(best, (abs(float(sps.skew(y))), float(lb)))
    lb = best[1]
    y = np.log(v) if lb == 0 else (v ** lb - 1) / lb
    lam.append(lb)
    mean.append(float(y.mean()))
    std.append(float(y.std()))
    print(f"level {i:2d}: lam {lb:.2f} mean {y.mean():.4g} sd {y.std():.4g}")

with open(PATH) as f:
    stats = json.load(f)
stats["q"]["boxcox"] = {"eps": EPS, "lam": lam, "mean": mean, "std": std,
                        "fit": "train store, 256 slices, seed 7, "
                               "lam grid [0,1]/0.05, min |skew|, "
                               "2026-09-08"}
with open(PATH, "w") as f:
    json.dump(stats, f, indent=1)
print("wrote boxcox block to", PATH)
