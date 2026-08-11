"""Fit the phase-2 submissions on all 3072 training groups."""
import numpy as np
import torch

from mylib import DEV, assign, kmeans, t
from phase2 import EX2, groups
from pipeline import crossfit, level0

S1, S2 = 0.55, 0.80          # shrinkage tuned on the group-disjoint split

V, X, G = groups()

# --- reference baseline: plain greedy RVQ k-means on all vectors ----------
Xall = t(V)
r, base = Xall.clone(), []
for lv in range(3):
    C = kmeans(r, iters=40, seed=lv)
    base.append(C)
    r = r - C[assign(r, C)]
    print(f"baseline level{lv} train resid {float(r.pow(2).sum(1).mean())/EX2:.6f}")
for i, C in enumerate(base):
    np.save(f"base_C{i}.npy", C.cpu().numpy())
del Xall, r
torch.cuda.empty_cache()

# --- the pipeline --------------------------------------------------------
C0 = level0(G, seed=0)
print("L0 done")
R1 = crossfit(G, n_parts=6, n_folds=12, seed=0)
C1 = kmeans(R1, iters=60, seed=1) * S1
print("L1 done, pool", R1.shape[0])
R2 = crossfit(G, n_parts=6, n_folds=12, books_prev=(C1,), seed=0)
C2 = kmeans(R2, iters=60, seed=2) * S2
print("L2 done, pool", R2.shape[0])
for i, C in enumerate([C0, C1, C2]):
    np.save(f"fin_C{i}.npy", C.cpu().numpy())

r = t(V).clone()
for C in [C0, C1, C2]:
    r = r - C[assign(r, C)]
    print("train-set resid (optimistic, groups seen):",
          float(r.pow(2).sum(1).mean()) / EX2)
print("saved base_C*, fin_C*")
