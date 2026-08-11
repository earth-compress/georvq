"""Final phase-2 model (v2) fitted on all 3072 training groups."""
import numpy as np
import torch

from mylib import assign, kmeans, t
from phase2 import EX2, groups
from pipeline2 import crossfit, level0

S1, S2 = 0.55, 0.80

V, X, G = groups()
C0 = level0(G, seed=0)
print("L0 done", flush=True)
R1 = crossfit(G, n_parts=8, n_folds=12, seed=0)
C1 = kmeans(R1, iters=70, seed=1) * S1
print("L1 done pool", R1.shape[0], flush=True)
R2 = crossfit(G, n_parts=8, n_folds=12, books_prev=(C1,), seed=0)
C2 = kmeans(R2, iters=70, seed=2) * S2
print("L2 done pool", R2.shape[0], flush=True)
for i, C in enumerate([C0, C1, C2]):
    np.save(f"fin2_C{i}.npy", C.cpu().numpy())
print("saved fin2_C*")
