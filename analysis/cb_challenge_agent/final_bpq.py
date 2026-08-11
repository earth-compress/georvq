"""Final construction: per level, a 64x64 balanced product quantizer on two
deep-tail PCA directions, zero offset.  Assignment factorises into two
independent scalar quantizers whose decision boundaries sit exactly at the
1/64...63/64 quantiles, so each digit is uniform by construction and the
level entropy is 12 bits minus the (tiny) mutual information between the
two digits.  Deep-tail directions are used because that is where the
sign/cell pattern of the data is closest to independent.
"""
import itertools
import sys

import numpy as np
import torch

from challenge_eval import score_local
from mylib import DEV, assign, entropy_bits, t
from sweep_tail import build, pca

V = np.load("/data/tessera-samples/cb_challenge/vectors_train.npy")
Xc, Xcv = t(V[:700000]), t(V[700000:])          # contiguous ~ group-disjoint
ev, Uc = pca(Xc)

# pick 3 disjoint direction pairs, greedily, scored on the contiguous val
cand = []
for a, b in itertools.combinations(range(236, 256), 2):
    C = build(Xc, Uc, torch.tensor([a, b], device=DEV), 2, 64)
    if C is not None:
        cand.append((entropy_bits(assign(Xcv, C)), a, b))
cand.sort(reverse=True)
chosen, used = [], set()
for h, a, b in cand:
    if a in used or b in used:
        continue
    chosen.append((a, b))
    used |= {a, b}
    if len(chosen) == 3:
        break
print("chosen pairs", chosen, "expected H", [round(c[0], 4) for c in cand[:1]])

# validate the full 3-level stack on the held-out contiguous tail
books = [build(Xc, Uc, torch.tensor(p, device=DEV), 2, 64) for p in chosen]
s = score_local([b.cpu().numpy() for b in books], V[700000:])
print("CONTIG-VAL", s["sum_perplexity"], s["mean_entropy_bits"], s["residual"],
      [round(l["entropy_bits"], 4) for l in s["per_level"]])

# refit on ALL training vectors, using the same PCA basis indices
Xa = t(V)
eva, Ua = pca(Xa)
final = [build(Xa, Ua, torch.tensor(p, device=DEV), 2, 64) for p in chosen]
s2 = score_local([b.cpu().numpy() for b in final], V[700000:])
print("FULLFIT on contig-val", s2["sum_perplexity"], s2["mean_entropy_bits"],
      s2["residual"])
for i, C in enumerate(final):
    np.save(f"/data/tessera-samples/cb_challenge/final_C{i}.npy", C.cpu().numpy())
print("saved")
