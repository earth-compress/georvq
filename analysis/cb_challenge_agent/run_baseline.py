"""Reference point: plain residual k-means RVQ (distortion-optimal-ish)."""
import numpy as np
import torch

from challenge_eval import score_local
from mylib import assign, entropy_bits, kmeans, load_split, t

tr, val = load_split()
Xt, Xv = t(tr), t(val)
r = Xt.clone()
books = []
for lv in range(3):
    C = kmeans(r, iters=40, seed=lv, verbose=True)
    a = assign(r, C)
    print(f"level {lv}: trainH={entropy_bits(a):.3f}")
    r = r - C[a]
    books.append(C)
    np.save(f"/data/tessera-samples/cb_challenge/km_C{lv}.npy", C.cpu().numpy())
s = score_local([b.cpu().numpy() for b in books], val)
print("VAL", {k: s[k] for k in ("sum_perplexity", "mean_entropy_bits", "residual")})
print([l["entropy_bits"] for l in s["per_level"]], [l["used"] for l in s["per_level"]])
