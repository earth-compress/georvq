"""Final: 3 levels x (16 x 16 x 16) balanced product quantizer on
mid-tail PCA directions {164-166}, {171-173}, {161-163}.

Each level's 4096 codewords are the Cartesian product of three 16-point
scalar codebooks on three orthonormal directions, with zero common offset.
Nearest-neighbour assignment therefore factorises into three independent
scalar quantizers whose decision boundaries are placed exactly at the
1/16..15/16 quantiles of the corresponding projection, so every digit is
marginally uniform.  Direction depth is chosen by sweep: shallower (higher
variance) directions have strongly dependent cell indices; deeper ones have
independent indices but codewords so closely spaced that the evaluator's
float32  ||x||^2 - 2x.c + ||c||^2  loses the ordering.  Depth ~165 is the
optimum of that trade-off.
"""
import numpy as np
import torch

from bpq import _recon_from_boundaries
from faithful import H, eval_assign, eval_stack
from mylib import DEV, t
from sweep_tail import pca

TRIPLES = [(164, 165, 166), (171, 172, 173), (161, 162, 163)]
L = 16


def build_all(X):
    ev, U = pca(X)
    P = (X @ U).cpu().numpy().astype(np.float64)
    dig = np.stack(np.unravel_index(np.arange(4096), (L, L, L)), 1)
    books = []
    for tri in TRIPLES:
        cv = []
        for i in tri:
            b = np.maximum.accumulate(np.quantile(P[:, i], np.arange(1, L) / L))
            c = _recon_from_boundaries(b)
            assert c is not None, f"infeasible dir {i}"
            cv.append(c)
        coef = np.stack([cv[j][dig[:, j]] for j in range(3)], 1)
        books.append((t(coef) @ U[:, list(tri)].t()).contiguous())
    return books


if __name__ == "__main__":
    V = np.load("/data/tessera-samples/cb_challenge/vectors_train.npy")
    books = build_all(t(V))                      # fit on all 786k vectors
    hs, us, resid, sp = eval_stack(t(V[700000:]), books)
    print("fit-all, contig-tail check:", [round(h, 4) for h in hs],
          "mean", round(float(np.mean(hs)), 4), "sump", round(sp, 1),
          "resid", resid, "used", us)
    for i, C in enumerate(books):
        np.save(f"/data/tessera-samples/cb_challenge/v2_C{i}.npy",
                C.cpu().numpy())
    print("saved v2_C0..2")
