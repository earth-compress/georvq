"""Full phase-2 pipeline.

level 0  covering codebook: k-means on group means + edge-interpolated
         synthetic manifold samples (fixes the 3072-samples / 4096-codewords
         memorisation).
level 1  fitted on CROSS-FITTED residuals: residuals that groups held out of
         the level-0 fit actually produce.  Fitting on in-sample residuals is
         the trap that makes levels 1-2 worthless (train residual -> 0).
level 2  same, on cross-fitted residuals after levels 0 and 1.
"""
import numpy as np
import torch

from mylib import DEV, assign, kmeans, t
from pq import fit_pq

K = 4096


def edge_aug(Gs, n, k=12, a=0.5, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    D = torch.cdist(Gs, Gs) + torch.eye(Gs.shape[0], device=DEV) * 1e9
    nb = D.topk(min(k, Gs.shape[0] - 1), largest=False).indices
    i = torch.randint(0, Gs.shape[0], (n,), generator=g, device=DEV)
    j = nb[i, torch.randint(0, nb.shape[1], (n,), generator=g, device=DEV)]
    u = torch.rand(n, 1, generator=g, device=DEV)
    lam = u if a == 1 else (u ** a) / (u ** a + (1 - u) ** a)
    return Gs[i] * (1 - lam) + Gs[j] * lam


def level0(Gs, naug=300000, k=12, a=0.5, iters=50, seed=0):
    A = edge_aug(Gs, naug, k=k, a=a, seed=seed + 1)
    return kmeans(torch.cat([Gs, A]), iters=iters, seed=seed)


def crossfit(Gs, n_parts=3, n_folds=12, books_prev=(), l0kw=None, seed=0):
    """Residual pool as seen by groups excluded from the level-0 fit."""
    l0kw = l0kw or {}
    out = []
    for p in range(n_parts):
        g = torch.Generator(device=DEV).manual_seed(seed * 100 + p)
        perm = torch.randperm(Gs.shape[0], generator=g, device=DEV)
        for f in range(n_folds):
            va = perm[f::n_folds]
            tr = torch.cat([perm[o::n_folds] for o in range(n_folds)
                            if o != f])
            C0 = level0(Gs[tr], seed=seed * 100 + p * 20 + f, **l0kw)
            r = Gs[va] - C0[assign(Gs[va], C0)]
            for C in books_prev:
                r = r - C[assign(r, C)]
            out.append(r)
    return torch.cat(out)


def build(Gs, l1="vq", l2="vq", scale1=1.0, scale2=1.0, n_parts=3,
          n_folds=12, seed=0, l0kw=None, verbose=True):
    l0kw = l0kw or {}
    C0 = level0(Gs, seed=seed, **l0kw)
    R1 = crossfit(Gs, n_parts, n_folds, (), l0kw, seed)
    C1 = (kmeans(R1, iters=50, seed=seed + 1) if l1 == "vq"
          else fit_pq(R1, l1[0], l1[1], iters=40, seed=seed + 1)) * scale1
    R2 = crossfit(Gs, n_parts, n_folds, (C1,), l0kw, seed)
    C2 = (kmeans(R2, iters=50, seed=seed + 2) if l2 == "vq"
          else fit_pq(R2, l2[0], l2[1], iters=40, seed=seed + 2)) * scale2
    if verbose:
        print(f"   pool sizes {R1.shape[0]}/{R2.shape[0]}, "
              f"rms {float(R1.pow(2).sum(1).mean().sqrt()):.3f}/"
              f"{float(R2.pow(2).sum(1).mean().sqrt()):.3f}")
    return [C0, C1, C2]
