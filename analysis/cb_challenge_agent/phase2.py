"""Phase 2: minimise residual.

Structural facts measured from the data (see report):
  * the 786432 vectors are 3072 contiguous groups of 256;
  * within-group variance is 0.083% of the energy, so there are effectively
    only 3072 distinct points, not 786432;
  * NN distance between group means scales like n^(-1/4), i.e. the data sits
    on a locally ~4-dimensional manifold in R^256.

So this is not a capacity problem (12288 codewords for 3072 effective
samples) but a coverage/generalisation problem.  Everything here is fitted
on group means and validated on held-out *groups*.
"""
import numpy as np
import torch

from mylib import DEV, K, assign, kmeans, t

NG, GS = 3072, 256
EX2 = 1077.1687          # E||x||^2 over the training vectors


def groups():
    V = np.load("/data/tessera-samples/cb_challenge/vectors_train.npy")
    X = t(V).view(NG, GS, 256)
    return V, X, X.mean(1)


def gsplit(n_val=512, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    p = torch.randperm(NG, generator=g, device=DEV)
    return p[n_val:], p[:n_val]


def rvq_apply(X, books):
    r = X.clone()
    per = []
    for C in books:
        r = r - C[assign(r, C)]
        per.append(float(r.pow(2).sum(1).mean()))
    return r, per


def resid_frac(X, books, denom=None):
    r, per = rvq_apply(X, books)
    d = denom if denom is not None else float(X.pow(2).sum(1).mean())
    return [p / d for p in per]


def augment(G, n_out, k=12, seed=0, mode="edge"):
    """Synthesise extra points on the manifold by interpolating between
    mutually-near group means.  Fights the 3072-sample coverage limit."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    D = torch.cdist(G, G) + torch.eye(G.shape[0], device=DEV) * 1e9
    nb = D.topk(k, largest=False).indices                    # (n,k)
    n = G.shape[0]
    i = torch.randint(0, n, (n_out,), generator=g, device=DEV)
    j = nb[i, torch.randint(0, k, (n_out,), generator=g, device=DEV)]
    lam = torch.rand(n_out, 1, generator=g, device=DEV)
    if mode == "edge":
        return G[i] * (1 - lam) + G[j] * lam
    raise ValueError(mode)


def crossfit_residuals(G, n_folds=6, kmeans_iters=25, seed=0, level0=None,
                       books_prev=()):
    """Residuals a *held-out* group would see.  Fitting levels 1/2 on the
    in-sample residuals of a memorising level 0 is the trap: those residuals
    are ~1000x smaller than what an unseen group produces."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    perm = torch.randperm(G.shape[0], generator=g, device=DEV)
    out = []
    for f in range(n_folds):
        va = perm[f::n_folds]
        tr = torch.cat([perm[o::n_folds] for o in range(n_folds) if o != f])
        C0 = level0(G[tr], seed=seed + f)
        r = G[va] - C0[assign(G[va], C0)]
        for C in books_prev:
            r = r - C[assign(r, C)]
        out.append(r)
    return torch.cat(out)
