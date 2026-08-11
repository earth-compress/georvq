"""Phase-2 pipeline v2: density-flattened manifold augmentation for level 0.

Optimal-quantizer theory says the codeword density should be the source
density raised to d/(d+2); with intrinsic dimension ~4 that means flattening
the empirical density.  Oversampling each group in proportion to its
nearest-neighbour distance does exactly that, and it is where the coverage
error actually lives.
"""
import numpy as np
import torch

from mylib import DEV, assign, kmeans, t

K = 4096


def wedge_aug(Gs, n=600000, k=12, a=0.5, p=1.0, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    D = torch.cdist(Gs, Gs) + torch.eye(Gs.shape[0], device=DEV) * 1e9
    kk = min(k, Gs.shape[0] - 1)
    nb = D.topk(kk, largest=False).indices
    nn = D.min(1).values
    w = (nn / nn.mean()) ** p
    i = torch.multinomial(w, n, replacement=True, generator=g)
    j = nb[i, torch.randint(0, kk, (n,), generator=g, device=DEV)]
    u = torch.rand(n, 1, generator=g, device=DEV)
    lam = u if a == 1 else (u ** a) / (u ** a + (1 - u) ** a)
    return Gs[i] * (1 - lam) + Gs[j] * lam


def level0(Gs, n=600000, k=12, a=0.5, p=1.0, iters=60, seed=0):
    A = wedge_aug(Gs, n, k, a, p, seed=seed + 1)
    return kmeans(torch.cat([Gs, A]), iters=iters, seed=seed)


def crossfit(Gs, n_parts=12, n_folds=12, books_prev=(), seed=0, l0kw=None):
    l0kw = l0kw or {}
    out = []
    for q in range(n_parts):
        g = torch.Generator(device=DEV).manual_seed(seed * 100 + q)
        perm = torch.randperm(Gs.shape[0], generator=g, device=DEV)
        for f in range(n_folds):
            va = perm[f::n_folds]
            tr = torch.cat([perm[o::n_folds] for o in range(n_folds)
                            if o != f])
            C0 = level0(Gs[tr], seed=seed * 100 + q * 20 + f, **l0kw)
            r = Gs[va] - C0[assign(Gs[va], C0)]
            for C in books_prev:
                r = r - C[assign(r, C)]
            out.append(r)
    return torch.cat(out)
