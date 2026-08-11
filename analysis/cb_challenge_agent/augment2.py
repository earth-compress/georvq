"""Augmentation schemes for the level-0 covering problem.

There are only 3072 distinct points and 4096 codewords, so fitting k-means to
the samples memorises them.  What we actually want is to minimise
E[min_c ||y-c||^2] for a *fresh* y, i.e. k-means on the underlying density.
These routines synthesise samples from that density using the local manifold
structure (intrinsic dimension ~4).
"""
import torch

from mylib import DEV


def _nbrs(G, k):
    D = torch.cdist(G, G) + torch.eye(G.shape[0], device=DEV) * 1e9
    return D.topk(k, largest=False).indices


def edge(G, n, k=16, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = _nbrs(G, k)
    i = torch.randint(0, G.shape[0], (n,), generator=g, device=DEV)
    j = nb[i, torch.randint(0, k, (n,), generator=g, device=DEV)]
    lam = torch.rand(n, 1, generator=g, device=DEV)
    return G[i] * (1 - lam) + G[j] * lam


def simplex(G, n, k=16, s=4, alpha=1.0, seed=0):
    """Dirichlet-weighted convex combination of a point and s-1 of its
    neighbours: fills the interior of the local manifold patch, not just the
    edges between sample pairs."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = _nbrs(G, k)
    i = torch.randint(0, G.shape[0], (n,), generator=g, device=DEV)
    sel = torch.randint(0, k, (n, s - 1), generator=g, device=DEV)
    idx = torch.cat([i[:, None], nb[i].gather(1, sel)], 1)          # (n,s)
    w = -torch.rand(n, s, generator=g, device=DEV).log() * alpha
    w = w / w.sum(1, keepdim=True)
    return (G[idx] * w[:, :, None]).sum(1)


def local_gauss(G, n, k=16, shrink=1.0, seed=0):
    """Kernel density estimate on the manifold: sample from the local
    covariance of each point's k-neighbourhood."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = _nbrs(G, k)
    i = torch.randint(0, G.shape[0], (n,), generator=g, device=DEV)
    P = G[nb[i]]                                                    # (n,k,256)
    mu = P.mean(1)
    Dv = P - mu[:, None]
    w = torch.randn(n, k, generator=g, device=DEV) / (k ** 0.5)
    return G[i] + shrink * torch.einsum("nk,nkd->nd", w, Dv)


def tangent(G, n, k=16, dim=4, scale=1.0, seed=0):
    """Sample in the local tangent space spanned by the top `dim` local PCA
    directions, with the empirical local scale."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = _nbrs(G, k)
    i = torch.randint(0, G.shape[0], (n,), generator=g, device=DEV)
    P = G[nb[i]]
    Dv = P - P.mean(1, keepdim=True)
    U, S, _ = torch.linalg.svd(Dv, full_matrices=False)
    c = torch.randn(n, min(dim, S.shape[1]), generator=g, device=DEV)
    coef = c * S[:, :c.shape[1]] / (k ** 0.5)
    return G[i] + scale * torch.einsum("nj,nkj->nk", coef,
                                       torch.linalg.svd(Dv, full_matrices=False)[2].transpose(1, 2)[:, :, :c.shape[1]])
