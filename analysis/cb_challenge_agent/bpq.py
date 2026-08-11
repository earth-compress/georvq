"""Balanced Product Quantizer (BPQ).

Idea: if the K=L^m codewords form a Cartesian product of m scalar
codebooks along m *orthonormal* directions (plus any common offset), then
nearest-neighbour assignment factorises: the index digit along direction j
is just the scalar nearest-neighbour of <v,u_j>.  So if each scalar
quantizer is built so its decision boundaries sit exactly at the
j/L quantiles of the projection, every digit is exactly uniform and the
level entropy is m*log2(L) = 12 bits minus the mutual information between
digits.
"""
import numpy as np
import torch

from mylib import DEV, D, K, assign, entropy_bits, t


def _recon_from_boundaries(b):
    """Scalar reconstruction points whose midpoints are exactly the given
    boundaries b (sorted, length L-1).  Solves c_{i+1} = 2 b_i - c_i with a
    feasible starting offset d1 = b_1 - c_1 (need 0 < d_i < gap_i)."""
    b = np.asarray(b, dtype=np.float64)
    L = len(b) + 1
    if L == 2:
        return np.array([b[0] - 1.0, b[0] + 1.0])  # scale fixed later
    gap = np.diff(b)
    # d_i = A_i + (-1)^(i-1) d1  with A_1 = 0
    A = np.zeros(L - 1)
    for i in range(1, L - 1):
        A[i] = gap[i - 1] - A[i - 1]
    # constraints 0 < d_i < gap_i (last one only > 0)
    lo, hi = -np.inf, np.inf
    for i in range(L - 1):
        s = 1.0 if i % 2 == 0 else -1.0
        ub = gap[i] if i < L - 2 else np.inf
        # 0 < A_i + s*d1 < ub
        if s > 0:
            lo = max(lo, -A[i]); hi = min(hi, ub - A[i])
        else:
            lo = max(lo, A[i] - ub); hi = min(hi, A[i])
    if not (lo < hi):
        return None
    d1 = 0.5 * (max(lo, -1e9) + min(hi, 1e9))
    d = A + np.array([d1 if i % 2 == 0 else -d1 for i in range(L - 1)])
    c = np.empty(L)
    c[:L - 1] = b - d
    c[L - 1] = b[-1] + d[-1]
    return c


def build_level(X, m, L, dirs="pca", seed=0, alpha=None, use_offset=True):
    """X: (N,D) torch on GPU.  Returns codebook (K,D) torch."""
    assert L ** m == K
    N = X.shape[0]
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.t() @ Xc) / N
    ev, U = torch.linalg.eigh(cov.double())
    U = U.flip(1).float()                     # descending eigenvalue order
    if dirs == "pca":
        Umat = U[:, :m].contiguous()
    elif dirs.startswith("rot"):               # random rotation inside top-r
        r = int(dirs[3:])
        g = torch.Generator(device="cpu").manual_seed(seed)
        A = torch.randn(r, r, generator=g)
        Q, _ = torch.linalg.qr(A)
        Umat = (U[:, :r] @ Q.to(DEV))[:, :m].contiguous()
    elif dirs.startswith("skip"):              # take dirs m..2m (lower var)
        s = int(dirs[4:])
        Umat = U[:, s:s + m].contiguous()
    else:
        raise ValueError(dirs)

    P = X @ Umat                               # (N,m) projections
    Pn = P.cpu().numpy().astype(np.float64)
    qs = (np.arange(1, L) / L)
    cvals = np.zeros((m, L))
    for j in range(m):
        b = np.quantile(Pn[:, j], qs)
        b = np.maximum.accumulate(b)
        c = _recon_from_boundaries(b)
        if c is None:
            raise RuntimeError(f"infeasible scalar quantizer, dir {j}, L={L}")
        if L == 2:
            a = alpha if alpha is not None else float(
                np.abs(Pn[:, j] - b[0]).mean())
            c = np.array([b[0] - a, b[0] + a])
        cvals[j] = c

    # offset: part of the mean orthogonal to the chosen subspace
    o = mu - Umat @ (Umat.t() @ mu) if use_offset else torch.zeros_like(mu)

    digits = np.stack(np.unravel_index(np.arange(K), (L,) * m), 1)  # (K,m)
    coef = np.take_along_axis(cvals[None].repeat(K, 0),
                              digits[:, :, None], 2)[:, :, 0]       # (K,m)
    C = o[None] + t(coef) @ Umat.t()
    return C.contiguous()


def build_rvq(X, plan, seed=0, **kw):
    """plan: list of (m, L, dirs) per level."""
    r = X.clone()
    books, Hs = [], []
    for lv, (m, L, dirs) in enumerate(plan):
        C = build_level(r, m, L, dirs, seed=seed + lv, **kw)
        a = assign(r, C)
        Hs.append(entropy_bits(a))
        r = r - C[a]
        books.append(C)
    resid = float(r.pow(2).sum(1).mean() / X.pow(2).sum(1).mean())
    return books, Hs, resid
