"""Push the tail-subspace balanced product quantizer toward 12.000 bits.

Variants tried per level: number of directions m x levels-per-direction L
(m*log2(L)=12), PCA-axis-aligned vs randomly rotated inside a wider tail
subspace, and a group-disjoint (contiguous) train/val split as a proxy for
the holdout's disjoint-group shift.
"""
import numpy as np
import torch

from mylib import DEV, K, assign, entropy_bits, t
from bpq import _recon_from_boundaries

D = 256


def pca(X):
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.t() @ Xc) / X.shape[0]
    ev, U = torch.linalg.eigh(cov.double())
    return ev.flip(0).float(), U.flip(1).float()


def build(Xtr, U, idx, m, L, rot_from=None, seed=0):
    """Balanced product quantizer on m orthonormal directions, zero offset."""
    if rot_from is not None:
        pool = U[:, rot_from].contiguous()
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(pool.shape[1], pool.shape[1], generator=g)
        Q, _ = torch.linalg.qr(A)
        Um = (pool @ Q.to(DEV))[:, :m].contiguous()
    else:
        Um = U[:, idx[:m]].contiguous()
    P = (Xtr @ Um).cpu().numpy().astype(np.float64)
    qs = np.arange(1, L) / L
    cv = np.zeros((m, L))
    for j in range(m):
        b = np.maximum.accumulate(np.quantile(P[:, j], qs))
        if L == 2:
            cv[j] = [b[0] - 1.0, b[0] + 1.0]
        else:
            c = _recon_from_boundaries(b)
            if c is None:
                return None
            cv[j] = c
    dig = np.stack(np.unravel_index(np.arange(K), (L,) * m), 1)
    coef = np.take_along_axis(cv[None].repeat(K, 0), dig[:, :, None], 2)[:, :, 0]
    return (t(coef) @ Um.t()).contiguous()
