"""Transform-coding codebook: reverse water-filling bit allocation over the
residual's PCA directions, scalar Lloyd-Max per direction, expanded to an
explicit (4096,256) Cartesian-product codebook.

This is the construction that actually attains the Gaussian rate-distortion
bound (up to the scalar granular gap), which matters for the refinement
levels: their input is a ~65-effective-dimension near-Gaussian cloud, where a
4096-entry VQ is data-starved but a transform code is not.
"""
import numpy as np
import torch

from mylib import DEV, K, t

TOTAL_BITS = 12


def allocate(ev, total=TOTAL_BITS, max_bits=8):
    """Greedy marginal-gain integer bit allocation (reverse water-filling)."""
    ev = np.asarray(ev, dtype=np.float64)
    b = np.zeros(len(ev), dtype=int)
    for _ in range(total):
        # halving distortion per extra bit: gain = lam*4^-b * (1 - 1/4)
        gain = ev * (4.0 ** (-b.astype(np.float64)))
        gain[b >= max_bits] = -1
        b[int(np.argmax(gain))] += 1
    return b


def lloyd_max(x, L, iters=60):
    """1-D Lloyd-Max quantizer."""
    q = torch.quantile(x, torch.linspace(0, 1, L + 2, device=x.device)[1:-1])
    c = q.clone()
    for _ in range(iters):
        bnd = 0.5 * (c[1:] + c[:-1])
        a = torch.bucketize(x, bnd)
        s = torch.zeros(L, device=x.device).index_add_(0, a, x)
        n = torch.zeros(L, device=x.device).index_add_(
            0, a, torch.ones_like(x))
        keep = n > 0
        c[keep] = s[keep] / n[keep]
        c, _ = torch.sort(c)
    return c


def fit_transform_cb(R, total=TOTAL_BITS, max_bits=8, center=True):
    N = R.shape[0]
    mu = R.mean(0) if center else torch.zeros(256, device=DEV)
    Rc = R - mu
    cov = (Rc.t() @ Rc) / N
    ev, U = torch.linalg.eigh(cov.double())
    ev, U = ev.flip(0).float(), U.flip(1).float()
    P = Rc @ U
    b = allocate(ev.cpu().numpy(), total, max_bits)
    dims = np.nonzero(b)[0]
    Ls = [int(2 ** b[d]) for d in dims]
    assert int(np.prod(Ls)) == 2 ** total, (Ls, np.prod(Ls))
    cs = [lloyd_max(P[:, int(d)].contiguous(), L) for d, L in zip(dims, Ls)]
    dig = np.stack(np.unravel_index(np.arange(K), tuple(Ls)), 1)
    Cp = torch.zeros(K, 256, device=DEV)
    for j, d in enumerate(dims):
        Cp[:, int(d)] = cs[j][t(dig[:, j]).long()]
    return (Cp @ U.t() + mu).contiguous(), b[dims]
