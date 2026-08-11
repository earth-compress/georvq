"""Product quantizers materialised as explicit (4096, 256) codebooks.

If the subspaces are orthogonal and the code is the full Cartesian product,
plain nearest-neighbour in R^256 factorises exactly into independent
nearest-neighbour searches per subspace, so an explicit expansion behaves
identically to a real PQ under the fixed evaluator.

Why PQ rather than VQ for the refinement levels: after level 0 the residual
is a ~8-norm vector whose energy is spread over many dimensions.  A 4096-entry
VQ in R^256 can only place 4096 points in that cloud (it buys ~nothing at
0.047 bits/dim), whereas a PQ spends its 12 bits as m independent chunks and
scales with dimension.
"""
import numpy as np
import torch

from mylib import DEV, assign, kmeans, t

K = 4096


def balanced_subspaces(var, m):
    """Greedy bin-packing of dims into m groups of equal total variance."""
    order = np.argsort(-var)
    tot = np.zeros(m)
    sub = [[] for _ in range(m)]
    for d in order:
        j = int(np.argmin(tot))
        sub[j].append(int(d))
        tot[j] += var[d]
    return [sorted(s) for s in sub], tot


def fit_pq(R, m, L, iters=25, seed=0, rotate=True):
    """R: (N,256) residuals.  Returns explicit (4096,256) codebook."""
    assert int(round(L ** m)) == K
    N = R.shape[0]
    if rotate:
        mu = R.mean(0)
        Rc = R - mu
        cov = (Rc.t() @ Rc) / N
        _, U = torch.linalg.eigh(cov.double())
        U = U.flip(1).float()
    else:
        mu = torch.zeros(256, device=DEV)
        U = torch.eye(256, device=DEV)
    P = (R - mu) @ U
    var = P.pow(2).mean(0).cpu().numpy()
    subs, _ = balanced_subspaces(var, m)
    cbs = []
    for j, dims in enumerate(subs):
        idx = torch.tensor(dims, device=DEV)
        cbs.append(kmeans(P[:, idx].contiguous(), k=L, iters=iters,
                          seed=seed * 17 + j))
    dig = np.stack(np.unravel_index(np.arange(K), (L,) * m), 1)
    Cp = torch.zeros(K, 256, device=DEV)
    for j, dims in enumerate(subs):
        idx = torch.tensor(dims, device=DEV)
        Cp[:, idx] = cbs[j][t(dig[:, j]).long()]
    return (Cp @ U.t() + mu).contiguous()


def fit_mixed_pq(R, Ls, iters=25, seed=0):
    """Mixed-radix PQ: subspace j gets Ls[j] codewords (prod = 4096).
    High-variance subspaces get more codewords."""
    m = len(Ls)
    N = R.shape[0]
    mu = R.mean(0)
    Rc = R - mu
    cov = (Rc.t() @ Rc) / N
    _, U = torch.linalg.eigh(cov.double())
    U = U.flip(1).float()
    P = (R - mu) @ U
    var = P.pow(2).mean(0).cpu().numpy()
    # allocate dims so that subspace variance is proportional to log2(L)
    w = np.array([np.log2(L) for L in Ls], dtype=np.float64)
    w = w / w.sum()
    order = np.argsort(-var)
    tot = np.zeros(m)
    sub = [[] for _ in range(m)]
    for d in order:
        j = int(np.argmin(tot / w))
        sub[j].append(int(d))
        tot[j] += var[d]
    cbs = [kmeans(P[:, torch.tensor(sub[j], device=DEV)].contiguous(),
                  k=Ls[j], iters=iters, seed=seed * 17 + j) for j in range(m)]
    dig = np.stack(np.unravel_index(np.arange(K), tuple(Ls)), 1)
    Cp = torch.zeros(K, 256, device=DEV)
    for j in range(m):
        idx = torch.tensor(sub[j], device=DEV)
        Cp[:, idx] = cbs[j][t(dig[:, j]).long()]
    return (Cp @ U.t() + mu).contiguous()
