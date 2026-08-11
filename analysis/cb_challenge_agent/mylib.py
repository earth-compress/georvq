"""My own utilities for the codebook-utilization challenge (written from scratch)."""
import numpy as np
import torch

DEV = "cuda:1"
K = 4096
D = 256


def load_split(n_val=86432, seed=0):
    V = np.load("/data/tessera-samples/cb_challenge/vectors_train.npy")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(V.shape[0])
    val = V[perm[:n_val]].copy()
    tr = V[perm[n_val:]].copy()
    return tr, val


def t(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEV)


def assign(X, C, bs=32768):
    """nearest-neighbour indices of rows of X in codebook C (both torch, gpu)."""
    cn = C.pow(2).sum(1)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for i in range(0, X.shape[0], bs):
        ch = X[i:i + bs]
        d = cn[None] - 2.0 * (ch @ C.t())
        out[i:i + bs] = d.argmin(1)
    return out


def entropy_bits(a, k=K):
    c = torch.bincount(a, minlength=k).double()
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def kmeans(X, k=K, iters=30, seed=0, verbose=False, init=None):
    """Lloyd's algorithm on GPU with empty-cluster splitting."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    if init is None:
        if X.shape[0] >= k:
            idx = torch.randperm(X.shape[0], generator=g, device=X.device)[:k]
            C = X[idx].clone()
        else:                       # fewer points than codewords: jitter
            idx = torch.randint(0, X.shape[0], (k,), generator=g,
                                device=X.device)
            C = X[idx].clone() + torch.randn(
                k, X.shape[1], generator=g, device=X.device) * 0.02 * X.std()
    else:
        C = init.clone()
    for it in range(iters):
        a = assign(X, C)
        cnt = torch.bincount(a, minlength=k)
        S = torch.zeros_like(C)
        S.index_add_(0, a, X)
        nz = cnt > 0
        C[nz] = S[nz] / cnt[nz].unsqueeze(1).float()
        dead = (~nz).nonzero().squeeze(1)
        if dead.numel():
            # split the largest clusters
            big = cnt.argsort(descending=True)[:dead.numel()]
            noise = torch.randn(dead.numel(), C.shape[1], generator=g,
                                device=X.device) * 0.01 * X.std()
            C[dead] = C[big] + noise
            C[big] = C[big] - noise
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"   it{it} used={int(nz.sum())} H={entropy_bits(a,k):.3f}")
    return C
