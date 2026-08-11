"""Map the entropy-vs-variance trade-off over PCA direction blocks.

Because each level uses 12 mutually orthogonal directions and the offset is
zero, quantizing block A leaves the projections onto a disjoint block B
untouched.  Levels are therefore independent and per-level entropies can be
swept one block at a time.
"""
import numpy as np
import torch

from mylib import DEV, K, assign, entropy_bits, load_split, t

M = 12


def global_pca(X):
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.t() @ Xc) / X.shape[0]
    ev, U = torch.linalg.eigh(cov.double())
    return ev.flip(0).float(), U.flip(1).float(), mu


def sign_book(U, thr, alpha):
    """U:(D,12) orthonormal, thr:(12,) medians, alpha:(12,) half-steps."""
    S = ((torch.arange(K, device=DEV)[:, None] >>
          torch.arange(M, device=DEV)[None]) & 1).float() * 2 - 1
    coef = thr[None] + S * alpha[None]
    return (coef @ U.t()).contiguous()


def make_level(X, U, idx, alpha_scale=1.0):
    Um = U[:, idx].contiguous()
    P = X @ Um
    thr = P.median(0).values
    alpha = torch.full((M,), alpha_scale, device=DEV)
    return sign_book(Um, thr, alpha), P, thr


if __name__ == "__main__":
    tr, val = load_split()
    Xt, Xv = t(tr), t(val)
    ev, U, mu = global_pca(Xt)
    tot = float(ev.sum())
    print("start  var_frac_of_block   valH   used")
    for s in range(0, 245, 8):
        idx = torch.arange(s, s + M, device=DEV)
        C, _, _ = make_level(Xt, U, idx)
        a = assign(Xv, C)
        vf = float(ev[s:s + M].sum()) / tot
        print(f"{s:4d}  {vf:.5f}  {entropy_bits(a):7.4f}  "
              f"{int((torch.bincount(a, minlength=K) > 0).sum()):5d}")
