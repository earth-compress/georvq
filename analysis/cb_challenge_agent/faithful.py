"""Evaluator-faithful assignment.

challenge_eval computes  d = ||x||^2 - 2 x.c + ||c||^2  in float32.  The
||x||^2 term (~1030 here) sets the float32 rounding grid at ~1.2e-4, so a
codebook whose codewords are separated by much less than ~0.03 in the data
space is assigned essentially at random.  Any local sweep must reproduce
that, so this module mirrors the evaluator exactly.
"""
import numpy as np
import torch

K = 4096


def eval_assign(X, C, bs=16384):
    cn = C.pow(2).sum(1)
    out = []
    for i in range(0, X.shape[0], bs):
        ch = X[i:i + bs]
        d = ch.pow(2).sum(1, keepdim=True) - 2 * ch @ C.t() + cn[None]
        out.append(d.argmin(1))
    return torch.cat(out)


def H(a, k=K):
    c = torch.bincount(a, minlength=k).double()
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def eval_stack(X, books):
    r = X.clone()
    hs, used = [], []
    for C in books:
        a = eval_assign(r, C)
        hs.append(H(a))
        used.append(int((torch.bincount(a, minlength=K) > 0).sum()))
        r = r - C[a]
    resid = float(r.pow(2).sum(1).mean() / X.pow(2).sum(1).mean())
    return hs, used, resid, sum(2.0 ** h for h in hs)
