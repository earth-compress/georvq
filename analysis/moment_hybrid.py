"""Nonlinear stage first, linear transform coding on the residual.

`moment_rd.py` found a crossover: a bottleneck autoencoder beats the
linear-Gaussian bound by ~2.6x at 32 bits, matches it near 150-200, and
LOSES above -- because a hard bottleneck discards dimensions that a
transform coder would keep and code cheaply. Neither architecture is right
across the range.

Ali's proposal, implemented here: spend a small budget on the nonlinear
stage to fold the manifold flat, then PCA + entropy-coded scalar
quantization on what is left, which should be closer to a Gaussian with a
decaying spectrum -- the regime where transform coding is optimal.

Everything is measured against two honest comparators, both ACHIEVABLE
coders rather than bounds:
  * LINEAR ONLY -- the same transform coder applied straight to the moment
    vector, no autoencoder;
  * AE ONLY -- the pure-bottleneck curve from moment_rd.
so "the hybrid wins" has to mean it beats both at matched total rate.

The stage-2 rate is a real rate, not an entropy estimate on the data being
coded: the per-axis symbol distribution is fitted on TRAIN and the rate
charged is the CROSS-entropy of the val symbols under it, which is what an
ideal entropy coder built from train statistics would actually spend.
Val indices are clamped into the train alphabet and reconstruction uses
the clamped value, so the error accounts for it too.

The premise -- that the residual is whiter and more Gaussian -- is tested,
not assumed: if it were truly white, PCA on it would buy nothing.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moment_rd import train_one                          # noqa: E402
from patch_moments_pca import load_or_collect            # noqa: E402
from pixel_rd import RVQQuant                            # noqa: E402


@torch.no_grad()
def reconstruct(m, X, mu, sd, chunk=8192):
    out = []
    for i in range(0, X.shape[0], chunk):
        out.append(m((X[i:i + chunk] - mu) / sd)[0])
    return torch.cat(out) * sd + mu


def spectrum_stats(R, name):
    """Is this actually whiter / more Gaussian than what it came from?"""
    Rc = R - R.mean(0)
    C = torch.cov(Rc.T)
    lam = torch.linalg.eigvalsh(C).flip(0).clamp_min(1e-12)
    pr = float(lam.sum() ** 2 / lam.pow(2).sum())          # participation ratio
    cum = torch.cumsum(lam, 0) / lam.sum()
    d95 = int((cum < 0.95).sum()) + 1
    # excess kurtosis along principal axes: 0 is Gaussian, >0 heavy-tailed.
    # Transform coding's optimality argument assumes near-Gaussian marginals.
    V = torch.linalg.eigh(C)[1].flip(1)
    z = (Rc @ V)[:, :64]
    z = z / z.std(0).clamp_min(1e-12)
    kurt = float((z.pow(4).mean(0) - 3.0).mean())
    print(f"  {name:<22} PR {pr:>6.1f}/{R.shape[1]}   d95 {d95:>4}   "
          f"lam1/lam_mean {float(lam[0]/lam.mean()):>7.1f}   "
          f"mean excess kurtosis {kurt:>7.2f}")
    return lam


def transform_code(Rt, Rv, deltas, energy_ref):
    """PCA + uniform scalar quantization + entropy coding, fitted on Rt.

    Yields (rate_bits_per_window, reconstruction of Rv) for each step size.
    A single uniform step across all axes is the right structure under
    entropy coding: axes whose spread is below the step collapse to one
    symbol and cost ~0 bits automatically, so no explicit bit-allocation
    search is needed.
    """
    mu = Rt.mean(0)
    C = torch.cov((Rt - mu).T)
    V = torch.linalg.eigh(C)[1].flip(1)
    Zt, Zv = (Rt - mu) @ V, (Rv - mu) @ V
    for d in deltas:
        it = torch.round(Zt / d).long()
        iv = torch.round(Zv / d).long()
        rate = 0.0
        rec = torch.empty_like(Zv)
        for a in range(Zt.shape[1]):
            lo = int(it[:, a].min())
            hi = int(it[:, a].max())
            n = hi - lo + 1
            if n <= 1:                       # axis collapsed: 0 bits
                rec[:, a] = float(lo) * d
                continue
            # Laplace-smoothed train histogram = the coder's model
            h = torch.bincount(it[:, a] - lo, minlength=n).double() + 1.0
            p = h / h.sum()
            sym = iv[:, a].clamp(lo, hi) - lo     # clamp = real coder behaviour
            rate += float(-torch.log2(p[sym]).sum()) / Zv.shape[0]
            rec[:, a] = (sym + lo).to(rec.dtype) * d
        yield rate, rec @ V.T + mu


def report(tag, Xv, Xh, bits, nb, rows):
    num = (Xh - Xv).pow(2).sum(1).sqrt()
    all_rel = float((num / Xv.pow(2).sum(1).sqrt()).mean())
    a, b = Xh[:, :nb], Xv[:, :nb]
    m_rel = float(((a - b).pow(2).sum(1).sqrt()
                   / b.pow(2).sum(1).sqrt()).mean())
    m_cos = float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())
    a, b = Xh[:, nb:], Xv[:, nb:]
    s_rel = float(((a - b).pow(2).sum(1).sqrt()
                   / b.pow(2).sum(1).sqrt()).mean())
    print(f"{tag:>14} {bits:>8.1f} {bits/4096:>8.4f} {m_rel:>9.4f} "
          f"{m_cos:>9.5f} {s_rel:>8.4f} {all_rel:>8.4f}", flush=True)
    rows.append(dict(arm=tag, bits=bits, mean_rel=m_rel, mean_cos=m_cos,
                     std_rel=s_rel, all_rel=all_rel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--device", default="cuda",
                    help="torch device; cuda is the card Slurm allocated")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--bits", type=int, default=32)
    ap.add_argument("--codes", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--deltas",
                    default="4,2,1,0.5,0.25,0.125,0.0625,0.03125")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.set_grad_enabled(False)
    M, S, Mv, Sv = load_or_collect("", args.cache, 0, 0, 64, 0.9, 8)
    X = torch.from_numpy(np.concatenate([M, S], 1)).float().to(dev)
    Xv = torch.from_numpy(np.concatenate([Mv, Sv], 1)).float().to(dev)
    nb = M.shape[1]
    print(f"train {X.shape[0]:,}  val {Xv.shape[0]:,}  d={X.shape[1]}")

    # ---- stage 1: the nonlinear fold ----
    nl = args.bits // int(np.log2(args.codes))
    q = RVQQuant(nl, args.codes, args.dim).to(dev)
    torch.set_grad_enabled(True)
    r1, m, mu, sd = train_one(X, Xv, args.dim, q, args.steps, args.lr,
                              args.batch, dev, return_model=True)
    torch.set_grad_enabled(False)
    m.eval()
    print(f"\nstage 1: dim {args.dim}, {args.bits} bits -> "
          f"val all_rel {r1['all_rel']:.4f}, mean cos {r1['mean_cos']:.5f}")

    Rt = X - reconstruct(m, X, mu, sd)
    Rv = Xv - reconstruct(m, Xv, mu, sd)

    # ---- is the premise true? ----
    print("\nspectrum / Gaussianity (the claim being tested):")
    spectrum_stats(X, "moment vector")
    spectrum_stats(Rt, "stage-1 residual")
    print(f"  residual energy is {100*float(Rt.pow(2).sum(1).mean() / X.pow(2).sum(1).mean()):.1f}% "
          f"of the input's")

    deltas = [float(v) for v in args.deltas.split(",")]
    rows = []
    print(f"\n{'arm':>14} {'bits':>8} {'b/px':>8} {'mean rel':>9} "
          f"{'mean cos':>9} {'std rel':>8} {'all rel':>8}")
    report("stage1 only", Xv, Xv - Rv, float(args.bits), nb, rows)
    for rate, rec in transform_code(Rt, Rv, deltas, X):
        report("hybrid", Xv, (Xv - Rv) + rec, args.bits + rate, nb, rows)
    # the comparator: identical transform coder, no autoencoder at all
    for rate, rec in transform_code(X, Xv, deltas, X):
        report("linear only", Xv, rec, rate, nb, rows)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
