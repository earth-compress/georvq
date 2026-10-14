"""Is the moment sample big enough for the numbers we are quoting?

Two distinct worries, and they have different answers:

  THE BOUND.  The Gaussian rate-distortion comparator is a functional of a
  256x256 covariance. Sample covariances are biased in a specific way --
  the top eigenvalues are inflated and the bottom ones shrunk, by roughly
  sqrt(D/N) (Marchenko-Pastur) -- and reverse water-filling reads the
  SMALL eigenvalues, which are the badly-estimated ones. A shrunk tail
  makes the source look more compressible than it is, so the bound comes
  out too CHEAP and our "vs bound" ratios too flattering.

  THE EFFECTIVE N.  We have 252k windows but only 4k chips, 63 windows
  each, and windows from one chip see the same scene. The relevant N for
  estimator convergence is nearer the chip count than the window count.

The test separates the two by subsampling along each axis independently:
  * `windows`  -- fewer windows, drawn at random (mixes both axes);
  * `stride`   -- every s-th window, which at s=63 is ~one window per
                  chip: far fewer samples but nearly independent ones.
If the bound has converged, both should agree with the full-sample value.
A gap between `stride` at N and `windows` at the same N is the price of
intra-chip correlation, i.e. a direct read on effective sample size.

Also reports a CHIP-CLUSTERED bootstrap standard error on a plain
held-out statistic, which says how many digits of any val metric are real.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def gauss_bits(A, rel_targets):
    """Reverse water-filling; A is (N, D) double, centred internally."""
    lam = torch.linalg.eigvalsh(torch.cov((A - A.mean(0)).T)).clamp_min(1e-12)
    energy = float(A.pow(2).sum(1).mean())
    out = {}
    for rel in rel_targets:
        target = (rel ** 2) * energy
        if target >= float(lam.sum()):
            out[rel] = 0.0
            continue
        lo, hi = 1e-14, float(lam.max())
        for _ in range(200):
            th = 0.5 * (lo + hi)
            if float(torch.clamp(lam, max=th).sum()) > target:
                hi = th
            else:
                lo = th
        th = 0.5 * (lo + hi)
        out[rel] = 0.5 * float(torch.log2(lam[lam > th] / th).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--device", default="cuda",
                    help="torch device; cuda is the card Slurm allocated")
    ap.add_argument("--per-chip", type=int, default=63,
                    help="windows per chip in the cache (for the stride arm)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    z = np.load(args.cache)
    X = torch.from_numpy(np.concatenate([z["M"], z["S"]], 1)).double().to(dev)
    Xv = torch.from_numpy(np.concatenate([z["Mv"], z["Sv"]], 1)) \
        .double().to(dev)
    N, D = X.shape
    print(f"train {N:,} x {D}   val {Xv.shape[0]:,}")
    print(f"Marchenko-Pastur edge sqrt(D/N) = {np.sqrt(D/N):.4f} at the "
          f"window count, {np.sqrt(D/(N/args.per_chip)):.4f} at the "
          f"chip count\n")

    rels = [0.30, 0.20, 0.10, 0.05]
    full = gauss_bits(X, rels)
    g = torch.Generator(device=dev).manual_seed(0)

    print("RANDOM WINDOW SUBSETS (mixes 'fewer chips' with 'fewer per chip')")
    print(f"{'N':>9} " + " ".join(f"{'bits@'+str(int(100*r))+'%':>12}"
                                  for r in rels))
    for n in [1000, 4000, 16000, 63000, N]:
        if n > N:
            continue
        idx = torch.randperm(N, generator=g, device=dev)[:n]
        b = gauss_bits(X[idx], rels)
        print(f"{n:>9,} " + " ".join(f"{b[r]:>12.0f}" for r in rels))

    print("\nSTRIDED SUBSETS (stride s ~ one window per s -- s=63 is "
          "~one per chip,\nso these samples are nearly independent)")
    print(f"{'stride':>7} {'N':>9} " + " ".join(
        f"{'bits@'+str(int(100*r))+'%':>12}" for r in rels))
    for s in [63, 16, 4, 1]:
        sub = X[::s]
        b = gauss_bits(sub, rels)
        print(f"{s:>7} {sub.shape[0]:>9,} " +
              " ".join(f"{b[r]:>12.0f}" for r in rels))

    # Cross-check that generalises past any subsampling scheme: fit the
    # covariance on TRAIN and score the water level on VAL. If the train
    # spectrum is over-fitted, the bits it claims will not hold up on
    # geographically disjoint data.
    muT = X.mean(0)
    lam, V = torch.linalg.eigh(torch.cov((X - muT).T))
    lam, V = lam.flip(0).clamp_min(1e-12), V.flip(1)
    lam_v = ((Xv - muT) @ V).pow(2).mean(0)          # val energy per train PC
    print("\nTRAIN-fitted principal axes, energy measured on VAL:")
    print(f"{'axis':>6} {'train lam':>12} {'val lam':>12} {'ratio':>8}")
    for i in [0, 7, 31, 63, 127, 191, 255]:
        print(f"{i:>6} {float(lam[i]):>12.5f} {float(lam_v[i]):>12.5f} "
              f"{float(lam_v[i]/lam[i]):>8.3f}")
    tail = slice(D // 2, None)
    print(f"  bottom half of the spectrum: val/train energy ratio "
          f"{float(lam_v[tail].sum()/lam[tail].sum()):.3f}  "
          f"(>1 means the train tail is SHRUNK, so the bound is too cheap)")

    # chip-clustered bootstrap: how precise is any val average?
    stat = (Xv - muT).pow(2).sum(1).sqrt() / Xv.pow(2).sum(1).sqrt()
    nchip = Xv.shape[0] // args.per_chip
    blocks = stat[:nchip * args.per_chip].view(nchip, args.per_chip)
    boots = []
    for _ in range(400):
        sel = torch.randint(0, nchip, (nchip,), generator=g, device=dev)
        boots.append(float(blocks[sel].mean()))
    bs = torch.tensor(boots)
    iid = float(stat.std() / np.sqrt(stat.numel()))
    print(f"\nval rel-L2 of the constant predictor: {float(stat.mean()):.4f}")
    print(f"  chip-clustered bootstrap SE {float(bs.std()):.5f}   "
          f"iid-assumption SE {iid:.5f}   "
          f"(inflation {float(bs.std())/iid:.1f}x)")
    print(f"  => val metrics are meaningful to about "
          f"+/-{2*float(bs.std()):.4f} (2 SE)")


if __name__ == "__main__":
    main()
