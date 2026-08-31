"""Size a pyramid from the source: how much does each scale carry, and of
what rank?

A P-tier pyramid on a W-px window is a hierarchical block-mean
decomposition. With W=64 and grids 1/4/16 the tiers convey, in order:

    tier 0   the 64px window mean          (1 vector)
    tier 1   16px block means of the residual   (16 vectors)
    tier 2   4px block means of that residual   (256 vectors)
    left     whatever remains -- the pyramid CANNOT represent it at all

Two numbers per level decide the architecture:

  ENERGY SHARE  what fraction of the source's variance that scale carries.
                Tier 0's share is exactly the question "is a transmitted
                mean worth it": if the window mean is most of the energy
                and costs a few hundred bits, sending it directly beats
                asking a learned coder to rediscover it.
  RANK          the effective dimensionality of the vectors that level
                transmits (participation ratio and d90/d95/d99). This is
                the floor for that tier's latent width: a latent narrower
                than the rank cannot carry the level, and one much wider is
                paying for dimensions that carry no variance.

The `left` row is the honest ceiling: energy below the finest grid is
unreachable no matter how many bits the pyramid spends, so it bounds the
achievable reconstruction before any model is trained.

Rank is measured on MEAN-REMOVED vectors at each level (each level's own
global mean), so a large constant offset cannot inflate the apparent
dimensionality -- the trap that made the patch-moment PCA look low-rank
until the relative-error column was added.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_probe import iter_chips                             # noqa: E402


def block_mean(x, v, g):
    """Mean over each of g x g blocks, on valid pixels only.

    x: (B, C, W, W) float, v: (B, W, W) bool -> (B, C, g, g), (B, g, g) count
    """
    B, C, W, _ = x.shape
    p = W // g
    vm = v.unsqueeze(1).to(x.dtype)
    xs = (x * vm).reshape(B, C, g, p, g, p).sum((3, 5))
    n = vm.reshape(B, 1, g, p, g, p).sum((3, 5))
    return xs / n.clamp_min(1e-6), n[:, 0]


def upsample(m, W):
    """(B, C, g, g) -> (B, C, W, W) by block replication."""
    g = m.shape[-1]
    return m.repeat_interleave(W // g, -2).repeat_interleave(W // g, -1)


def rank_stats(V):
    """Effective rank of a set of vectors, after removing their own mean."""
    if V.shape[1] == 1:            # single band: rank is 1 by construction
        return 1.0, {0.90: 1, 0.95: 1, 0.99: 1}
    Vc = V - V.mean(0)
    C = torch.cov(Vc.T.double())
    lam = torch.linalg.eigvalsh(C).flip(0).clamp_min(0)
    tot = lam.sum().clamp_min(1e-20)
    cum = torch.cumsum(lam, 0) / tot
    pr = float(tot ** 2 / lam.pow(2).sum().clamp_min(1e-30))
    d = {q: int((cum < q).sum()) + 1 for q in (0.90, 0.95, 0.99)}
    return pr, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--source", required=True,
                    help="tessera | ae | dem | s2 | s1 | era5:<var>[+std]")
    ap.add_argument("--chips", type=int, default=30)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--grids", default="1,4,16")
    ap.add_argument("--min-valid", type=float, default=0.95)
    ap.add_argument("--max-vecs", type=int, default=120000)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    W = args.window
    grids = [int(g) for g in args.grids.split(",")]

    wins, vals = [], []
    for x, valid, _ in iter_chips(args.source, args.store, args.chips):
        C, H, _ = x.shape
        for r in range(0, H - W + 1, W):
            for c in range(0, H - W + 1, W):
                v = valid[r:r + W, c:c + W]
                if v.mean() < args.min_valid:
                    continue
                wins.append(x[:, r:r + W, c:c + W])
                vals.append(v)
        if len(wins) > 4000:
            break
    X = torch.from_numpy(np.stack(wins)).float()
    V = torch.from_numpy(np.stack(vals))
    del wins, vals
    print(f"{args.store} ({args.source})")
    print(f"  {X.shape[0]} windows of {W}px, {X.shape[1]} bands, "
          f"grids {grids}\n")

    vm = V.unsqueeze(1).float()
    total = float((X * vm).pow(2).sum())          # energy on valid pixels
    resid = X.clone()
    rows = []
    for g in grids:
        m, _ = block_mean(resid, V, g)
        up = upsample(m, W) * vm
        share = float(up.pow(2).sum()) / max(total, 1e-20)
        vec = m.permute(0, 2, 3, 1).reshape(-1, X.shape[1])
        if vec.shape[0] > args.max_vecs:
            sel = torch.randperm(vec.shape[0])[:args.max_vecs]
            vec = vec[sel]
        pr, d = rank_stats(vec)
        rows.append((g, share, pr, d, vec.shape[0]))
        resid = (resid - up) * vm

    left = float((resid * vm).pow(2).sum()) / max(total, 1e-20)
    nb = X.shape[1]
    # Share of the MEAN-REMOVED residual is the decision-relevant column:
    # total energy is dominated by the DC offset for any all-positive
    # source (reflectance, dB), which makes tier 0 look trivially huge and
    # says nothing about how to size the finer tiers.
    res_tot = sum(r[1] for r in rows[1:]) + left
    print(f"{'tier':>5} {'grid':>5} {'cell px':>8} {'energy':>8} {'cum':>7} "
          f"{'of resid':>9}   {'part.ratio':>11} {'d90':>5} {'d95':>5} "
          f"{'d99':>5}")
    cum = 0.0
    for i, (g, share, pr, d, n) in enumerate(rows):
        cum += share
        rs = ("-" if i == 0 else f"{100*share/max(res_tot,1e-20):.1f}%")
        print(f"{i:>5} {g:>5} {W//g:>8} {100*share:>7.2f}% {100*cum:>6.2f}% "
              f"{rs:>9}   {pr:>7.1f}/{nb:<3} {d[0.90]:>5} {d[0.95]:>5} "
              f"{d[0.99]:>5}")
    print(f"{'left':>5} {'-':>5} {'<'+str(W//grids[-1]):>8} "
          f"{100*left:>7.2f}% {100.0:>6.2f}% "
          f"{100*left/max(res_tot,1e-20):>8.1f}%   "
          f"(unreachable by this pyramid)")

    print(f"\nREADING THIS")
    t0 = rows[0][1]
    print(f"  tier 0 carries {100*t0:.1f}% of the energy in {nb} numbers per "
          f"{W}px window.")
    print(f"  -> transmitting it costs {nb}*b bits; at 6 bits/band that is "
          f"{nb*6/W**2:.3f} b/px.")
    print(f"  latent width floor per tier (d95 of what it transmits): "
          + ", ".join(f"tier{i}>={r[3][0.95]}" for i, r in enumerate(rows)))
    print(f"  unreachable floor: {100*left:.2f}% of energy sits below "
          f"{W//grids[-1]}px and no pyramid depth recovers it.")


if __name__ == "__main__":
    main()
