"""Is the within-cell principal direction estimable at all?

Splitting a cell along its top principal direction assumes that direction
can be estimated from the points in the cell. In D=384 with a few hundred
points that is not obvious: for a spiked covariance the sample top
eigenvector only correlates with the truth when the spike exceeds
sqrt(D/n) (the BBP transition), and below that it is asymptotically
ORTHOGONAL to it. A converged quantizer's cells are near-isotropic by
construction, which is exactly the no-spike case.

Two tests:

  STABILITY  split a cell's points in half, estimate the direction on
             each half independently, and take |cos| between them. An
             informative estimate gives |cos| near 1; noise gives the
             value expected between two random vectors in D dims,
             ~sqrt(2/(pi*D)) = 0.036 at D=384.

  DOES IT MATTER  build the whole tree twice -- once splitting along the
             estimated principal direction, once along a RANDOM direction
             -- and compare held-out residual. If the estimate is noise,
             these are the same. If they are the same, the estimate is
             either noise or irrelevant, and either way the elaborate
             power iteration is not earning its place.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_corpus_books import draw, shard_ids            # noqa: E402

from georvq.model.vq import _chunked_assign             # noqa: E402

STORE = "/data/tessera-samples/latents_w32hd_full"


@torch.no_grad()
def top_dir(X, iters=12):
    Xc = X - X.mean(0)
    v = torch.randn(X.shape[1], device=X.device)
    v /= v.norm().clamp_min(1e-12)
    for _ in range(iters):
        v = Xc.t() @ (Xc @ v)
        n = v.norm()
        if n < 1e-12:
            return None, 0.0, 0.0
        v = v / n
    lam1 = float((Xc @ v).pow(2).mean())
    bulk = float(Xc.pow(2).sum(1).mean() / X.shape[1])   # mean eigenvalue
    return v, lam1, bulk


@torch.no_grad()
def build_tree_dir(X, depth, mode="pc", iters=8, chunk=1 << 20):
    """Balanced median tree; `mode` picks the cut direction."""
    N, D = X.shape
    dev = X.device
    node = torch.zeros(N, dtype=torch.long, device=dev)
    for _ in range(depth):
        n_nodes = int(node.max().item()) + 1
        cnt = torch.bincount(node, minlength=n_nodes).clamp_min(1).float()
        MU = torch.zeros(n_nodes, D, device=dev)
        for i in range(0, N, chunk):
            MU.index_add_(0, node[i:i + chunk], X[i:i + chunk])
        MU /= cnt[:, None]
        V = torch.randn(n_nodes, D, device=dev)
        V /= V.norm(dim=1, keepdim=True)
        if mode == "pc":
            for _ in range(iters):
                A = torch.zeros(n_nodes, D, device=dev)
                for i in range(0, N, chunk):
                    nb = node[i:i + chunk]
                    xc = X[i:i + chunk] - MU[nb]
                    A.index_add_(0, nb, xc * (xc * V[nb]).sum(1, keepdim=True))
                V = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
        p = torch.empty(N, device=dev)
        for i in range(0, N, chunk):
            nb = node[i:i + chunk]
            p[i:i + chunk] = ((X[i:i + chunk] - MU[nb]) * V[nb]).sum(1)
        order = torch.argsort(node)
        ps = p[order]
        offs = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                          torch.bincount(node, minlength=n_nodes).cumsum(0)])
        thr = torch.zeros(n_nodes, device=dev)
        for nid in range(n_nodes):
            seg = ps[offs[nid]:offs[nid + 1]]
            if seg.numel():
                thr[nid] = seg.median()
        node = node * 2 + (p > thr[node]).long()
        del p, order, ps
    K = 1 << depth
    cnt = torch.bincount(node, minlength=K).clamp_min(1).float()
    C = torch.zeros(K, D, device=dev)
    for i in range(0, N, chunk):
        C.index_add_(0, node[i:i + chunk], X[i:i + chunk])
    return C / cnt[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--depth", type=int, default=11)
    ap.add_argument("--per-code", type=int, default=400)
    ap.add_argument("--eval-per-code", type=int, default=150)
    ap.add_argument("--cells", type=int, default=200)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    K = 1 << args.depth
    ids = shard_ids(STORE)
    rng = np.random.default_rng(0)
    X = torch.from_numpy(draw(STORE, ids, args.tier, args.per_code * K, 0,
                              rng)).to(dev).float().mul_(6 / 127)
    Xv = torch.from_numpy(draw(STORE, ids, args.tier,
                               args.eval_per_code * K, 1, rng)) \
        .to(dev).float().mul_(6 / 127)
    D = X.shape[1]
    rand_baseline = math.sqrt(2 / (math.pi * D))
    print(f"tier {args.tier}: N={X.shape[0]:,}  D={D}  K={K}")
    print(f"|cos| between two independent random unit vectors in {D}D "
          f"~ {rand_baseline:.4f}\n")

    # ---- STABILITY: split-half agreement of the estimated direction ----
    C = build_tree_dir(X, args.depth, mode="pc")
    a = _chunked_assign(X, C)
    order = torch.argsort(a)
    a_s = a[order]
    bounds = torch.searchsorted(a_s, torch.arange(K + 1, device=dev))
    sizes = (bounds[1:] - bounds[:-1])
    sel = torch.argsort(sizes, descending=True)[:args.cells]
    print(f"{'cell n':>8} {'|cos| split-half':>17} {'spike lam1/bulk':>17} "
          f"{'BBP thresh sqrt(D/n)':>22} {'above?':>8}")
    buckets = {}
    for j in sel.tolist():
        idx = order[bounds[j]:bounds[j + 1]]
        n = idx.numel()
        if n < 8:
            continue
        perm = idx[torch.randperm(n, device=dev)]
        h = n // 2
        v1, l1, b1 = top_dir(X[perm[:h]])
        v2, _, _ = top_dir(X[perm[h:]])
        if v1 is None or v2 is None:
            continue
        c = abs(float(v1 @ v2))
        spike = l1 / max(b1, 1e-12)
        key = int(2 ** round(math.log2(max(n, 1))))
        buckets.setdefault(key, []).append((c, spike, n))
    for key in sorted(buckets):
        cs = [b[0] for b in buckets[key]]
        sp = [b[1] for b in buckets[key]]
        ns = [b[2] for b in buckets[key]]
        nbar = sum(ns) / len(ns)
        thr = math.sqrt(D / (nbar / 2))          # each half has n/2 points
        print(f"{nbar:>8.0f} {sum(cs)/len(cs):>17.4f} "
              f"{sum(sp)/len(sp):>17.2f} {thr:>22.2f} "
              f"{'yes' if sum(sp)/len(sp) > thr else 'NO':>8}   "
              f"(n={len(cs)} cells)")

    # ---- DOES IT MATTER: principal vs random cut direction ----
    e0 = float(Xv.pow(2).sum(1).mean())
    out = {}
    for mode in ("pc", "random"):
        Cm = build_tree_dir(X, args.depth, mode=mode)
        am = _chunked_assign(Xv, Cm)
        r = float((Xv - Cm[am]).pow(2).sum(1).mean())
        cnt = torch.bincount(am, minlength=K).float()
        p = cnt / cnt.sum()
        perp = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
        out[mode] = (r / e0, perp)
    print(f"\n{'cut direction':>16} {'held-out resid':>15} {'perplexity':>12}")
    for k, (r, pp) in out.items():
        print(f"{k:>16} {r:>15.4f} {pp:>12.0f}")
    d = 100 * (out['random'][0] - out['pc'][0]) / out['pc'][0]
    print(f"\nrandom cuts cost {d:+.2f}% versus principal-direction cuts")


if __name__ == "__main__":
    main()
