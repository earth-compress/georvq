"""Tree-structured VQ: what does O(log K) encoding actually cost?

Instead of assigning by nearest centroid over all K (full search), descend
a binary tree of hyperplane cuts and take the leaf. Encoding drops from K
distance computations to 2*log2(K) -- 28 instead of 16384 -- and, if the
cuts are median splits, every leaf is populated BY CONSTRUCTION, so dead
codes and reset churn cannot occur at all.

Crucially the DECODER is unchanged: it only needs the leaf centroid the
index selects. The tree and its internal nodes are encoder-side, and
encoder size does not matter here.

The cost is that a greedy descent can land somewhere that is not the
nearest centroid. This measures that gap directly, against:

  tree-greedy   descend, take the leaf                     2*log2(K) dists
  tree-beam-B   keep the best B branches per level         2*B*log2(K)
  tree+NN       same centroids, exhaustive nearest         K
  lloyd+NN      Lloyd-refined centroids, exhaustive        K   <- baseline

so the two effects are separated: how much comes from the tree's cells
being intersections of half-spaces rather than Voronoi cells, and how
much from the centroids themselves being tree means rather than
Lloyd-optimal.

Reads the w32hd int8 latent store directly -- no encoder pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_corpus_books import draw, shard_ids            # noqa: E402

from georvq.model.vq import _chunked_assign             # noqa: E402

STORE = "/data/tessera-samples/latents_w32hd_full"


@torch.no_grad()
def build_tree(X, depth, iters=8, chunk=1 << 20):
    """Balanced median tree. Returns (mu, v, thr) per internal node level
    and the leaf means. Node n at level d splits on <x - mu_n, v_n> > thr_n.
    """
    N, D = X.shape
    dev = X.device
    node = torch.zeros(N, dtype=torch.long, device=dev)
    MUs, Vs, THs = [], [], []
    for d in range(depth):
        n_nodes = 1 << d
        cnt = torch.bincount(node, minlength=n_nodes).clamp_min(1).float()
        MU = torch.zeros(n_nodes, D, device=dev)
        for i in range(0, N, chunk):
            MU.index_add_(0, node[i:i + chunk], X[i:i + chunk])
        MU /= cnt[:, None]
        V = torch.randn(n_nodes, D, device=dev)
        V /= V.norm(dim=1, keepdim=True)
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
        MUs.append(MU); Vs.append(V); THs.append(thr)
        node = node * 2 + (p > thr[node]).long()
        del p, order, ps
    K = 1 << depth
    cnt = torch.bincount(node, minlength=K).clamp_min(1).float()
    C = torch.zeros(K, X.shape[1], device=dev)
    for i in range(0, N, chunk):
        C.index_add_(0, node[i:i + chunk], X[i:i + chunk])
    C /= cnt[:, None]
    return (MUs, Vs, THs), C, node


@torch.no_grad()
def tree_assign(X, tree, chunk=1 << 20):
    """Greedy descent: 1 projection per level."""
    MUs, Vs, THs = tree
    N = X.shape[0]
    node = torch.zeros(N, dtype=torch.long, device=X.device)
    for MU, V, TH in zip(MUs, Vs, THs):
        for i in range(0, N, chunk):
            nb = node[i:i + chunk]
            p = ((X[i:i + chunk] - MU[nb]) * V[nb]).sum(1)
            node[i:i + chunk] = nb * 2 + (p > TH[nb]).long()
    return node


@torch.no_grad()
def tree_beam(X, tree, C, beam=4, chunk=1 << 18):
    """Keep the best `beam` branches per level, scored by distance to the
    branch's running mean; resolve ties at the leaves by true distance."""
    MUs, Vs, THs = tree
    depth = len(MUs)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk]
        n = xb.shape[0]
        cand = torch.zeros(n, 1, dtype=torch.long, device=X.device)
        for d, (MU, V, TH) in enumerate(zip(MUs, Vs, THs)):
            B = cand.shape[1]
            flat = cand.reshape(-1)
            mu = MU[flat].reshape(n, B, -1)
            v = V[flat].reshape(n, B, -1)
            th = TH[flat].reshape(n, B)
            p = ((xb[:, None, :] - mu) * v).sum(-1) - th
            left = cand * 2
            right = cand * 2 + 1
            nxt = torch.cat([left, right], 1)
            # score: how far inside the chosen half-space (negative = left)
            score = torch.cat([p.clamp_min(0), (-p).clamp_min(0)], 1)
            k = min(beam, nxt.shape[1])
            sel = torch.topk(-score, k, dim=1).indices
            cand = torch.gather(nxt, 1, sel)
        d2 = ((xb[:, None, :] - C[cand]) ** 2).sum(-1)
        out[i:i + chunk] = torch.gather(cand, 1, d2.argmin(1, keepdim=True)) \
            .squeeze(1)
    return out


@torch.no_grad()
def greedy_tree(X, K, iters=8, min_pts=8):
    """Grow the tree by DISTORTION rather than to a fixed depth.

    Start with one cell; repeatedly split the leaf with the largest
    distortion contribution along its own principal direction, at its
    median, until there are K leaves. Leaves end at different depths --
    error-rich regions get cut many times, flat regions once -- and the
    result approaches equal distortion per cell, which the partial
    distortion theorem makes the optimality condition.

    (A balanced tree spends its cuts uniformly regardless of where the
    error is, which is why it costs +5.5% against Lloyd on the same data.)
    """
    import heapq
    N, D = X.shape
    dev = X.device
    idx_all = torch.arange(N, device=dev)
    mu = X.mean(0)
    sse = float((X - mu).pow(2).sum())
    cells = {0: (idx_all, mu)}
    heap = [(-sse, 0)]
    nxt = 1
    while len(cells) < K and heap:
        negs, cid = heapq.heappop(heap)
        idx, mu = cells.pop(cid)
        if idx.numel() < min_pts:
            cells[cid] = (idx, mu)          # too small to split; retire it
            continue
        Xi = X[idx]
        Xc = Xi - mu
        v = torch.randn(D, device=dev); v /= v.norm().clamp_min(1e-12)
        for _ in range(iters):
            v = Xc.t() @ (Xc @ v)
            n = v.norm()
            if n < 1e-12:
                break
            v = v / n
        p = Xc @ v
        thr = p.median()
        right = p > thr
        for m in (right, ~right):
            if int(m.sum()) == 0:
                continue
            sub = idx[m]
            c = X[sub].mean(0)
            e = float((X[sub] - c).pow(2).sum())
            cells[nxt] = (sub, c)
            heapq.heappush(heap, (-e, nxt))
            nxt += 1
    C = torch.stack([mu for (_, mu) in cells.values()])
    lab = torch.empty(N, dtype=torch.long, device=dev)
    for j, (idx, _) in enumerate(cells.values()):
        lab[idx] = j
    return C, lab


def dist(X, C, a):
    return float((X - C[a]).pow(2).sum(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--depth", type=int, default=14, help="log2(K)")
    ap.add_argument("--per-code", type=int, default=600)
    ap.add_argument("--eval-per-code", type=int, default=150)
    ap.add_argument("--lloyd", type=int, default=8)
    ap.add_argument("--beams", default="2,4,8,16")
    ap.add_argument("--no-greedy", action="store_true",
                    help="skip the greedy-growth arm. It gathers the\n                          root cell, transiently doubling memory, and\n                          has shown no benefit over median splitting\n                          at K=2048/4096/16384.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    K = 1 << args.depth
    ids = shard_ids(args.store)
    rng = np.random.default_rng(0)
    scale = 6.0 / 127.0
    print(f"tier {args.tier}, K={K} (depth {args.depth}), "
          f"{len(ids)} shards")
    X = torch.from_numpy(draw(args.store, ids, args.tier, args.per_code * K,
                              0, rng)).to(dev).float().mul_(scale)
    Xv = torch.from_numpy(draw(args.store, ids, args.tier,
                               args.eval_per_code * K, 1, rng)) \
        .to(dev).float().mul_(scale)
    print(f"fit {X.shape[0]:,} ({X.shape[0]/K:.0f}/code), "
          f"held-out {Xv.shape[0]:,}, dim {X.shape[1]}")

    t0 = time.time()
    tree, C, _ = build_tree(X, args.depth)
    print(f"tree built in {time.time()-t0:.0f}s")

    res = {}
    e0 = float(Xv.pow(2).sum(1).mean())

    a = tree_assign(Xv, tree)
    # ONE hyperplane projection per level (not two child-centroid
    # distances -- this is not classical two-centroid TSVQ)
    res["tree_greedy"] = (dist(Xv, C, a), args.depth)
    a = _chunked_assign(Xv, C)
    res["tree_centroids_NN"] = (dist(Xv, C, a), K)
    for b in [int(x) for x in args.beams.split(",")]:
        a = tree_beam(Xv, tree, C, beam=b)
        # b projections per level, plus b final distances to
        # resolve the surviving leaves
        res[f"tree_beam{b}"] = (dist(Xv, C, a), b * (args.depth + 1))

    if not args.no_greedy:
      t0 = time.time()
      Cg, _ = greedy_tree(X, K)
      print(f"greedy tree grown in {time.time()-t0:.0f}s -> {Cg.shape[0]} leaves")
      a = _chunked_assign(Xv, Cg)
      res["greedy_tree_NN"] = (dist(Xv, Cg, a), K)
      Cg2 = Cg.clone()
      ones_g = torch.ones(X.shape[0], device=dev)
      for _ in range(args.lloyd):
          a2 = _chunked_assign(X, Cg2)
          sums = torch.zeros_like(Cg2).index_add_(0, a2, X)
          cnt = torch.zeros(Cg2.shape[0], device=dev).index_add_(0, a2, ones_g)
          live = cnt > 0
          Cg2[live] = sums[live] / cnt[live, None]
      a = _chunked_assign(Xv, Cg2)
      res["greedy_tree+lloyd"] = (dist(Xv, Cg2, a), K)
      del Cg, Cg2
      torch.cuda.empty_cache()

    CL = C.clone()
    ones = torch.ones(X.shape[0], device=dev)
    for _ in range(args.lloyd):
        a2 = _chunked_assign(X, CL)
        sums = torch.zeros_like(CL).index_add_(0, a2, X)
        cnt = torch.zeros(K, device=dev).index_add_(0, a2, ones)
        live = cnt > 0
        CL[live] = sums[live] / cnt[live, None]
    a = _chunked_assign(Xv, CL)
    res["lloyd_NN"] = (dist(Xv, CL, a), K)

    base = res["lloyd_NN"][0]
    print(f"\n{'method':>20} {'held-out resid':>15} {'vs full search':>15} "
          f"{'D-dot-prods':>12} {'speedup':>9}")
    for k, (d, ops) in res.items():
        print(f"{k:>20} {d/e0:>15.4f} {100*(d-base)/base:>+14.2f}% "
              f"{ops:>12,} {K/ops:>8.0f}x")
    # occupancy: the structural claim
    a = tree_assign(Xv, tree)
    cnt = torch.bincount(a, minlength=K).float()
    p = cnt / cnt.sum()
    print(f"\ntree partition: {int((cnt==0).sum())} empty leaves of {K}, "
          f"perplexity {float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum())):.0f}"
          f" ({100*float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum()))/K:.1f}% of K)")
    a = _chunked_assign(Xv, CL)
    cnt = torch.bincount(a, minlength=K).float()
    p = cnt / cnt.sum()
    print(f"lloyd + NN:     {int((cnt==0).sum())} empty of {K}, "
          f"perplexity {float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum())):.0f}"
          f" ({100*float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum()))/K:.1f}% of K)")

    if args.out:
        json.dump({k: {"resid": v[0]/e0, "ops": v[1]} for k, v in res.items()},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
