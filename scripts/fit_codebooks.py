"""Fit RVQ codebooks corpus-wide from the cached latent store.

With a frozen encoder, EMA codebook training needs no decoder and no
gradients — codebook geometry depends only on latents and assignments.
This fits each tier's residual stack by GPU Lloyd iterations over
millions of cached vectors (vs the 32k reservoir of the in-loop k-means),
writes the books into a fresh model built from --config, and saves a
warm-start checkpoint (stage-1 trunk + fitted books, EMA state consistent:
ema_sum = centroid * count). Empty clusters are reseeded at the points
with the largest quantization error (max-distortion, fps-flavoured).

Usage:
  python scripts/fit_codebooks.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.repro import save_checkpoint
from georvq.train.warmstart import load_partial, mature_codebooks


def load_vectors(mm, rows, scale, dev, chunk=200_000):
    """Sorted-row memmap gather -> (n_rows*cells, dim) bf16 on GPU."""
    rows = np.sort(rows)
    parts = []
    for i in range(0, len(rows), chunk):
        a = mm[rows[i:i + chunk]]                    # (c, cells, dim) int8
        t = torch.from_numpy(a.astype(np.float32)).to(dev)
        parts.append((t * scale).to(torch.bfloat16).flatten(0, 1))
    return torch.cat(parts)


@torch.no_grad()
def assign(v, C, chunk=1 << 18):
    """Nearest-centroid ids + squared error. Scores in bf16 (tensor-core
    fp32 accumulation; ~0.5% relative resolution only risks near-tie flips,
    harmless for Lloyd) — fp32 scores at K=16k would need tens of GB."""
    Cb = C.to(torch.bfloat16).t().contiguous()
    c2 = (C.float() ** 2).sum(1).to(torch.bfloat16)
    ids = torch.empty(len(v), dtype=torch.long, device=v.device)
    err = torch.empty(len(v), dtype=torch.float32, device=v.device)
    for i in range(0, len(v), chunk):
        d = c2[None] - 2 * (v[i:i + chunk] @ Cb)
        e, a = d.min(1)
        ids[i:i + chunk] = a
        err[i:i + chunk] = (e.float()
                            + (v[i:i + chunk].float() ** 2).sum(1))
    return ids, err.clamp_min_(0)


@torch.no_grad()
def lloyd(v, K, iters, dev, seed=0):
    """K-means on (N, dim) bf16 vectors. Returns (centroids fp32, counts)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = v[torch.randperm(len(v), generator=g)[:K].to(dev)].float()
    C += 1e-4 * torch.randn_like(C)
    for it in range(iters):
        ids, err = assign(v, C)
        sums = torch.zeros(K, v.shape[1], device=dev)
        cnts = torch.zeros(K, device=dev)
        for j in range(0, len(v), 1 << 22):     # chunked: v.float() whole
            sl = slice(j, j + (1 << 22))        # would be tens of GiB
            sums.index_add_(0, ids[sl], v[sl].float())
            cnts.index_add_(0, ids[sl],
                            torch.ones(len(ids[sl]), device=dev))
        dead = cnts == 0
        if dead.any():                     # reseed at max-distortion points
            k = int(dead.sum())
            worst = err.topk(k).indices
            sums[dead] = v[worst].float()
            cnts[dead] = 1.0
        C = sums / cnts[:, None]
    ids, err = assign(v, C)
    cnts = torch.zeros(K, device=dev).index_add_(
        0, ids, torch.ones(len(ids), device=dev))
    return C, cnts, ids, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dem128_live_frozen.yaml")
    ap.add_argument("--lat-store", default="/data/georvq-stores/copdem_lat128")
    ap.add_argument("--out", default="checkpoints/dem128_books_corpus/model.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--max-vectors", type=int, default=60_000_000)
    ap.add_argument("--min-valid", type=float, default=0.05)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    cfg = load_config(args.config)
    model = build_model(cfg, dev)
    if cfg.train.init_from:
        load_partial(model, cfg.train.init_from, dev)

    man = json.load(open(os.path.join(args.lat_store, "manifest.json")))
    idx = pd.read_parquet(os.path.join(args.lat_store, "index.parquet"))
    good = np.flatnonzero(idx.valid_frac.to_numpy() >= args.min_valid)
    print(f"{len(good)}/{len(idx)} windows usable "
          f"(valid_frac >= {args.min_valid})", flush=True)
    rng = np.random.default_rng(0)

    report = []
    for i, spec in enumerate(model.specs):
        mm = np.load(os.path.join(args.lat_store, f"lat_t{i}.i8.npy"),
                     mmap_mode="r")
        cells = spec.grid ** 2
        n_win = min(len(good), args.max_vectors // cells)
        rows = rng.choice(good, n_win, replace=False)
        t0 = time.time()
        v = load_vectors(mm, rows, man["scales"][i], dev)
        print(f"tier{i}: {len(v):,} vectors ({time.time()-t0:.0f}s load)",
              flush=True)
        e0 = float(sum((v[j:j + (1 << 22)].float() ** 2).sum()
                       for j in range(0, len(v), 1 << 22))) / v.numel()
        for lev_i, lev in enumerate(model.rvqs[i].levels):
            K = lev.codebook.weight.shape[0]
            C, cnts, ids, err = lloyd(v, K, args.iters, dev,
                                      seed=100 * i + lev_i)
            lev.codebook.weight.data.copy_(C)
            cn = (cnts / cnts.sum() * K).clamp_min(1e-4)   # mean count = 1
            lev.ema_count.copy_(cn)
            lev.ema_sum.copy_(C * cn[:, None])
            lev.initialized.fill_(True)
            alive = int((cnts > 0.01 * cnts.mean()).sum())
            p = cnts.clamp_min(1e-9); p = p / p.sum()
            H = float(-(p * p.log2()).sum())
            res_e = float(err.mean()) / (e0 * v.shape[1])
            report.append((i, lev_i, K, alive, H, res_e))
            print(f"  L{lev_i}: alive {alive}/{K} ({alive/K*100:.1f}%) "
                  f"H={H:.2f}/{np.log2(K):.0f} bits  "
                  f"residual_energy={res_e*100:.2f}%", flush=True)
            for j in range(0, len(v), 1 << 22):            # residual chain,
                sl = slice(j, j + (1 << 22))               # chunked in place
                v[sl] = (v[sl].float() - C[ids[sl]]).to(torch.bfloat16)
        del v
        torch.cuda.empty_cache()

    mature_codebooks(model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_checkpoint(args.out, model, cfg, 0,
                    {"fit": "corpus lloyd", "iters": args.iters,
                     "lat_store": args.lat_store,
                     "report": [list(r) for r in report]})
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
