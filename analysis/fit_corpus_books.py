"""Fit the three stage-2 codebooks at corpus scale from the int8 latent store.

Why this exists rather than `kmeans_init_rvq`: that helper hard-caps its
fit set at 1e6 vectors, which is 61 samples per centroid at K=16384 --
far too few to place a 384-d book. The store lets us fit on ~16M vectors
per tier drawn uniformly across every shard (i.e. across the whole
geographic corpus) instead of from whatever the pool happened to hold.

Protocol, following the measured lessons in docs/FINDINGS.md:
  * fit/eval split is by CHIP (the store's own val flag), never by cell --
    adjacent cells inside a window are near-duplicates and a cell-level
    split makes any fit look ~500x better than it is;
  * codebooks are quantized to int4 IMMEDIATELY after each level is fit,
    so level l+1 fits the residual the decoder will actually see. Fitting
    in fp32 and quantizing at the end is the non-embedded mistake;
  * the report is per-level residual energy and perplexity on the HELD-OUT
    chips, not on the fit set.

Output is a checkpoint that merges the stage-1 weights with the fitted
`rvqs.*` buffers, so stage-2 picks it up through the ordinary
`train.init_from` path with no new loader.

  python analysis/fit_corpus_books.py --gpu 0 --out /path/books.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

STORE = "/data/tessera-samples/latents_w32hd_full"
CKPT = "/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt"


# ---------------------------------------------------------------- sampling
def shard_ids(store):
    ids = sorted(int(f[3:7]) for f in os.listdir(store)
                 if f.startswith("t0_") and f.endswith(".npy"))
    return ids


def draw(store, ids, tier, n_target, want_val, rng, win_per_shard=None):
    """Uniform sample of cell vectors across every shard.

    Reads whole window rows via mmap (a row is contiguous, so this touches
    only the pages it needs) and keeps a random subset of the cells in
    each, which decorrelates the sample without the I/O of a full pass.
    """
    per_shard = max(1, n_target // max(len(ids), 1))
    out, n_seen = [], 0
    for sh in ids:
        meta = np.load(f"{store}/meta_{sh:04d}.npy")          # (chips, 2+256)
        is_val, ok = meta[:, 1], meta[:, 2:]                   # (chips, 256)
        keep = (ok == 1) & ((is_val == 1)[:, None] == want_val)
        wid = np.flatnonzero(keep.reshape(-1))                 # window rows
        if wid.size == 0:
            continue
        z = np.load(f"{store}/t{tier}_{sh:04d}.npy", mmap_mode="r")
        cells = z.shape[1]
        if win_per_shard is None:
            nw = int(np.ceil(per_shard / cells))
        else:
            nw = win_per_shard
        nw = min(nw, wid.size)
        sel = np.sort(rng.choice(wid, nw, replace=False))
        blk = np.asarray(z[sel])                               # (nw, cells, D)
        if cells > 1:
            take = max(1, min(cells, int(np.ceil(per_shard / nw))))
            ci = rng.choice(cells, take, replace=False)
            blk = blk[:, ci]
        out.append(blk.reshape(-1, blk.shape[-1]))
        n_seen += out[-1].shape[0]
        if n_seen >= n_target:
            break
    x = np.concatenate(out)
    if x.shape[0] > n_target:
        x = x[rng.choice(x.shape[0], n_target, replace=False)]
    return x


# ---------------------------------------------------- int4 codebook packing
def nf4_levels(x, n_lev=16, iters=40):
    """Shared Lloyd-Max level set on the unit-scaled codeword population."""
    q = torch.quantile(x.float().flatten()[:1_000_000],
                       torch.linspace(0.5 / n_lev, 1 - 0.5 / n_lev, n_lev,
                                      device=x.device))
    for _ in range(iters):
        edges = (q[1:] + q[:-1]) / 2
        a = torch.bucketize(x.flatten(), edges)
        s = torch.zeros_like(q).index_add_(0, a, x.flatten())
        c = torch.zeros_like(q).index_add_(0, a, torch.ones_like(x.flatten()))
        live = c > 0
        q[live] = s[live] / c[live]
    return q


def quantize_book(C, bits=4):
    """Per-codeword absmax scale + a shared Lloyd-Max level set.

    Returns the DEQUANTIZED book: bit-exact to what a packed int4 decoder
    would reconstruct, so downstream numbers are honest."""
    if not bits:
        return C
    s = C.abs().amax(1, keepdim=True).clamp_min(1e-8)
    u = C / s
    lev = nf4_levels(u, 1 << bits)
    edges = (lev[1:] + lev[:-1]) / 2
    idx = torch.bucketize(u, edges)
    return lev[idx] * s


# ------------------------------------------------------------------- k-means
def assign(x, C, chunk=1 << 15):
    cn = C.pow(2).sum(1)
    out = []
    for xc in x.split(chunk):
        out.append((xc.pow(2).sum(1, keepdim=True) - 2 * xc @ C.t()
                    + cn[None]).argmin(1))
    return torch.cat(out)


def sub_assigned_(r, C, a, chunk=1 << 20):
    """In-place r -= C[a], chunked.

    `r = r - C[a]` would hold three full-size tensors at once (the gather,
    the result, and the original). At 16M x 384 fp32 that is 74 GiB and
    OOMs an 80 GiB card on the finer tiers."""
    for i in range(0, r.shape[0], chunk):
        r[i:i + chunk] -= C[a[i:i + chunk]]


def fit_rvq(fit, ev, K, levels, iters, bits, dev, tag):
    """Sequential residual k-means; each level is quantized before the next
    level sees its residual."""
    C_all, rows = [], []
    r_fit, r_ev = fit, ev
    e0 = float(r_ev.pow(2).sum(1).mean())
    N = r_fit.shape[0]
    ones = torch.ones(N, device=dev)
    for l in range(levels):
        C = r_fit[torch.randperm(N, device=dev)[:K]].clone()
        for it in range(iters):
            a = assign(r_fit, C)
            s = torch.zeros_like(C).index_add_(0, a, r_fit)
            c = torch.zeros(K, device=dev).index_add_(0, a, ones)
            live = c > 0
            C[live] = s[live] / c[live, None]
            nd = int((~live).sum())
            if nd:                      # reseed dead codes from the data
                C[~live] = r_fit[torch.randint(0, N, (nd,), device=dev)] \
                    + 1e-4 * torch.randn(nd, C.shape[1], device=dev)
        C = quantize_book(C, bits)
        a_f = assign(r_fit, C)
        a_e = assign(r_ev, C)
        sub_assigned_(r_fit, C, a_f)
        sub_assigned_(r_ev, C, a_e)
        cnt = torch.bincount(a_e, minlength=K).float()
        p = cnt / cnt.sum()
        perp = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
        rows.append((l, perp, 100 * perp / K,
                     float(r_ev.pow(2).sum(1).mean()) / e0))
        C_all.append(C)
        print(f"  [{tag}] lvl {l}: perp {perp:8.0f} ({100*perp/K:5.1f}% of K) "
              f"resid {rows[-1][3]:.4f}", flush=True)
    return C_all, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default="/home/ali/projects/ring-space/"
                                     "checkpoints/tessera_w32hd_books.pt")
    ap.add_argument("--n-fit", type=int, default=16_000_000)
    ap.add_argument("--n-eval", type=int, default=1_000_000)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--bits", type=int, default=4,
                    help="codebook bit depth (0 = keep fp32)")
    ap.add_argument("--tiers", default="", help="comma list, blank = all")
    ap.add_argument("--dry-run", action="store_true",
                    help="fit only, skip writing the merged checkpoint")
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    man = json.load(open(f"{args.store}/manifest.json"))
    scale = man["scale"]
    specs = man["tiers"]
    Ks = [int(s.split(":")[2]) for s in specs]
    Ls = [int(s.split(":")[3]) for s in specs]
    ids = shard_ids(args.store)
    print(f"store: {len(ids)} shards, tiers {specs}, scale {scale:.6f}",
          flush=True)
    rng = np.random.default_rng(0)

    want = {int(x) for x in args.tiers.split(",")} if args.tiers else None
    books, report = {}, {}
    for t, (K, L) in enumerate(zip(Ks, Ls)):
        if want is not None and t not in want:
            continue
        t0 = time.time()
        fit = torch.from_numpy(draw(args.store, ids, t, args.n_fit, 0, rng)) \
            .to(dev).float().mul_(scale)
        ev = torch.from_numpy(draw(args.store, ids, t, args.n_eval, 1, rng)) \
            .to(dev).float().mul_(scale)
        print(f"tier {t}: K={K} L={L} fit {fit.shape[0]:,} "
              f"({fit.shape[0]/K:.0f}/centroid) eval {ev.shape[0]:,} "
              f"[{time.time()-t0:.0f}s io]", flush=True)
        C_all, rows = fit_rvq(fit, ev, K, L, args.iters, args.bits,
                              dev, f"t{t}")
        books[t] = [c.cpu() for c in C_all]
        report[f"tier{t}"] = rows
        del fit, ev
        torch.cuda.empty_cache()
        print(f"tier {t} done in {(time.time()-t0)/60:.1f}m", flush=True)

    if args.dry_run:
        print("dry run: no checkpoint written", flush=True)
        return
    # merge into a stage-1 checkpoint so train.init_from picks both up
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    for t, C_all in books.items():
        for l, C in enumerate(C_all):
            pre = f"rvqs.{t}.levels.{l}"
            sd[f"{pre}.codebook.weight"] = C
            sd[f"{pre}.ema_sum"] = C.clone()
            sd[f"{pre}.ema_count"] = torch.ones(C.shape[0])
            sd[f"{pre}.initialized"] = torch.tensor(True)
            sd[f"{pre}.step"] = torch.tensor(0)
    ck["model"] = sd
    ck["book_fit"] = {"n_fit": args.n_fit, "iters": args.iters,
                      "bits": args.bits, "report": report,
                      "store": args.store, "shards": len(ids)}
    torch.save(ck, args.out)
    mb = sum(c.numel() for cs in books.values() for c in cs) \
        * (args.bits or 32) / 8 / 2**20
    print(f"\nwrote {args.out}\n"
          f"codebook payload at {args.bits or 32} bits: {mb:.1f} MiB",
          flush=True)


if __name__ == "__main__":
    main()
