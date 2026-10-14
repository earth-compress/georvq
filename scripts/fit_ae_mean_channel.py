"""Fit the RVQ-PCA mean channel of the AE v2 release (`RVQPCAMeanChannel`).

A stats_cond model trains against the exact per-window mean; the release
codes that mean with a channel fitted afterwards on the training split's
window means. The channel depends only on the store and the split, not
on the trained model. Three stages:

1. Per-window means. Each chip goes through `AESource.prep` (dequant,
   per-pixel L2 normalisation, nodata zeroed) and each of its
   non-overlapping 64 px windows gets the masked mean over valid pixels,
   as the model's `patch_stats` computes it. Windows with no valid pixel
   are dropped. About 80 minutes on 24 CPU workers for chips_1024.
2. RVQ levels of K codes over all training means. Each level is seeded by
   distortion-driven splitting (`greedy_split_init`) and refined by Lloyd
   iterations, on what the previous levels leave.
3. A PCA coder for the RVQ residual (`RVQPCAMeanChannel.fit_residual`),
   fitted on a fixed random sample of training residuals, with the bits
   left after the RVQ indices allocated greedily across components.

The release used 2 levels of 4096 (24 bits) and 440 bits in all:

    python scripts/fit_ae_mean_channel.py --means means.npz \\
        --out mean_channel.npz

`--means` caches stage 1 (2.6 GB for chips_1024): it is read when the
file exists and written otherwise. `--books` takes RVQ books from an npz
with arrays L0, L1, ... and skips stage 2.

The released tables were fitted on a GPU, and a CPU run does not
reproduce them bit for bit. Stage 1 is exact. Stage 3 run on CPU from the
released books and means gives the same books, bit allocation and clips,
but mu0, s and V differ (V by up to 1.3e-4), because the nearest-code
search breaks near-ties differently on the two devices and so a few of
the 2M sampled residuals come from a different code. Coded with either
set of tables, 6% of validation means get different fields, the decoded
means differ by at most 0.0022 and the median relative L2 error is
0.00604 for both. Stage 2 differs further, since the scatter-adds
accumulate in another order and the random draws are not the GPU's,
but a full CPU refit from the means (65 minutes on 16 cores, 22 GB)
also reaches a median relative L2 error of 0.00604. The
tables enter the bitstream_id, so a refit is a new mean channel and the
released mean_channel.npz is the one that decodes released files.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from georvq.config import load_config
from georvq.data.store import open_store
from georvq.inference.mean_codec import RVQPCAMeanChannel
from georvq.model.vq import _chunked_assign

W = 64
_ctx = {}


# ---- stage 1: per-window means ----

def _init(cfg_path):
    torch.set_num_threads(1)
    from georvq.sources import get_source
    cfg = load_config(cfg_path, [])
    _ctx["store"] = open_store(cfg.data.store_dir)
    _ctx["src"] = get_source("ae", cfg.model)


def _chip(job):
    shard, slot, _ = job
    s, src = _ctx["store"], _ctx["src"]
    a = np.array(s.open_shard(shard)[slot])
    x, v = src.prep(torch.from_numpy(a)[None], "cpu")
    g = a.shape[-1] // W
    xs = (x[0] * v[0]).reshape(64, g, W, g, W).sum((2, 4))
    n = v[0].reshape(g, W, g, W).sum((1, 3)).float()
    m = (xs / n.clamp_min(1)).permute(1, 2, 0).reshape(g * g, 64)
    keep = (n.reshape(-1) > 0)
    return (m[keep].numpy().astype(np.float32),
            (n.reshape(-1)[keep] / W ** 2).numpy().astype(np.float16),
            keep.nonzero().flatten().numpy().astype(np.uint8))


def window_means(rows, workers, cfg_path, label):
    """Means (N, 64), valid fractions, chip ids and window positions
    (row * 16 + col) of every window with a valid pixel in `rows`."""
    # chip_id, not the frame label: store.split renumbers its frames from 0
    jobs = [(int(r["shard"]), int(r["slot"]), int(r["chip_id"])) for _, r in
            rows.sort_values(["shard", "slot"]).iterrows()]
    out_m, out_f, out_c, out_p = [], [], [], []
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(cfg_path,)) as p:
        for i, (m, f, pos) in enumerate(p.imap(_chip, jobs, chunksize=4)):
            out_m.append(m)
            out_f.append(f)
            out_p.append(pos)
            out_c.append(np.full(len(m), jobs[i][2], np.int32))
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"{label}: {i + 1}/{len(jobs)} chips, {el / 60:.1f} min, "
                      f"eta {el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min",
                      flush=True)
    return (np.concatenate(out_m), np.concatenate(out_f),
            np.concatenate(out_c), np.concatenate(out_p))


# ---- stage 2: RVQ books ----

def _sse(cnt, s, sq):
    return sq - (s * s).sum(1) / cnt.clamp_min(1)


@torch.no_grad()
def greedy_split_init(X, K, frac=0.2, power_iters=8, refine=4, log=None):
    """K seeds for X (N, D) by distortion-driven splitting.

    Start from one cell holding every point. Each leaf without a
    candidate gets one: its principal direction (power iteration) and a
    threshold on the projection refined by 1-D 2-means, scored by the
    squared error the split would remove, SSE(parent) - SSE(left) -
    SSE(right). Each round accepts the best `frac` of all pending
    candidates (at least one, at most the remaining budget); only the two
    children of an accepted leaf get new candidates. Stops at K leaves
    and returns their means. Candidates of a round are evaluated together
    with scatter-reductions, as in georvq.model.vqinit.tree_init.
    """
    N, D = X.shape
    dev = X.device
    node = torch.zeros(N, dtype=torch.long, device=dev)
    gain = torch.full((K,), -math.inf, device=dev)
    MU = torch.zeros(K, D, device=dev)
    V = torch.zeros(K, D, device=dev)
    THR = torch.zeros(K, device=dev)
    need = torch.zeros(1, dtype=torch.long, device=dev)
    n_leaves, rounds = 1, 0
    while n_leaves < K:
        # candidates for the leaves in `need`
        lut = torch.zeros(K, dtype=torch.bool, device=dev)
        lut[need] = True
        sel = lut[node].nonzero().flatten()
        Xs, nb = X[sel], node[sel]
        cnt = torch.zeros(K, device=dev).index_add_(
            0, nb, torch.ones(len(nb), device=dev))
        mu = torch.zeros(K, D, device=dev).index_add_(0, nb, Xs)
        mu = mu / cnt.clamp_min(1)[:, None]
        xc = Xs - mu[nb]
        v = torch.randn(K, D, device=dev)
        v = v / v.norm(dim=1, keepdim=True)
        for _ in range(power_iters):
            a = torch.zeros(K, D, device=dev).index_add_(
                0, nb, xc * (xc * v[nb]).sum(1, keepdim=True))
            v = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
        p = (xc * v[nb]).sum(1)
        t = torch.zeros(K, device=dev)
        for _ in range(refine):
            r = p > t[nb]
            cr = torch.zeros(K, device=dev).index_add_(0, nb, r.float())
            sr = torch.zeros(K, device=dev).index_add_(0, nb, p * r)
            sl = torch.zeros(K, device=dev).index_add_(0, nb, p * ~r)
            cl = cnt - cr
            mr, ml = sr / cr.clamp_min(1), sl / cl.clamp_min(1)
            t = torch.where((cr > 0) & (cl > 0), 0.5 * (ml + mr), t)
        r = p > t[nb]
        sq = (Xs * Xs).sum(1)
        stats = []
        for side in (~r, r):
            f = side.float()
            c = torch.zeros(K, device=dev).index_add_(0, nb, f)
            s = torch.zeros(K, D, device=dev).index_add_(0, nb, Xs * f[:, None])
            q = torch.zeros(K, device=dev).index_add_(0, nb, sq * f)
            stats.append((c, s, q))
        (cl, sl, ql), (cr, sr, qr) = stats
        g = _sse(cl + cr, sl + sr, ql + qr) - _sse(cl, sl, ql) - _sse(cr, sr, qr)
        g = torch.where((cl > 0) & (cr > 0), g, torch.full_like(g, -math.inf))
        gain[need], MU[need], V[need], THR[need] = \
            g[need], mu[need], v[need], t[need]
        del Xs, nb, xc, p, r, sq

        # accept the best `frac` of all pending candidates
        k = min(K - n_leaves, max(1, math.ceil(frac * n_leaves)))
        top = torch.topk(gain[:n_leaves], k).indices
        if not torch.isfinite(gain[top]).all():
            top = top[torch.isfinite(gain[top])]
            k = len(top)
            if k == 0:
                break
        newid = torch.full((K,), -1, dtype=torch.long, device=dev)
        newid[top] = torch.arange(n_leaves, n_leaves + k, device=dev)
        lut = torch.zeros(K, dtype=torch.bool, device=dev)
        lut[top] = True
        sel = lut[node].nonzero().flatten()
        nb = node[sel]
        go = ((X[sel] - MU[nb]) * V[nb]).sum(1) > THR[nb]
        node[sel[go]] = newid[nb[go]]
        need = torch.cat([top, newid[top]])
        gain[need] = -math.inf
        n_leaves += k
        rounds += 1
        if log and (rounds % 10 == 0 or n_leaves == K):
            log(f"  greedy split: round {rounds}, {n_leaves} leaves")
    cnt = torch.bincount(node, minlength=K).clamp_min(1).float()
    C = torch.zeros(K, D, device=dev).index_add_(0, node, X) / cnt[:, None]
    return C


def lloyd(X, C, iters):
    """Lloyd iterations on X from seeds C. A code left empty is re-seeded
    halfway between the centroid of the cell with the largest total
    squared error and a random member of it."""
    N, K = X.shape[0], C.shape[0]
    ones = torch.ones(N, device=X.device)
    for it in range(iters):
        a = _chunked_assign(X, C)
        sums = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.zeros(K, device=X.device).index_add_(0, a, ones)
        live = cnt > 0
        C[live] = sums[live] / cnt[live, None]
        dead = (~live).nonzero().flatten()
        if len(dead):
            sse = torch.zeros(K, device=X.device).index_add_(
                0, a, (X - C[a]).pow(2).sum(1))
            for d in dead.tolist():
                j = int(sse.argmax())
                members = (a == j).nonzero().flatten()
                p = X[members[torch.randint(len(members), (1,))]][0]
                C[d] = C[j] + 0.5 * (p - C[j])
                sse[j] /= 2
    a = _chunked_assign(X, C)
    return C, a


def fit_books(Mtr, levels, K, iters, frac, dev):
    """(levels, K, D) RVQ books fitted on the training means."""
    rtr = torch.from_numpy(Mtr).to(dev)
    books = []
    for L in range(levels):
        t0 = time.time()
        C = greedy_split_init(rtr, K, frac=frac,
                              log=lambda m: print(m, flush=True))
        C, asg = lloyd(rtr, C, iters)
        rtr = rtr - C[asg]
        dead = int((torch.bincount(asg, minlength=K) == 0).sum())
        print(f"level {L + 1}: {dead} empty codes, "
              f"{time.time() - t0:.0f} s", flush=True)
        books.append(C.cpu().numpy())
    return np.stack(books)


# ---- stage 3: PCA residual coder ----

def fit_channel(Mtr, books, total_bits, n_sample, dev):
    X = torch.from_numpy(Mtr).to(dev)
    for C in books:
        Ct = torch.from_numpy(C).to(dev)
        X = X - Ct[_chunked_assign(X, Ct)]
    sel = np.random.default_rng(0).choice(len(Mtr), min(n_sample, len(Mtr)),
                                          replace=False)
    resid = X[torch.from_numpy(sel).to(dev)].cpu().numpy()
    del X
    return RVQPCAMeanChannel.fit_residual(books, resid, total_bits=total_bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/ae/ae_v2.yaml",
                    help="preset giving the store and the train/val split")
    ap.add_argument("--means", default=None,
                    help="npz cache of the window means: read if it exists, "
                         "written otherwise")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--books", default=None,
                    help="npz of RVQ books L0, L1, ...: skip the RVQ fit")
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--split-frac", type=float, default=0.2,
                    help="share of pending split candidates accepted per round")
    ap.add_argument("--resid-sample", type=int, default=2_000_000,
                    help="training residuals the PCA coder is fitted on")
    ap.add_argument("--total-bits", type=int, default=440)
    ap.add_argument("--device", default=None,
                    help="default: cuda when available, else cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available()
                                    else "cpu"))

    if a.means and os.path.exists(a.means):
        z = np.load(a.means)
        Mtr, Mva = z["Mtr"], z["Mva"]
    else:
        cfg = load_config(a.config, [])
        store = open_store(cfg.data.store_dir)
        tr, va = store.split(cfg.data.val_frac, seed=cfg.data.split_seed)
        Mva, Fva, Cva, Pva = window_means(va, a.workers, a.config, "val")
        Mtr, Ftr, Ctr, Ptr = window_means(tr, a.workers, a.config, "train")
        if a.means:
            np.savez(a.means, Mtr=Mtr, Ftr=Ftr, Ctr=Ctr, Ptr=Ptr,
                     Mva=Mva, Fva=Fva, Cva=Cva, Pva=Pva)
    print(f"means: train {Mtr.shape}, val {Mva.shape}", flush=True)

    if a.books:
        bk = np.load(a.books)
        books = np.stack([bk[f"L{i}"] for i in range(a.levels)])
    else:
        books = fit_books(Mtr, a.levels, a.K, a.iters, a.split_frac, dev)
    ch = fit_channel(Mtr, books, a.total_bits, a.resid_sample, dev)
    ch.save(a.out)
    rel = np.median(np.linalg.norm(ch.round_trip(Mva) - Mva, axis=1)
                    / np.linalg.norm(Mva, axis=1))
    print(f"wrote {a.out}: {ch.cost_bits()} bits, {ch.record_bytes()} B, "
          f"components sent {int((ch.bits > 0).sum())}, "
          f"val median relative L2 {rel:.5f}", flush=True)


if __name__ == "__main__":
    main()
