"""Per-level iterative codebook fitting against a frozen encoder.

One level at a time, to convergence, before touching the next. That
ordering is not fastidiousness -- it follows from three things we
measured:

  * SAMPLE BUDGET. Below ~5*d_eff vectors per codeword a refit is worse
    than doing nothing (measured crossover: 250/code at d_eff 51,
    500/code at d_eff 91). The online EMA gives tier-1 level 7 just 54
    vectors per code inside its window, 8x short, which is why that level
    sheds 20% of its codes at every reset no matter how the reset places
    them. Collecting once against a frozen encoder removes that limit
    entirely.
  * CHURN. Reset counts plateau at ~6500 codes per event and stop
    falling, so a single pass is nowhere near a fixed point.
  * LEECHING. After a split, only ~59% of the donor's points stay with
    the pair once Lloyd settles, and ~36% of the pair's final membership
    arrives from other cells. A split is not a local edit, so its effect
    has to be iterated rather than assumed.

Because level l+1's input is level l's residual, and refitting level l
makes 85-98% of vectors change codeword downstream (the muddling ratio),
there is no point converging level l+1 against a level l that is still
moving.

Reports, per iteration: the DISTRIBUTION of distortion across the book --
mean, width, and percentiles -- which should fall and narrow as high-
distortion cells are split and low-usage ones recycled. The partial
distortion theorem makes an equal contribution from every cell the
optimality condition, so the width narrowing IS convergence, not a
by-product. Held-out residual is tracked throughout so improvement can be
distinguished from memorisation.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from georvq.data.pool import ChipPool
from georvq.data.store import ChipStore
from georvq.factory import from_checkpoint
from georvq.model.vq import _chunked_assign
from georvq.model.vqinit import tree_init


@torch.no_grad()
def collect(m, cfg, pool, dev, n_batches, caps, tag=""):
    """Encoder-only pass, kept in fp16 on the GPU. The encoder is frozen,
    so this is done ONCE and reused for every level and every iteration."""
    specs = cfg.model.tiers
    bufs = [[] for _ in specs]
    t0 = time.time()
    for b in range(n_batches):
        s = pool.sample_batch(cfg.train.batch_windows, cfg.model.window,
                              align=cfg.data.pool.align_crops)
        x, _ = m.source.prep(s, dev)
        for i, z in enumerate(m.encode_levels(x)):
            if len(bufs[i]) * cfg.train.batch_windows * specs[i].cells \
                    < caps[i]:
                bufs[i].append(z.reshape(-1, z.shape[-1]).half())
        if (b + 1) % 200 == 0:
            print(f"    {tag}{b+1}/{n_batches} batches "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    out = []
    for i, bl in enumerate(bufs):
        z = torch.cat(bl)[:caps[i]]
        out.append(z)
        del bl
    return out


@torch.no_grad()
def book_stats(r, C, chunk=1 << 15):
    """Per-code count and distortion contribution, plus the summary the
    convergence policy watches."""
    K = C.shape[0]
    a = _chunked_assign(r, C)
    se = (r - C[a]).pow(2).sum(1)
    cnt = torch.bincount(a, minlength=K).float()
    sse = torch.zeros(K, device=r.device).index_add_(0, a, se)
    live = cnt > 0
    s = sse[live]
    q = torch.quantile(s.float(),
                       torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95],
                                    device=r.device))
    return {
        "resid": float(se.mean()),
        "dead": int((~live).sum()),
        "cv": float(s.std() / s.mean().clamp_min(1e-12)),
        "sse_mean": float(s.mean()),
        "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]),
        "live_frac": float(live.float().mean()),
        "perp": float(torch.exp(-((cnt / cnt.sum())
                                  * (cnt / cnt.sum()).clamp_min(1e-12).log()
                                  ).sum())),
    }, a, cnt, sse


@torch.no_grad()
def _principal_dir(r, idx, iters=8, cap=4096):
    if idx.numel() < 2:
        return None, 0.0
    if idx.numel() > cap:
        idx = idx[torch.randperm(idx.numel(), device=r.device)[:cap]]
    X = r[idx]
    Xc = X - X.mean(0)
    v = torch.randn(X.shape[1], device=r.device)
    v /= v.norm().clamp_min(1e-12)
    for _ in range(iters):
        v = Xc.t() @ (Xc @ v)
        n = v.norm()
        if n < 1e-12:
            return None, 0.0
        v = v / n
    return v, float((Xc @ v).pow(2).mean()) ** 0.5


@torch.no_grad()
def fit_level(r, C, r_val, recycle_frac=0.05, dead_frac=0.02, lloyd=2,
              max_iters=60, tol=1e-4, patience=3, eps=0.5, log=None):
    """Iterate split-and-settle on ONE level until held-out residual stops
    improving. Returns (C, history)."""
    hist = []
    best, bad = float("inf"), 0
    K = C.shape[0]
    for it in range(1, max_iters + 1):
        st, a, cnt, sse = book_stats(r, C)
        vst, _, _, _ = book_stats(r_val, C)
        st["val_resid"] = vst["resid"]
        st["val_cv"] = vst["cv"]
        st["iter"] = it
        hist.append(st)
        if log:
            log(it, st)
        if st["val_resid"] < best - tol * abs(best):
            best, bad = st["val_resid"], 0
        else:
            bad += 1
            if bad >= patience:
                break
        # --- victims: underused first, topped up to recycle_frac of K by
        # lowest distortion contribution (a small count carrying large
        # error is useful work; negligible error is a good victim)
        victim = cnt < dead_frac * (r.shape[0] / K)
        n_min = int(recycle_frac * K)
        if int(victim.sum()) < n_min:
            order = torch.argsort(sse)
            need = n_min - int(victim.sum())
            extra = order[~victim[order]][:need]
            victim[extra] = True
        vict = torch.nonzero(victim).flatten()
        # --- donors: highest distortion contribution among survivors
        score = sse.clone()
        score[victim] = -1.0
        donors = torch.argsort(score, descending=True)[:vict.numel()]
        order_pts = torch.argsort(a)
        a_sorted = a[order_pts]
        bounds = torch.searchsorted(a_sorted,
                                    torch.arange(K + 1, device=r.device))
        n_split = 0
        for j in range(vict.numel()):
            d = int(donors[j])
            if score[d] <= 0:
                continue
            idx = order_pts[bounds[d]:bounds[d + 1]]
            v, sig = _principal_dir(r, idx)
            if v is None or sig == 0.0:
                continue
            C[d] = C[d] + eps * sig * v
            C[int(vict[j])] = C[d] - 2 * eps * sig * v
            n_split += 1
        st["n_recycled"] = int(vict.numel())
        st["n_split"] = n_split
        for _ in range(lloyd):
            a2 = _chunked_assign(r, C)
            sums = torch.zeros_like(C).index_add_(0, a2, r)
            c2 = torch.zeros(K, device=r.device).index_add_(
                0, a2, torch.ones(r.shape[0], device=r.device))
            live = c2 > 0
            C[live] = sums[live] / c2[live, None]
    return C, hist


def level_figure(hist, tier, level, out):
    it = [h["iter"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
    ax[0].plot(it, [h["resid"] for h in hist], "o-", label="fit")
    ax[0].plot(it, [h["val_resid"] for h in hist], "s--", label="held-out")
    ax[0].set(xlabel="iteration", ylabel="residual", title=
              f"tier {tier} level {level}")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].fill_between(it, [h["p05"] for h in hist], [h["p95"] for h in hist],
                       alpha=0.25, label="p05-p95")
    ax[1].fill_between(it, [h["p25"] for h in hist], [h["p75"] for h in hist],
                       alpha=0.4, label="p25-p75")
    ax[1].plot(it, [h["p50"] for h in hist], "k-", lw=1.5, label="median")
    ax[1].set(xlabel="iteration", ylabel="per-code distortion",
              title="distortion DISTRIBUTION over the book")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    ax[2].plot(it, [h["cv"] for h in hist], "o-", color="crimson",
               label="CV (fit)")
    ax[2].plot(it, [h["val_cv"] for h in hist], "s--", color="darkorange",
               label="CV (held-out)")
    ax2 = ax[2].twinx()
    ax2.plot(it, [h["dead"] for h in hist], "^:", color="steelblue",
             label="dead codes")
    ax2.set_ylabel("dead codes", color="steelblue")
    ax[2].set(xlabel="iteration", ylabel="CV of per-code distortion",
              title="equidistortion convergence")
    ax[2].legend(fontsize=7, loc="upper right"); ax[2].grid(alpha=0.3)
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tessera_vq8_split2/model.pt")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batches", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=200)
    ap.add_argument("--pool", type=int, default=8000)
    ap.add_argument("--cap-gib", type=float, default=22.0,
                    help="per-tier GPU budget for the fit sample")
    ap.add_argument("--tiers", default="0,1,2")
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--lloyd", type=int, default=2)
    ap.add_argument("--recycle-frac", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--out", default="reports/level_fit")
    ap.add_argument("--save-books", default=None)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    specs = cfg.model.tiers
    os.makedirs(args.out, exist_ok=True)
    print(f"{args.ckpt}: window {cfg.model.window}, "
          f"tiers {[str(s) for s in specs]}")

    store = ChipStore(cfg.data.store_dir)
    tr, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    p = cfg.data.pool
    mk = lambda rows, seed: ChipPool(
        store, rows, args.pool, dev, workers=p.workers,
        refresh_chips_per_s=p.refresh_per_s, seed=seed,
        slot_rows=p.slot_rows, pool_device="cpu", cycle=p.cycle,
        period=store.grid_period or 1)

    caps = [int(args.cap_gib * 2**30 / (sp.dim * 2)) for sp in specs]
    print("fit-sample caps: " + ", ".join(
        f"tier{i} {c:,} vecs ({c/sp.codes:,.0f}/code)"
        for i, (c, sp) in enumerate(zip(caps, specs))))

    print("\ncollecting FIT latents (frozen encoder -> done once) ...")
    Z = collect(m, cfg, mk(tr, 0), dev, args.batches, caps)
    print("collecting HELD-OUT latents ...")
    Zv = collect(m, cfg, mk(va, 1), dev, args.val_batches,
                 [c // 8 for c in caps], tag="val ")
    for i, (z, zv) in enumerate(zip(Z, Zv)):
        print(f"  tier{i}: fit {z.shape[0]:,} ({z.shape[0]/specs[i].codes:,.0f}"
              f"/code), held-out {zv.shape[0]:,}")

    want = [int(t) for t in args.tiers.split(",")]
    report = {}
    for i in want:
        sp, rvq = specs[i], m.rvqs[i]
        r = Z[i].float(); Z[i] = None
        rv = Zv[i].float(); Zv[i] = None
        torch.cuda.empty_cache()
        print(f"\n{'='*72}\n=== TIER {i}  K={sp.codes} dim={sp.dim} "
              f"N={r.shape[0]:,} ({r.shape[0]/sp.codes:,.0f}/code)\n{'='*72}")
        report[f"tier{i}"] = {}
        for l, lev in enumerate(rvq.levels):
            C = lev.codebook.weight.data.float().clone()
            print(f"\n-- level {l+1} --")
            print(f"{'it':>3} {'fit resid':>10} {'HELD-OUT':>10} {'dead':>6} "
                  f"{'perp':>7} {'CV':>6} {'p50':>9} {'p95/p50':>8}")

            def log(it, st):
                print(f"{it:>3} {st['resid']:>10.4f} {st['val_resid']:>10.4f} "
                      f"{st['dead']:>6} {st['perp']:>7.0f} {st['cv']:>6.2f} "
                      f"{st['p50']:>9.3f} "
                      f"{(st['p95']/st['p50'] if st['p50'] > 1e-6 else float('inf')):>8.1f}",
                      flush=True)

            C, hist = fit_level(r, C, rv, recycle_frac=args.recycle_frac,
                                lloyd=args.lloyd, max_iters=args.max_iters,
                                tol=args.tol, patience=args.patience, log=log)
            lev.codebook.weight.data.copy_(C.to(lev.codebook.weight.dtype))
            level_figure(hist, i, l + 1,
                         f"{args.out}/tier{i}_L{l+1}.png")
            report[f"tier{i}"][f"L{l+1}"] = hist
            # propagate: level l+1 fits the residual THIS level now leaves
            r -= C[_chunked_assign(r, C)]
            rv -= C[_chunked_assign(rv, C)]
            print(f"   -> level {l+1} settled after {len(hist)} iters; "
                  f"residual passed on: fit {float(r.pow(2).sum(1).mean()):.4f}"
                  f" held-out {float(rv.pow(2).sum(1).mean()):.4f}")
            json.dump(report, open(f"{args.out}/report.json", "w"), indent=1)
        del r, rv
        torch.cuda.empty_cache()

    if args.save_books:
        from georvq.repro import save_checkpoint
        save_checkpoint(args.save_books, m, cfg, 0, {"level_fit": True})
        print(f"\nwrote {args.save_books}")


if __name__ == "__main__":
    main()
