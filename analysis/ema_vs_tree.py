"""The comparison that actually decides anything: the codebooks we
TRAINED (EMA, online, with resets) versus the best tree we can build,
at the SAME K, on the same data, scored on held-out chips.

Everything else has compared construction methods to each other. This
compares them to the incumbent.

Design points that keep it fair:
  * K is fixed per tier by the trained checkpoint -- the books exist at
    2048 / 8192 / 16384 and cannot be re-chosen.
  * The residual entering level l is computed with the TRAINED books for
    levels 0..l-1 in every arm, so each level's arms all see identical
    input. Otherwise a good level-1 book would flatter its own level-2.
  * Fit and held-out latents come from disjoint chip splits.
  * The encoder is frozen, so latents are collected once and reused.

Arms per level:
  ema          the trained codebook, untouched
  tree         balanced median tree at the same K, leaf means
  tree+lloyd   the above, refined -- helps above ~50 vectors/code and
               hurts below it, so it is reported separately rather than
               folded in
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from georvq.data.pool import ChipPool
from georvq.data.store import ChipStore
from georvq.factory import from_checkpoint
from georvq.model.vq import _chunked_assign
from georvq.model.vqinit import tree_init


@torch.no_grad()
def collect(m, cfg, pool, dev, n_batches, caps, tag=""):
    specs = cfg.model.tiers
    bufs = [[] for _ in specs]
    got = [0] * len(specs)
    t0 = time.time()
    for b in range(n_batches):
        s = pool.sample_batch(cfg.train.batch_windows, cfg.model.window,
                              align=cfg.data.pool.align_crops)
        x, _ = m.source.prep(s, dev)
        for i, z in enumerate(m.encode_levels(x)):
            if got[i] >= caps[i]:
                continue
            zf = z.reshape(-1, z.shape[-1]).half()
            zf = zf[:caps[i] - got[i]]
            bufs[i].append(zf); got[i] += zf.shape[0]
        if (b + 1) % 200 == 0:
            print(f"    {tag}{b+1}/{n_batches} [{time.time()-t0:.0f}s]",
                  flush=True)
    return [torch.cat(b) for b in bufs]


def stats(r, C):
    a = _chunked_assign(r, C)
    cnt = torch.bincount(a, minlength=C.shape[0]).float()
    p = cnt / cnt.sum().clamp_min(1)
    return (float((r - C[a]).pow(2).sum(1).mean()),
            float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum())),
            int((cnt == 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="checkpoints/tessera_vq8_split2/model_best.pt")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batches", type=int, default=1500)
    ap.add_argument("--val-batches", type=int, default=200)
    ap.add_argument("--pool", type=int, default=6000)
    ap.add_argument("--cap-gib", type=float, default=8.0)
    ap.add_argument("--lloyd", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    specs = cfg.model.tiers
    store = ChipStore(cfg.data.store_dir)
    tr, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    p = cfg.data.pool
    mk = lambda rows, seed: ChipPool(
        store, rows, args.pool, dev, workers=p.workers,
        refresh_chips_per_s=p.refresh_per_s, seed=seed,
        slot_rows=p.slot_rows, pool_device="cpu", cycle=p.cycle,
        period=store.grid_period or 1)

    caps = [int(args.cap_gib * 2**30 / (sp.dim * 2)) for sp in specs]
    print(f"{args.ckpt}\ncollecting fit latents ...", flush=True)
    Z = collect(m, cfg, mk(tr, 0), dev, args.batches, caps)
    print("collecting held-out latents ...", flush=True)
    Zv = collect(m, cfg, mk(va, 1), dev, args.val_batches,
                 [c // 6 for c in caps], tag="val ")

    report = {}
    for i, (sp, rvq) in enumerate(zip(specs, m.rvqs)):
        r = Z[i].float(); Z[i] = None
        rv = Zv[i].float(); Zv[i] = None
        torch.cuda.empty_cache()
        print(f"\n=== tier {i}  K={sp.codes} dim={sp.dim}  "
              f"fit {r.shape[0]:,} ({r.shape[0]/sp.codes:,.0f}/code)  "
              f"held-out {rv.shape[0]:,} ===")
        print(f"{'lvl':>3} {'N/code':>8} | {'EMA resid':>10} {'EMA perp':>9} "
              f"| {'tree resid':>11} {'tree perp':>10} {'vs EMA':>8} "
              f"| {'tree+lloyd':>11} {'vs EMA':>8}")
        rows = []
        for l, lev in enumerate(rvq.levels):
            C_ema = lev.codebook.weight.data.float()
            e_r, e_p, e_d = stats(rv, C_ema)

            C_t = tree_init(r, sp.codes)
            t_r, t_p, t_d = stats(rv, C_t)

            C_l = C_t.clone()
            ones = torch.ones(r.shape[0], device=dev)
            for _ in range(args.lloyd):
                a2 = _chunked_assign(r, C_l)
                sums = torch.zeros_like(C_l).index_add_(0, a2, r)
                cnt = torch.zeros(sp.codes, device=dev).index_add_(0, a2, ones)
                live = cnt > 0
                C_l[live] = sums[live] / cnt[live, None]
            l_r, l_p, l_d = stats(rv, C_l)

            print(f"{l+1:>3} {r.shape[0]/sp.codes:>8,.0f} | {e_r:>10.4f} "
                  f"{e_p:>9.0f} | {t_r:>11.4f} {t_p:>10.0f} "
                  f"{100*(t_r-e_r)/e_r:>+7.2f}% | {l_r:>11.4f} "
                  f"{100*(l_r-e_r)/e_r:>+7.2f}%", flush=True)
            rows.append({"level": l + 1, "n_per_code": r.shape[0]/sp.codes,
                         "ema": e_r, "ema_perp": e_p, "ema_dead": e_d,
                         "tree": t_r, "tree_perp": t_p, "tree_dead": t_d,
                         "tree_lloyd": l_r, "tree_lloyd_perp": l_p})
            # propagate with the TRAINED book so every level's arms see
            # identical input
            r -= C_ema[_chunked_assign(r, C_ema)]
            rv -= C_ema[_chunked_assign(rv, C_ema)]
        report[f"tier{i}"] = rows
        del r, rv
        torch.cuda.empty_cache()
        if args.out:
            json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
