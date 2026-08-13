"""Dry-run the offline refit procedure on a trained checkpoint.

Answers, without committing to a training run:
  * what the fit actually does level by level, per tier;
  * the MUDDLING RATIO -- of the vectors entering level l+1, what fraction
    change which codeword they pick, purely because level l was refitted
    underneath them and before level l+1 is itself refitted. This is the
    direct measure of how much churn at one level disturbs the structure
    the next level had already adapted to, and it is what justifies (or
    refutes) doing the levels strictly in cascade order;
  * decoded cosine on held-out chips, before vs after, across the rate
    ladder -- the only number that decides whether any of it was worth it.

  python analysis/refit_dryrun.py --ckpt checkpoints/tessera_vq8_split/model.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from georvq.data.pool import ChipPool, fixed_val_tokens
from georvq.data.store import ChipStore
from georvq.factory import from_checkpoint
from georvq.model.vq import _chunked_assign
from georvq.train.refit import _cell_stats, _perp, collect_latents, refit_level
from georvq.train.validate import tname, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tessera_vq8_split/model.pt")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--per-code", type=int, default=100)
    ap.add_argument("--pool", type=int, default=6000)
    ap.add_argument("--lloyd", type=int, default=3)
    ap.add_argument("--dead-frac", type=float, default=0.02)
    ap.add_argument("--recycle-frac", type=float, default=0.05)
    ap.add_argument("--val-tokens", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=1,
                    help="successive refit rounds. Each draws a FRESH "
                         "sample: re-using one sample would let Lloyd "
                         "converge on it and drop muddling for a reason "
                         "that says nothing about the procedure.")
    ap.add_argument("--save-books", default=None,
                    help="write a checkpoint with the refitted books")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    specs = cfg.model.tiers
    print(f"{args.ckpt}\n  window {cfg.model.window}, tiers "
          f"{[str(s) for s in specs]}, batch {cfg.train.batch_windows}")

    store = ChipStore(cfg.data.store_dir)
    tr, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val_s = fixed_val_tokens(store, va, token=cfg.model.window,
                             max_tokens=args.val_tokens).to(dev)
    p = cfg.data.pool
    pool = ChipPool(store, tr, args.pool, dev, workers=p.workers,
                    refresh_chips_per_s=p.refresh_per_s, seed=0,
                    slot_rows=p.slot_rows, pool_device="cpu",
                    cycle=p.cycle,
                    period=store.grid_period or 1)

    # ---- decoded accuracy BEFORE -------------------------------------
    TS = [(8, 0, 0), (8, 1, 0), (8, 4, 0), (8, 8, 0),
          (8, 8, 1), (8, 8, 2), (8, 8, 4), (8, 8, 8)]
    TS = [t for t in TS if all(L <= s.depth for L, s in zip(t, specs))]
    print("\nvalidating BEFORE ...", flush=True)
    m.eval()
    vs0, cb0 = validate(m, m.source, val_s, TS, set())

    report = {}
    for rnd in range(1, args.rounds + 1):
        print(f"\n########## REFIT ROUND {rnd}/{args.rounds} ##########")
        t0 = time.time()
        m.eval()
        samples, nb = collect_latents(m, m.source, pool, cfg, dev,
                                      per_code=args.per_code)
        print(f"collected {nb} fresh batches -> " +
              ", ".join(f"tier{i} {z.shape[0]:,} "
                        f"({z.shape[0]/s.codes:.0f}/code)"
                        for i, (z, s) in enumerate(zip(samples, specs)))
              + f"  [{time.time()-t0:.0f}s]", flush=True)
        report[f"round{rnd}"] = _one_round(m, specs, samples, args)
        del samples
        torch.cuda.empty_cache()

    _tail(m, val_s, TS, specs, cfg, cb0, vs0, args, report)

    if args.save_books:
        from georvq.repro import save_checkpoint
        save_checkpoint(args.save_books, m, cfg, 0,
                        {"refit_rounds": args.rounds})
        print(f"\nwrote refitted books -> {args.save_books}")


def _one_round(m, specs, samples, args):
    out = {}
    for i, (rvq, sp, Z) in enumerate(zip(m.rvqs, specs, samples)):
        print(f"\n=== tier {i}  (K={sp.codes}, dim={sp.dim}, "
              f"cells={sp.cells}, N={Z.shape[0]:,}) ===")
        print(f"{'lvl':>3} {'zero':>6} {'recyc':>6} {'perp b':>8} {'perp a':>8} "
              f"{'resid b':>9} {'resid a':>9} {'gain':>7} {'muddle':>7}")
        r_new = Z.clone()          # residual under refitted upstream books
        r_old = Z.clone()          # residual under the ORIGINAL books
        rows = []
        for l, lev in enumerate(rvq.levels):
            C_old = lev.codebook.weight.data.float().clone()

            # muddling: same level-(l) book, but two different inputs --
            # the residual the old upstream chain produces vs the refitted
            # one. Fraction of vectors that pick a different codeword.
            if l > 0:
                a_o = _chunked_assign(r_old, C_old)
                a_n = _chunked_assign(r_new, C_old)
                muddle = float((a_o != a_n).float().mean())
            else:
                muddle = 0.0

            C_new, a_new, st = refit_level(
                r_new, C_old, dead_frac=args.dead_frac,
                recycle_frac=args.recycle_frac, lloyd_iters=args.lloyd)
            lev.codebook.weight.data.copy_(
                C_new.to(lev.codebook.weight.dtype))
            gain = 100 * (st["resid_before"] - st["resid_after"]) \
                / max(st["resid_before"], 1e-12)
            print(f"{l+1:>3} {st['n_zero_before']:>6} "
                  f"{st['n_recycled']:>6.0f} "
                  f"{st['perp_before']:>8.0f} {st['perp_after']:>8.0f} "
                  f"{st['resid_before']:>9.4f} {st['resid_after']:>9.4f} "
                  f"{gain:>6.1f}% {100*muddle:>6.1f}%", flush=True)
            rows.append({"level": l + 1, "muddle": muddle, "gain_pct": gain,
                         **{k: v for k, v in st.items()}})
            r_new = r_new - C_new[a_new]
            r_old = r_old - C_old[_chunked_assign(r_old, C_old)]
        out[f"tier{i}"] = rows
        del r_new, r_old
    return out


def _tail(m, val_s, TS, specs, cfg, cb0, vs0, args, report):
    # ---- decoded accuracy AFTER --------------------------------------
    print("\nvalidating AFTER ...", flush=True)
    m.eval()
    vs1, cb1 = validate(m, m.source, val_s, TS, set())

    px = cfg.model.window ** 2
    print(f"\n{'tuple':>9} {'b/px':>8} {'cos before':>11} {'cos after':>10} "
          f"{'delta':>9}")
    for t in TS:
        b = sum(s.cells * L * math.log2(s.codes) for s, L in zip(specs, t)) / px
        n = tname(t)
        print(f"{n:>9} {b:>8.3f} {cb0[n]:>11.4f} {cb1[n]:>10.4f} "
              f"{cb1[n]-cb0[n]:>+9.4f}")
    print(f"{'ceiling':>9} {'-':>8} {vs0['cos_ceiling']:>11.4f} "
          f"{vs1['cos_ceiling']:>10.4f} "
          f"{vs1['cos_ceiling']-vs0['cos_ceiling']:>+9.4f}")

    if len(report) > 1:
        print("\nMUDDLING BY ROUND (mean over levels 2+, per tier):")
        hdr = " ".join(f"{r:>10}" for r in report)
        print(f"{'tier':>5} {hdr}")
        for i, sp in enumerate(specs):
            cells = []
            for r in report:
                mv = [row["muddle"] for row in report[r][f"tier{i}"]
                      if row["level"] > 1]
                cells.append(f"{100*sum(mv)/len(mv):>9.1f}%")
            print(f"{i:>5} " + " ".join(f"{c:>10}" for c in cells))

    if args.out:
        json.dump({"ckpt": args.ckpt, "report": report,
                   "cos_before": {tname(t): cb0[tname(t)] for t in TS},
                   "cos_after": {tname(t): cb1[tname(t)] for t in TS}},
                  open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
