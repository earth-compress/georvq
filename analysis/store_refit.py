"""Iterative codebook refit against the FROZEN w32hd encoder, straight off
the int8 latent store. No encoder passes at all.

Two modes.

  --mode sweep     How much data does a refit actually need? Fit at a
                   range of N/K, score every fit on a held-out sample
                   drawn from VAL chips that no fit ever sees. A small-N
                   fit looks good on its own data; the held-out column is
                   what says whether the book is better or merely
                   memorised.

  --mode iterate   Run split -> Lloyd -> propagate over the whole cascade
                   repeatedly, REDRAWING a fresh sample every iteration so
                   the books converge on the distribution rather than on
                   one sample. Convergence is tracked by the coefficient
                   of variation of per-code distortion contribution: the
                   partial distortion theorem says an optimal fixed-rate
                   quantizer contributes equal distortion from every cell,
                   so CV -> small IS the optimality condition, and
                   splitting the highest-contribution cell into the lowest
                   is gradient descent on it.

The frozen encoder is what makes this cheap: 2.66 billion tier-2 vectors
already sit on disk, so sample size stops being a constraint and the
iteration can run to a stopping criterion instead of a step budget.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_corpus_books import draw, shard_ids          # noqa: E402

from georvq.model.vq import _chunked_assign           # noqa: E402
from georvq.train.refit import _cell_stats, _perp, refit_level  # noqa: E402

STORE = "/data/tessera-samples/latents_w32hd_full"


def load_books(path, device):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    tiers, t = [], 0
    while f"rvqs.{t}.levels.0.codebook.weight" in sd:
        lv, l = [], 0
        while f"rvqs.{t}.levels.{l}.codebook.weight" in sd:
            lv.append(sd[f"rvqs.{t}.levels.{l}.codebook.weight"]
                      .float().to(device))
            l += 1
        tiers.append(lv)
        t += 1
    return tiers


def equidistortion_cv(r, C):
    """std/mean of per-code distortion contribution over OCCUPIED cells.

    The partial distortion theorem's optimality condition is that this
    goes to zero; it never quite does at finite K, but it is the right
    thing to watch."""
    _, cnt, sse = _cell_stats(r, C)
    live = cnt > 0
    s = sse[live]
    return float(s.std() / s.mean().clamp_min(1e-12)), int(live.sum())


@torch.no_grad()
def cascade(books_t, Z, lloyd=0, do_refit=False, **kw):
    """Run (and optionally refit) one tier's cascade. Returns per-level
    stats and the final residual energy ratio."""
    r = Z.float()
    e0 = float(r.pow(2).sum(1).mean())
    rows = []
    for l, C in enumerate(books_t):
        if do_refit:
            C_new, a, st = refit_level(r, C, lloyd_iters=lloyd, **kw)
            books_t[l] = C_new
            C = C_new
        else:
            a = _chunked_assign(r, C)
            st = {}
        cv, nlive = equidistortion_cv(r, C)
        st.update({"cv": cv, "live": nlive,
                   "resid": float((r - C[a]).pow(2).sum(1).mean()) / e0})
        rows.append(st)
        r = r - C[a]
    return rows, float(r.pow(2).sum(1).mean()) / e0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("sweep", "iterate"), default="iterate")
    ap.add_argument("--books", default="/home/ali/projects/ring-space/"
                                       "checkpoints/tessera_w32hd_books_tree8.pt")
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--per-code", type=int, default=250)
    ap.add_argument("--eval-per-code", type=int, default=60)
    ap.add_argument("--lloyd", type=int, default=3)
    ap.add_argument("--tiers", default="0,1,2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    books = load_books(args.books, dev)
    Ks = [b[0].shape[0] for b in books]
    ids = shard_ids(args.store)
    rng = np.random.default_rng(0)
    want = [int(t) for t in args.tiers.split(",")]
    print(f"books {args.books}\n  tiers {[len(b) for b in books]} levels, "
          f"K {Ks}, store {len(ids)} shards")

    def sample(t, per_code, val):
        n = per_code * Ks[t]
        x = draw(args.store, ids, t, n, val, rng)
        return torch.from_numpy(x).to(dev).float().mul_(6.0 / 127.0)

    result = {}
    for t in want:
        ev = sample(t, args.eval_per_code, 1)
        _, base = cascade([c.clone() for c in books[t]], ev)
        print(f"\n=== tier {t} (K={Ks[t]}, {len(books[t])} levels) ===")
        print(f"held-out sample {ev.shape[0]:,} vectors; "
              f"residual with the given books: {base:.4f}")

        if args.mode == "iterate":
            cur = [c.clone() for c in books[t]]
            print(f"\n{'iter':>5} {'N':>11} {'fit resid':>10} "
                  f"{'HELD-OUT':>10} {'vs start':>9} {'CV L1':>7} "
                  f"{'CV Ldeep':>9} {'dead':>6} {'s':>5}")
            rows = []
            for it in range(1, args.iters + 1):
                t0 = time.time()
                Z = sample(t, args.per_code, 0)          # REDRAW each pass
                st, fit_res = cascade(cur, Z, lloyd=args.lloyd,
                                      do_refit=True)
                _, ho = cascade(cur, ev)
                dead = sum(Ks[t] - s["live"] for s in st)
                print(f"{it:>5} {Z.shape[0]:>11,} {fit_res:>10.4f} "
                      f"{ho:>10.4f} {100*(ho-base)/base:>+8.2f}% "
                      f"{st[0]['cv']:>7.2f} {st[-1]['cv']:>9.2f} "
                      f"{dead:>6} {time.time()-t0:>5.0f}", flush=True)
                rows.append({"iter": it, "fit": fit_res, "heldout": ho,
                             "cv": [s["cv"] for s in st],
                             "resid": [s["resid"] for s in st],
                             "dead": dead})
                del Z
                torch.cuda.empty_cache()
            result[f"tier{t}"] = {"base": base, "rows": rows}
            books[t] = cur
        else:
            print(f"\n{'N/K':>7} {'N':>12} {'fit resid':>10} {'HELD-OUT':>10} "
                  f"{'vs start':>9} {'CV L1':>7} {'CV Ldeep':>9} {'s':>5}")
            rows = []
            for pc in (10, 25, 50, 100, 250, 500, 1000):
                if pc * Ks[t] > 12_000_000:
                    continue
                t0 = time.time()
                cur = [c.clone() for c in books[t]]
                Z = sample(t, pc, 0)
                st, fit_res = cascade(cur, Z, lloyd=args.lloyd,
                                      do_refit=True)
                _, ho = cascade(cur, ev)
                print(f"{pc:>7} {Z.shape[0]:>12,} {fit_res:>10.4f} "
                      f"{ho:>10.4f} {100*(ho-base)/base:>+8.2f}% "
                      f"{st[0]['cv']:>7.2f} {st[-1]['cv']:>9.2f} "
                      f"{time.time()-t0:>5.0f}", flush=True)
                rows.append({"n_per_code": pc, "N": Z.shape[0],
                             "fit": fit_res, "heldout": ho, "base": base})
                del Z, cur
                torch.cuda.empty_cache()
            result[f"tier{t}"] = {"base": base, "rows": rows}
        del ev
        torch.cuda.empty_cache()

    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
