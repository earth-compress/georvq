"""How much data does a refit phase actually need?

Collects one large latent sample, then repeats the whole level-by-level
refit using 512 / 256 / ... / 8 batches of it, restoring the original
books each time, and scores every fit on a HELD-OUT slice that no fit
ever sees. Small-N fits will look good on their own data and bad on the
held-out one; the gap is the thing worth knowing before choosing
refit_per_code.

Also reports, per level:
  * muddling  -- fraction of vectors that switch codeword at level l+1
                 purely because level l was refitted underneath them;
  * retention -- of a donor cell's points, the fraction still in one of
                 the two children of its split;
  * leech-in  -- of the split pair's final population, the fraction that
                 arrived from OTHER cells. On a curved manifold a third
                 codeword is often nearer than either child, so a split
                 is not the clean bisection the name suggests.

Tier 1 and 2 keep every cell of each batch (no subsampling), so their
N/K range runs far higher than tier 0's.
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
from georvq.train.refit import refit_level


@torch.no_grad()
def collect(m, cfg, pool, dev, n_batches, caps):
    """fp16 latents, every cell kept, capped per tier by memory."""
    specs = cfg.model.tiers
    bufs = [[] for _ in specs]
    for b in range(n_batches):
        s = pool.sample_batch(cfg.train.batch_windows, cfg.model.window,
                              align=cfg.data.pool.align_crops)
        x, _ = m.source.prep(s, dev)
        for i, z in enumerate(m.encode_levels(x)):
            if b < caps[i]:
                bufs[i].append(z.reshape(-1, z.shape[-1]).half())
        if (b + 1) % 64 == 0:
            print(f"    {b+1}/{n_batches} batches", flush=True)
    return [torch.cat(b) for b in bufs]


@torch.no_grad()
def chain_residual(rvq, Z, upto=None):
    """Energy ratio remaining after running the whole RVQ cascade."""
    r = Z.float()
    e0 = float(r.pow(2).sum(1).mean())
    for lev in list(rvq.levels)[:upto]:
        C = lev.codebook.weight.data.float()
        r = r - C[_chunked_assign(r, C)]
    return float(r.pow(2).sum(1).mean()) / e0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tessera_vq8_split/model.pt")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batches", type=int, default=512)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--pool", type=int, default=6000)
    ap.add_argument("--lloyd", type=int, default=3)
    ap.add_argument("--tier2-cap", type=int, default=160,
                    help="batches of tier-2 latents to keep (134 MiB each)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    specs = cfg.model.tiers
    store = ChipStore(cfg.data.store_dir)
    tr, _ = store.split(cfg.data.val_frac, cfg.data.split_seed)
    p = cfg.data.pool
    pool = ChipPool(store, tr, args.pool, dev, workers=p.workers,
                    refresh_chips_per_s=p.refresh_per_s, seed=0,
                    slot_rows=p.slot_rows, pool_device="cpu", cycle=p.cycle,
                    period=store.grid_period or 1)

    nb = args.batches + args.eval_batches
    caps = [nb, nb, min(args.tier2_cap + args.eval_batches, nb)]
    caps = caps[:len(specs)]
    print(f"collecting {nb} batches (caps per tier: {caps}) ...", flush=True)
    t0 = time.time()
    Zs = collect(m, cfg, pool, dev, nb, caps)
    per_batch = [z.shape[0] // c for z, c in zip(Zs, caps)]
    print("  " + ", ".join(
        f"tier{i}: {z.shape[0]:,} vecs ({z.nbytes/2**30:.1f} GiB), "
        f"{pb:,}/batch" for i, (z, pb) in enumerate(zip(Zs, per_batch)))
        + f"  [{time.time()-t0:.0f}s]", flush=True)

    # held-out slice: the LAST eval_batches of each tier, never fitted on
    fit_Z, ev_Z = [], []
    for i, z in enumerate(Zs):
        n_ev = args.eval_batches * per_batch[i]
        ev_Z.append(z[-n_ev:])
        fit_Z.append(z[:-n_ev])
    del Zs

    orig = [[l.codebook.weight.data.clone() for l in rvq.levels]
            for rvq in m.rvqs]
    base = [chain_residual(rvq, ev) for rvq, ev in zip(m.rvqs, ev_Z)]
    print("\nheld-out residual with the ORIGINAL books: " +
          ", ".join(f"tier{i} {b:.4f}" for i, b in enumerate(base)))

    sizes = [s for s in (512, 256, 128, 64, 32, 16, 8) if s <= args.batches]
    rows = {}
    for i, (rvq, sp) in enumerate(zip(m.rvqs, specs)):
        avail = (fit_Z[i].shape[0] // per_batch[i])
        print(f"\n=== tier {i} (K={sp.codes}, {per_batch[i]:,} vecs/batch, "
              f"{avail} fit batches available) ===")
        print(f"{'batches':>8} {'N':>12} {'N/K':>8} {'muddle':>8} "
              f"{'retain':>8} {'leech':>7} {'fit resid':>10} "
              f"{'HELD-OUT':>10} {'vs orig':>9}")
        rows[f"tier{i}"] = []
        for s in sizes:
            if s > avail:
                continue
            for l, C in zip(rvq.levels, orig[i]):        # restore
                l.codebook.weight.data.copy_(C)
            Zf = fit_Z[i][:s * per_batch[i]].float()
            r_new, r_old = Zf.clone(), Zf.clone()
            mud, ret, lch = [], [], []
            for l, lev in enumerate(rvq.levels):
                C_old = lev.codebook.weight.data.float().clone()
                if l > 0:
                    mud.append(float((_chunked_assign(r_old, C_old)
                                      != _chunked_assign(r_new, C_old))
                                     .float().mean()))
                C_new, a_new, st = refit_level(r_new, C_old,
                                               lloyd_iters=args.lloyd)
                lev.codebook.weight.data.copy_(
                    C_new.to(lev.codebook.weight.dtype))
                ret.append(st["retention_final"])
                lch.append(st["leech_in_final"])
                r_new = r_new - C_new[a_new]
                r_old = r_old - C_old[_chunked_assign(r_old, C_old)]
            fit_res = float(r_new.pow(2).sum(1).mean()) \
                / float(Zf.pow(2).sum(1).mean())
            ho = chain_residual(rvq, ev_Z[i])
            n = Zf.shape[0]
            print(f"{s:>8} {n:>12,} {n/sp.codes:>8.0f} "
                  f"{100*sum(mud)/len(mud):>7.1f}% "
                  f"{100*sum(ret)/len(ret):>7.1f}% "
                  f"{100*sum(lch)/len(lch):>6.1f}% "
                  f"{fit_res:>10.4f} {ho:>10.4f} "
                  f"{100*(ho-base[i])/base[i]:>+8.2f}%", flush=True)
            rows[f"tier{i}"].append(
                {"batches": s, "N": n, "n_per_code": n / sp.codes,
                 "muddle": sum(mud)/len(mud), "retention": sum(ret)/len(ret),
                 "leech_in": sum(lch)/len(lch), "fit_resid": fit_res,
                 "heldout_resid": ho, "base_resid": base[i]})
            del Zf, r_new, r_old
            torch.cuda.empty_cache()
        for l, C in zip(rvq.levels, orig[i]):            # leave as found
            l.codebook.weight.data.copy_(C)

    if args.out:
        json.dump({"ckpt": args.ckpt, "rows": rows}, open(args.out, "w"),
                  indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
