"""Decoder finetune from cached latents: zero encoder cycles.

Consumes the corpus-fit books checkpoint (scripts/fit_codebooks.py) and
trains ONLY the decoder: cached int8 latents -> dequant -> RVQ quantize
(EMA books keep refining at the matured decay) -> decode -> loss vs live
pixels. The frozen encoder never runs; its latents come from the store.

Pixels ride the standard ChipPool (whole-chip slots, cycle mode); each
sampled chip contributes one of its 9 cached windows, and the matching
latent rows are gathered from the int8 memmaps by a prefetch thread so
memmap IO overlaps GPU compute.

Usage:
  python scripts/train_decoder_cached.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from georvq.data.pool import ChipPool
from georvq.data.store import open_store
from georvq.factory import from_checkpoint
from georvq.model.codec import monotone_tuples, sample_monotone_depths
from georvq.repro import save_checkpoint, seed_all
from georvq.train.figures import (codebook_figure, error_hist_figure,
                                  rd_figure, recon_figure)
from georvq.train.validate import canonical_tuples, tname, validate

OFFSETS = (0, 114, 228)
WIN = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="checkpoints/dem128_books_corpus/model.pt")
    ap.add_argument("--lat-store", default="/data/georvq-stores/copdem_lat128")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--pool-slots", type=int, default=200000)  # ~94 GiB chips
    ap.add_argument("--val-windows", type=int, default=2048)
    ap.add_argument("--val-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=2500)
    ap.add_argument("--run-tag", default="dem128_dec_cached")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume from (continues LR schedule)")
    ap.add_argument("--wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = ap.parse_args()

    seed_all(0)
    dev = torch.device(f"cuda:{args.gpu}")
    model = from_checkpoint(args.books, dev)
    cfg = model.full_config
    model.train()
    source = model.source
    for n, p in model.named_parameters():   # decoder-only optimization
        p.requires_grad_(n.startswith("decoder."))
    dec_p = [p for p in model.decoder.parameters()]
    print(f"decoder params: {sum(p.numel() for p in dec_p)/1e6:.2f}M "
          f"(encoder never runs)", flush=True)

    man = json.load(open(os.path.join(args.lat_store, "manifest.json")))
    scales = man["scales"]
    mms = [np.load(os.path.join(args.lat_store, f"lat_t{i}.i8.npy"),
                   mmap_mode="r") for i in range(len(model.specs))]
    n_chips = man["n_chips"]

    store = open_store(man["pixel_store"])
    rows = store.index
    pool = ChipPool(store, rows, args.pool_slots, torch.device("cpu"),
                    workers=6, refresh_chips_per_s=100.0, seed=0,
                    slot_rows=None, pool_device="cpu", cycle=True)
    chip_ids = rows.chip_id.to_numpy()

    run_dir = os.path.join(cfg.log.out_dir, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)

    # ---- deterministic val windows (closed distribution: fit the corpus).
    # nearly-all-nodata windows are excluded — masked R^2 is undefined on
    # them (training is safe: the masked loss zeroes their gradient)
    import pandas as pd
    lidx = pd.read_parquet(os.path.join(args.lat_store, "index.parquet"))
    good = np.flatnonzero(lidx.valid_frac.to_numpy() >= 0.5)
    vrng = np.random.default_rng(123)
    vw = np.sort(vrng.choice(good, args.val_windows, replace=False))
    vx = []
    for w in vw:
        chip = store.read_chip(int(w) // 9)
        r0, c0 = OFFSETS[(w % 9) // 3], OFFSETS[w % 3]
        vx.append(chip[:, r0:r0 + WIN, c0:c0 + WIN])
    val_s = torch.from_numpy(np.stack(vx)).to(dev)
    mins = cfg.model.min_depths
    all_tuples = [t for t in monotone_tuples(model.specs)
                  if all(L >= m for L, m in zip(t, mins))]
    canon = canonical_tuples(model.specs, mins)
    print(f"val: {len(vw)} windows, canon "
          f"{ {k: tname(v) for k, v in canon.items()} }", flush=True)

    # ---- prefetch pipeline: (pixels from pool, latents from memmap) ------
    stop = threading.Event()
    q: queue.Queue = queue.Queue(maxsize=3)

    def producer(seed):
        g = torch.Generator().manual_seed(seed)
        while not stop.is_set():
            ci = torch.randint(0, args.pool_slots, (args.batch,), generator=g)
            wo = torch.randint(0, 9, (args.batch,), generator=g)
            with pool.cuda_lock:
                chips = pool.pool[ci].clone()
                ridx = pool.slot_row[ci].clone()
            cids = chip_ids[ridx.numpy()]
            wids = cids * 9 + wo.numpy()
            order = np.argsort(wids)
            zs = [torch.from_numpy(m[wids[order]].astype(np.float32)) * s
                  for m, s in zip(mms, scales)]
            inv = np.empty_like(order); inv[order] = np.arange(len(order))
            zs = [z[inv] for z in zs]
            px = torch.stack([
                chips[b, :, OFFSETS[int(wo[b]) // 3]:OFFSETS[int(wo[b]) // 3] + WIN,
                      OFFSETS[int(wo[b]) % 3]:OFFSETS[int(wo[b]) % 3] + WIN]
                for b in range(args.batch)])
            q.put((px, zs))
        q.put(None)

    prod = threading.Thread(target=producer, args=(1,), daemon=True)
    prod.start()

    opt = torch.optim.AdamW(dec_p, lr=args.lr, weight_decay=0.01)

    def lr_fn(s):
        return (min(1.0, (s + 1) / args.warmup)
                * 0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps))))

    start_step = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        start_step = ck["step"] + 1
        print(f"resumed {args.resume} at step {ck['step']} (fresh AdamW "
              "moments; LR schedule continues)", flush=True)

    if args.wandb:
        import wandb
        wandb.init(project=cfg.log.wandb_project, name=args.run_tag,
                   config={"books": args.books, "lat_store": args.lat_store,
                           "steps": args.steps, "batch": args.batch,
                           "lr": args.lr}, resume="allow",
                   id=args.run_tag.replace("/", "_"))

    rt = ([[int(x_) for x_ in t.split("-")] for t in cfg.eval.recon_tuples]
          if cfg.eval.recon_tuples else None)
    canon_names = {tname(t) for t in canon.values()}
    acc, na, t0, best = {"loss": 0.0, "q": 0.0}, 0, time.time(), -1.0
    for step in range(start_step, args.steps + 1):
        lr = args.lr * lr_fn(step)
        for g in opt.param_groups:
            g["lr"] = lr
        px, zs = q.get()
        x = px.to(dev, non_blocking=True)
        zs = [z.to(dev, non_blocking=True) for z in zs]
        xq, valid = source.prep(x, dev)
        Ls = sample_monotone_depths(x.shape[0], model.specs, dev, mins=mins,
                                    full_frac=cfg.model.full_frac)
        with torch.no_grad():
            zqs, _, _ = model.quantize(zs, Ls)
        zqs = [zq.detach().requires_grad_(False) for zq in zqs]
        with torch.autocast("cuda", torch.bfloat16):
            out = model.decode(zqs, Ls)
        loss, m = source.recon_stats(out.float(), xq, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec_p, 1.0)
        opt.step()
        acc["loss"] += float(loss); acc["q"] += float(m["cos"]); na += 1

        if step % 100 == 0:
            a = {k: val / na for k, val in acc.items()}
            acc = {k: 0.0 for k in acc}; na = 0
            print(f"step {step}: loss={a['loss']:.4f} R2={a['q']:.4f} "
                  f"lr={lr:.1e} "
                  f"{(time.time()-t0)/max(step-start_step+1,1)*1000:.0f}"
                  "ms/step", flush=True)
            if args.wandb:
                wandb.log({"train/loss": a["loss"], "train/q": a["q"],
                           "train/lr": lr, "data/epochs": pool.epochs},
                          step=step)
        if step % args.val_interval == 0 or step == args.steps:
            # canonical points + a rotating sample of the tuple space —
            # the same protocol (and figures) as the standard trainer
            others = [t for t in all_tuples if t not in canon.values()]
            vrng2 = np.random.default_rng(step)
            k = min(12, len(others))
            val_ts = list(canon.values()) + \
                [others[i] for i in vrng2.choice(len(others), k,
                                                 replace=False)]
            vs, cbt = validate(model, source, val_s, val_ts, canon_names)
            ccbt = vs.pop("_ccos", {})
            print(f"step {step} VAL: "
                  + " ".join(f"{k_}({tname(t)})={cbt[tname(t)]:.4f}"
                             for k_, t in canon.items())
                  + f" ceil={vs['cos_ceiling']:.4f}"
                  + f" blockmean={vs['cos_blockmean']:.4f}", flush=True)
            if args.wandb:
                scalars = {f"val/cos_{k_}": cbt[tname(t)]
                           for k_, t in canon.items()}
                scalars["val/cos_ceiling"] = vs["cos_ceiling"]
                scalars["val/cos_blockmean"] = vs["cos_blockmean"]
                scalars.update({f"val/ccos_{k_}": ccbt[tname(t)]
                                for k_, t in canon.items()
                                if tname(t) in ccbt})
                # per-level codebook health scalars (ema alive/perplexity)
                scalars.update({f"books/{k_}": v_ for k_, v_ in vs.items()
                                if k_.startswith("pyr")})
                f1 = rd_figure(cbt, vs, model.specs, val_ts, step,
                               px=cfg.model.window ** 2)
                f2 = codebook_figure(vs, model.specs, step)
                scalars["figs/rd"] = wandb.Image(f1)
                scalars["figs/codebook"] = wandb.Image(f2)
                wandb.log(scalars, step=step)
                plt.close(f1); plt.close(f2)
                if step % (2 * args.val_interval) == 0 \
                        or step == args.steps:
                    f3 = recon_figure(model, source, val_s, step, mins=mins,
                                      tuples=rt)
                    f5 = error_hist_figure(model, source, val_s, step,
                                           tuples=rt, mins=mins)
                    wandb.log({"figs/recon": wandb.Image(f3),
                               "figs/err_hist": wandb.Image(f5)}, step=step)
                    plt.close(f3); plt.close(f5)
            full = cbt[tname(canon["full"])]
            if full > best:
                best = full
                save_checkpoint(os.path.join(run_dir, "model_best.pt"),
                                model, cfg, step, {"cos_full": full})
        if step % args.save_interval == 0 or step == args.steps:
            save_checkpoint(os.path.join(run_dir, "model.pt"), model, cfg,
                            step, {"step": step})
    stop.set()
    pool.shutdown()
    print(f"done -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
