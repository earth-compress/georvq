"""Decoder-only training from a cached bitstream.

Consumes the index store (scripts/build_index_store.py) and trains ONLY
the decode path: cached RVQ indices -> codebook lookup -> decode -> loss
against live pixels. The encoder never runs, and the codebooks are
frozen, so the bitstream is fixed for the whole run -- every checkpoint
this produces decodes packets the released encoder already emits.

That is the point of the experiment: `docs/BITSTREAM.md` says decoder
improvements are free, and this measures how much is actually there.

    python scripts/train_decoder_indexed.py --init checkpoints/.../init.pt \
        --idx-store /data/georvq-stores/tessera_idx64 --gpu 0
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
import numpy as np
import pandas as pd
import torch

from georvq.data.pool import ChipPool
from georvq.data.store import open_store
from georvq.factory import from_checkpoint
from georvq.inference.api import Codec
from georvq.model.codec import path_tuples, sample_path_depths
from georvq.repro import save_checkpoint, seed_all
from georvq.sources.report import resolve_report
from georvq.train.validate import canonical_tuples, tname, validate

# Trained: the decode path. Frozen: everything that defines the bitstream.
TRAIN_PREFIXES = ("decoder.", "stats_dec.")


def set_trainable(model):
    n_tr = n_fr = 0
    for n, p in model.named_parameters():
        train = any(n.startswith(pre) for pre in TRAIN_PREFIXES)
        p.requires_grad_(train)
        n_tr += p.numel() if train else 0
        n_fr += 0 if train else p.numel()
    return n_tr, n_fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="grown checkpoint")
    ap.add_argument("--idx-store", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr0-steps", type=int, default=500,
                    help="steps at lr=0 first: asserts the surgery was "
                         "function-preserving before anything moves")
    ap.add_argument("--pool-chips", type=int, default=1500)
    ap.add_argument("--pool-workers", type=int, default=8)
    ap.add_argument("--refresh-per-s", type=float, default=12.0)
    ap.add_argument("--val-windows", type=int, default=1024)
    ap.add_argument("--val-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=2500)
    ap.add_argument("--run-tag", default="tessera_dec_grown")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--amp", action="store_true", default=True,
                    help="bf16 autocast on the DECODER forward. FINDINGS: "
                         "bf16 belongs on decoder forwards only -- it is "
                         "the encoder whose latents must stay bit-stable, "
                         "and here the encoder does not run at all.")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--grad-ckpt", action="store_true", default=False,
                    help="recompute activations in backward. Off by "
                         "default here: decoder-only training has memory "
                         "to spare, and checkpointing trades ~30% speed "
                         "for memory we are not short of.")
    ap.add_argument("--bench-only", action="store_true",
                    help="measure step time at this batch and exit")
    args = ap.parse_args()

    seed_all(0)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                       else "cpu")
    model = from_checkpoint(args.init, dev)
    cfg = model.full_config
    model.decoder.grad_ckpt = args.grad_ckpt
    model.tokenizer.grad_ckpt = args.grad_ckpt
    codec = Codec(model)
    spec = resolve_report(model.source, cfg)
    n_tr, n_fr = set_trainable(model)
    print(f"trainable {n_tr/1e6:.1f}M | frozen {n_fr/1e6:.1f}M "
          f"(encoder never runs; codebooks fixed)", flush=True)

    man = json.load(open(os.path.join(args.idx_store, "manifest.json")))
    assert man["window"] == cfg.model.window, "window mismatch"
    mms = [np.load(os.path.join(args.idx_store, f"idx_t{i}.u8.npy"),
                   mmap_mode="r") for i in range(len(model.specs))]
    mean_mm = (np.load(os.path.join(args.idx_store, "mean.f16.npy"),
                       mmap_mode="r") if man["has_mean"] else None)
    lidx = pd.read_parquet(os.path.join(args.idx_store, "index.parquet"))
    win = man["window"]

    store = open_store(man["chip_store"])
    rows = store.index
    n_off = (store.chip_px // win) ** 2
    side = store.chip_px // win
    # The cache numbered windows by ROW POSITION in store.index; if the
    # store were reordered the pairing would silently rot, so check it.
    assert int(lidx.chip_id.iloc[0]) == int(rows.chip_id.iloc[0]), \
        "index store row order does not match the chip store"
    assert len(lidx) == len(rows) * n_off, "index store / chip store mismatch"

    pool = ChipPool(store, rows, args.pool_chips, torch.device("cpu"),
                    workers=args.pool_workers,
                    refresh_chips_per_s=args.refresh_per_s, seed=0,
                    slot_rows=None, pool_device="cpu", cycle=True)

    # ---- deterministic val windows -------------------------------------
    vrng = np.random.default_rng(123)
    vw = np.sort(vrng.choice(len(lidx), args.val_windows, replace=False))
    vpix = []
    for w in vw:
        r = lidx.iloc[int(w)]
        chip = np.asarray(store.open_shard(int(r.shard))[int(r.slot)])
        if getattr(store, "aux_bands", 0):
            chip = np.concatenate([chip, store.aux_bytes(
                int(r.shard), int(r.slot), 0, store.chip_px)])
        vpix.append(chip[:, int(r.r0):int(r.r0) + win,
                         int(r.c0):int(r.c0) + win])
    # Validation deliberately re-encodes from pixels rather than reading
    # cached indices: it then measures the WHOLE codec, which is what
    # makes its numbers comparable with the released model's.
    val_s = torch.from_numpy(np.stack(vpix)).to(dev)
    mins = cfg.model.min_depths
    canon = canonical_tuples(model.specs, mins, cfg.model.depth_sampling)
    print(f"val: {len(vw)} windows; canon "
          f"{ {k: tname(v) for k, v in canon.items()} }", flush=True)

    # ---- prefetch: pixels from the resident pool, indices from memmap ---
    stop = threading.Event()
    q: queue.Queue = queue.Queue(maxsize=3)

    def producer(seed):
        g = torch.Generator().manual_seed(seed)
        while not stop.is_set():
            ci = torch.randint(0, args.pool_chips, (args.batch,), generator=g)
            wo = torch.randint(0, n_off, (args.batch,), generator=g)
            r = (wo // side) * win
            c = (wo % side) * win
            # ONE gather kernel over the resident pool, exactly as
            # ChipPool.sample_batch does. Slicing whole chips out first
            # would move 34 MB per window (a 130x512x512 int8 chip) to
            # extract 0.5 MB, which measured as a hard ~70 win/s ceiling
            # independent of batch size.
            tt = torch.arange(win)
            bands = torch.arange(pool.bands)
            with pool.cuda_lock:
                pix = pool.pool[ci[:, None, None, None],
                                bands[None, :, None, None],
                                (r[:, None] + tt)[:, None, :, None],
                                (c[:, None] + tt)[:, None, None, :]]
                ridx = pool.slot_row[ci].clone().numpy()
            wo = wo.numpy()
            wids = ridx * n_off + wo
            order = np.argsort(wids)          # memmap locality
            inv = np.empty_like(order); inv[order] = np.arange(len(order))
            got = [torch.from_numpy(m[wids[order]][inv].astype(np.int64))
                   for m in mms]
            mu = (torch.from_numpy(mean_mm[wids[order]][inv].astype(np.float32))
                  if mean_mm is not None else None)
            try:
                q.put((pix, got, mu), timeout=5)
            except queue.Full:
                continue

    prod = [threading.Thread(target=producer, args=(7 + i,), daemon=True)
            for i in range(2)]
    for t in prod:
        t.start()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=cfg.train.optim.wd)

    def lr_at(step):
        """Flat zero, then flat low. No schedule: the experiment asks what
        capacity buys, and a decaying lr would confound that with the
        schedule's own shape."""
        return 0.0 if step <= args.lr0_steps else args.lr

    out_dir = args.out_dir or str(cfg.log.out_dir)
    run_dir = os.path.join(out_dir, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    use_wandb = args.wandb and not args.bench_only
    if use_wandb:
        import wandb
        wandb.init(project=cfg.log.wandb_project, name=args.run_tag,
                   config={"init": args.init, "batch": args.batch,
                           "lr": args.lr, "steps": args.steps,
                           "trainable_M": n_tr / 1e6})

    model.train()
    src = model.source
    t0 = time.time()
    for step in range(1, args.steps + 1):
        pix, idxs, mu = q.get()
        raw = pix.to(dev, non_blocking=True)
        x, valid = src.prep(raw, dev)
        idxs = [t.to(dev, non_blocking=True) for t in idxs]
        mean = mu.to(dev, non_blocking=True) if mu is not None else None
        Ls = sample_path_depths(x.shape[0], model.specs, dev, mins=mins,
                                full_frac=cfg.model.full_frac,
                                path_frac=1.0 - cfg.model.full_frac)
        # Per-ROW depths, so the shipped Codec._lookup (one depth for the
        # whole batch) does not apply: mask each level in by row instead.
        # Same arithmetic, same codebooks, just batched over depths.
        zqs = []
        for t_i, (rvq, ti) in enumerate(zip(model.rvqs, idxs)):
            depth = model.specs[t_i].depth
            cum = torch.zeros(ti.shape[0], ti.shape[1],
                              model.specs[t_i].dim, device=dev)
            for l in range(depth):
                keep = (Ls[:, t_i] > l).float()[:, None, None]
                cum = cum + keep * rvq.levels[l].codebook(ti[..., l])
            zqs.append(cum)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast(dev.type, torch.bfloat16,
                            enabled=args.amp and dev.type == "cuda"):
            out = model.decode(zqs, Ls, mean, valid)
        out = out.float()
        loss, m = src.recon_stats(out, x, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.model.vq and cfg.model.vq.clip_norm:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.model.vq.clip_norm)
        opt.step()

        if args.bench_only and step == 30:
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 30
            peak = torch.cuda.max_memory_allocated(dev) / 2**30
            print(f"batch {args.batch}: {dt*1000:.0f} ms/step  "
                  f"{args.batch/dt:.0f} win/s  peak {peak:.1f} GiB")
            stop.set(); pool.shutdown()
            return
        if step % 100 == 0:
            el = time.time() - t0
            msg = (f"step {step}: loss={loss.item():.5f} "
                   f"lr={lr_at(step):.2e} {step*args.batch/el:,.0f} win/s")
            print(msg, flush=True)
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr_at(step),
                           **{f"train/{k}": float(v.detach())
                              for k, v in m.items()}}, step=step)

        if step % args.val_interval == 0 or step == args.steps:
            vs, cbt = validate(model, src, val_s, list(canon.values()),
                               {tname(t) for t in canon.values()}, spec=spec)
            q_key = spec.quality.key
            print(f"step {step} VAL[{q_key}]: "
                  + " ".join(f"{k}={cbt[tname(t)]:.4f}"
                             for k, t in canon.items())
                  + f" ceil={vs[f'{q_key}_ceiling']:.4f}", flush=True)
            if use_wandb:
                wandb.log({f"val/{q_key}_{k}": cbt[tname(t)]
                           for k, t in canon.items()}
                          | {f"val/{q_key}_ceiling": vs[f"{q_key}_ceiling"]},
                          step=step)
            model.train()
        if step % args.save_interval == 0 or step == args.steps:
            save_checkpoint(os.path.join(run_dir, "model.pt"), model, cfg,
                            step, {"decoder_only": True})

    stop.set(); pool.shutdown()
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
