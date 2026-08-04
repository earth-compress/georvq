"""Stage-1 trainer: a continuous, noise-robust, sigma-conditioned pyramidal
autoencoder.

No quantization. Each tier's latent is L2-normalized (× sqrt(dim)) then
corrupted with additive uniform noise whose magnitude is drawn per-sample
per-tier from [0, sigma_max]; the decoder is conditioned on that per-tier
sigma vector instead of a matryoshka depth thermometer. Objective: a
maximally expressive, near-lossless, quantization-AWARE transform to
warm-start (or cache latents for) the quantized stages. The sigma=0
reconstruction is the ceiling.
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.metrics import centered_cos
from georvq.repro import (run_metadata, save_checkpoint, seed_all,
                          write_resolved_config)
from georvq.train.figures import stage1_recon_figure
from georvq.train.loop import resolve_device, setup_data
from georvq.train.warmstart import load_partial

SIG_FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]   # val sigma levels (fractions of max)


def sample_sigmas(sig_cfg, batch, n_tiers, device):
    """Per-tier noise levels for one training batch.

    Legacy: independent U(0, max) per tier. With sig_cfg.truncate: nested
    matryoshka sampling — draw how many coarsest tiers to keep (probs
    ordered [all, ..., coarse only]), dropped tiers get sigma=max, kept
    tiers U(0, kept_max). Nesting invariant: a dropped tier implies every
    finer tier is dropped."""
    if not sig_cfg.truncate:
        return torch.rand(batch, n_tiers, device=device) * sig_cfg.max
    probs = torch.tensor(sig_cfg.truncate, dtype=torch.float, device=device)
    n_drop = torch.multinomial(probs, batch, replacement=True)  # 0 = keep all
    dropped = torch.arange(n_tiers, device=device)[None] >= (n_tiers - n_drop)[:, None]
    sigmas = torch.rand(batch, n_tiers, device=device) * sig_cfg.kept_max
    return torch.where(dropped, torch.full_like(sigmas, sig_cfg.max), sigmas)


@torch.no_grad()
def validate_stage1(model, source, val_s, sigma_max, device, batch=256):
    """Recon cosine at several uniform (all-tier) sigma levels. frac=0 is
    the near-lossless ceiling; higher frac = more corruption = lossier."""
    model.eval()
    n_tiers = len(model.specs)
    chunks = [source.prep(val_s[i:i + batch], device)
              for i in range(0, val_s.shape[0], batch)]
    out = {}
    for frac in SIG_FRACS:
        agg, n = {}, 0
        for x, valid in chunks:
            sig = torch.full((x.shape[0], n_tiers), sigma_max * frac,
                             device=device)
            rec = model.forward_stage1(x, sig)
            _, st = source.recon_stats(rec, x, valid)
            nb = int(valid.sum()); n += nb
            for k in ("cos", "vec_rel_l2"):
                if k in st:
                    agg[k] = agg.get(k, 0.0) + float(st[k]) * nb
            agg["ccos"] = agg.get("ccos", 0.0) \
                + float(centered_cos(rec, x, valid)) * nb
        out[frac] = {k: v / n for k, v in agg.items()}
    # truncation ladder: keep the k coarsest tiers clean, drop the rest
    # (sigma=max) — the stage-1 analog of the stage-2 recon tuples
    for k in range(1, n_tiers):
        cs, n = 0.0, 0
        for x, valid in chunks:
            sig = torch.full((x.shape[0], n_tiers), float(sigma_max),
                             device=device)
            sig[:, :k] = 0.0
            rec = model.forward_stage1(x, sig)
            _, st = source.recon_stats(rec, x, valid)
            nb = int(valid.sum()); cs += float(st["cos"]) * nb; n += nb
        out[f"keep{k}"] = {"cos": cs / n}
    model.train()
    return out


def train(cfg: Config, gpu=0):
    assert cfg.model.sigma is not None, \
        "stage-1 training needs model.sigma set (and model.vq: null)"
    if cfg.model.sigma.truncate is not None:
        assert len(cfg.model.sigma.truncate) == len(cfg.model.tiers), \
            "sigma.truncate needs one probability per tier"
    seed_all(cfg.train.seed)
    device = resolve_device(gpu)
    if cfg.train.tf32:
        torch.set_float32_matmul_precision("high")
    amp_on = cfg.train.amp and device.type == "cuda"
    sigma_max = cfg.model.sigma.max
    n_tiers = len(cfg.model.tiers)
    run_dir = os.path.join(cfg.log.out_dir, cfg.log.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    write_resolved_config(run_dir, cfg)
    print(f"stage 1: {[str(s) for s in cfg.model.tiers]} "
          f"sigma_max={sigma_max}", flush=True)

    model = build_model(cfg, device)
    source = model.source
    store, train_rows, val_rows, val_s, pool = setup_data(cfg, source, device)
    val_rows.to_parquet(os.path.join(run_dir, "val_chips.parquet"))
    print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"params", flush=True)
    if cfg.train.init_from:
        load_partial(model, cfg.train.init_from, device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.optim.lr,
                            weight_decay=cfg.train.optim.wd)
    warmup, lr0 = cfg.train.optim.warmup, cfg.train.optim.lr0_steps

    def lr_fn(s):
        if s < lr0:                   # AdamW warms its moments at lr=0 first
            return 0.0
        s -= lr0
        total = max(cfg.train.steps - lr0, 1)
        return (min(1.0, (s + 1) / warmup)
                * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    use_wandb = cfg.log.wandb
    if use_wandb:
        import wandb
        wandb.init(project=cfg.log.wandb_project, name=cfg.log.run_tag,
                   config=cfg.model_dump(mode="json") | run_metadata(cfg))

    batch = cfg.train.batch_windows
    align = cfg.data.pool.align_crops
    t0 = time.time()
    acc = {"loss": 0.0, "cos": 0.0}; na = 0
    vs = {0.0: {}}
    for step in range(1, cfg.train.steps + 1):
        s = pool.sample_batch(batch, token=cfg.model.window, align=align)
        x, valid = source.prep(s, device)
        sigmas = sample_sigmas(cfg.model.sigma, x.shape[0], n_tiers, device)
        with torch.autocast(device.type, torch.bfloat16, enabled=amp_on):
            rec = model.forward_stage1(x, sigmas)
        rec = rec.float()
        loss, st = source.recon_stats(rec, x, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        acc["loss"] += loss.item(); acc["cos"] += float(st["cos"]); na += 1
        if "vec_rel_l2" in st:
            acc["rel"] = acc.get("rel", 0.0) + float(st["vec_rel_l2"])

        if step % 100 == 0:
            a = {k: val / na for k, val in acc.items()}
            acc = {k: 0.0 for k in acc}; na = 0
            print(f"step {step}: loss={a['loss']:.4f} cos={a['cos']:.4f} "
                  + (f"rel={a['rel']:.4f} " if "rel" in a else "")
                  + f"gn={gn:.2f} lr={sched.get_last_lr()[0]:.1e} "
                  f"{(time.time()-t0)/step*1000:.0f}ms/step", flush=True)
            if use_wandb:
                log = {"train/loss": a["loss"], "train/cos": a["cos"],
                       "train/grad_norm": float(gn),
                       "train/lr": sched.get_last_lr()[0]}
                if "rel" in a:
                    log["train/vec_rel_l2"] = a["rel"]
                if pool.cycle:
                    log["data/epochs"] = pool.epochs
                    log["data/reuse_per_patch"] = pool.reuse_estimate
                wandb.log(log, step=step)
        if step % cfg.train.val_interval == 0 or step == cfg.train.steps:
            vs = validate_stage1(model, source, val_s, sigma_max, device)
            ceil = vs[0.0]["cos"]
            trunc = {k: v for k, v in vs.items() if isinstance(k, str)}
            print(f"step {step} VAL ceiling(σ=0)={ceil:.4f}"
                  + (f" rel={vs[0.0]['vec_rel_l2']:.4f}"
                     if "vec_rel_l2" in vs[0.0] else "") + "  "
                  + " ".join(f"σ{f:.2f}={vs[f]['cos']:.4f}"
                             for f in SIG_FRACS[1:])
                  + ("  " + " ".join(f"{k}={v['cos']:.4f}"
                                     for k, v in trunc.items())
                     if trunc else ""), flush=True)
            if use_wandb:
                log = {f"val/cos_sigma{f:.2f}": vs[f]["cos"]
                       for f in SIG_FRACS}
                log["val/ceiling_cos"] = ceil
                log.update({f"val/ccos_sigma{f:.2f}": vs[f]["ccos"]
                            for f in SIG_FRACS})
                log.update({f"val/vec_rel_l2_sigma{f:.2f}": vs[f]["vec_rel_l2"]
                            for f in SIG_FRACS if "vec_rel_l2" in vs[f]})
                log.update({f"val/cos_{k}": v["cos"]
                            for k, v in trunc.items()})
                if step % cfg.train.fig_interval == 0:
                    fig = stage1_recon_figure(model, source, val_s,
                                              sigma_max, step)
                    log["figs/recon"] = wandb.Image(fig); plt.close(fig)
                wandb.log(log, step=step)
        if step % cfg.train.save_interval == 0 or step == cfg.train.steps:
            save_checkpoint(os.path.join(run_dir, "model.pt"), model, cfg,
                            step, {"cos_sigma0": vs[0.0].get("cos")
                                   if step >= cfg.train.val_interval else None})
    pool.shutdown()
    print(f"done -> {run_dir}", flush=True)
    return run_dir
