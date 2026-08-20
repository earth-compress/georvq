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
from georvq.repro import (run_metadata, save_checkpoint, seed_all,
                          write_resolved_config)
from georvq.sources.report import resolve_report
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
def validate_stage1(model, source, val_s, sigma_max, device, batch=256,
                    trunc=False, spec=None):
    """Reconstruction quality at several uniform (all-tier) sigma levels.
    frac=0 is the near-lossless ceiling; higher frac = more corruption =
    lossier. Metrics come from the run's `ReportSpec`, so this reports the
    source's own headline measure rather than a cosine by assumption."""
    if spec is None:
        spec = source.report(getattr(model, "cfg", None))
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
            nb = int(valid.sum()); n += nb
            for m in spec.metrics:
                agg[m.key] = agg.get(m.key, 0.0) + float(m(rec, x, valid)) * nb
        out[frac] = {k: v / n for k, v in agg.items()}
    # truncation ladder: keep the k coarsest tiers clean, drop the rest
    # (sigma=max) — the stage-1 analog of the stage-2 recon tuples. Only
    # meaningful when the run trains truncation (sigma.truncate).
    for k in (range(1, n_tiers) if trunc else ()):
        cs, n = 0.0, 0
        for x, valid in chunks:
            sig = torch.full((x.shape[0], n_tiers), float(sigma_max),
                             device=device)
            sig[:, :k] = 0.0
            rec = model.forward_stage1(x, sig, drop=sig == sigma_max)
            nb = int(valid.sum())
            cs += float(spec.quality(rec, x, valid)) * nb; n += nb
        out[f"keep{k}"] = {spec.quality.key: cs / n}
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
    # same contract as the stage-2 trainer: the source declares what this
    # run reports, log.report.* may narrow it, and a bad selection fails
    # here rather than at the first validation
    report = resolve_report(source, cfg)
    qm = report.quality
    print(f"report: quality={qm.key} extra={[m.key for m in report.extra]}",
          flush=True)
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
    # one-deep prefetch: assemble batch N+1 (CPU-pool gather + H2D) while
    # the GPU computes batch N — hides ~100ms/step of serial stall
    from concurrent.futures import ThreadPoolExecutor
    _fetcher = ThreadPoolExecutor(1)

    def _fetch():
        return pool.sample_batch(batch, token=cfg.model.window, align=align)
    nxt = _fetcher.submit(_fetch)
    t0 = time.time()
    acc = {"loss": 0.0, qm.key: 0.0}; na = 0
    vs = {0.0: {}}
    for step in range(1, cfg.train.steps + 1):
        s = nxt.result()
        nxt = _fetcher.submit(_fetch)
        x, valid = source.prep(s, device)
        sigmas = sample_sigmas(cfg.model.sigma, x.shape[0], n_tiers, device)
        # truncation mode: dropped tiers (pinned at sigma=max) are ZEROED —
        # sigma=max noise alone still leaks signal the decoder can mine
        drop = (sigmas == sigma_max) if cfg.model.sigma.truncate else None
        with torch.autocast(device.type, torch.bfloat16, enabled=amp_on):
            rec = model.forward_stage1(x, sigmas, drop=drop)
        rec = rec.float()
        loss, st = source.recon_stats(rec, x, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        acc["loss"] += loss.item(); na += 1
        # the source's own train-time metrics, whatever it named them.
        # detach first: a source may return its loss term among the
        # metrics (Tessera's "mse" IS the loss), and float() on a
        # grad-carrying tensor warns.
        for k, val in st.items():
            if k != "loss":
                acc[k] = acc.get(k, 0.0) + float(
                    val.detach() if torch.is_tensor(val) else val)

        if step % 100 == 0:
            a = {k: val / na for k, val in acc.items()}
            acc = {k: 0.0 for k in acc}; na = 0
            now = time.time()
            sps = 100 / (now - t0)          # windowed, not cumulative
            t0 = now
            qstr = (f"{qm.key}={a[qm.key]:.4f} " if qm.key in a else "")
            print(f"step {step}: loss={a['loss']:.4f} {qstr}"
                  + f"gn={gn:.2f} lr={sched.get_last_lr()[0]:.1e} "
                  f"{1000/sps:.0f}ms/step", flush=True)
            if use_wandb:
                log = {"train/loss": a["loss"],
                       "train/grad_norm": float(gn),
                       "train/lr": sched.get_last_lr()[0],
                       "train/steps_per_s": sps,
                       "train/windows_per_s": sps * batch}
                log.update({f"train/{k}": val for k, val in a.items()
                            if k != "loss"})
                if pool.cycle:
                    log["data/epochs"] = pool.epochs
                    log["data/reuse_per_patch"] = pool.reuse_estimate
                wandb.log(log, step=step)
        if step % cfg.train.val_interval == 0 or step == cfg.train.steps:
            vs = validate_stage1(model, source, val_s, sigma_max, device,
                                 trunc=cfg.model.sigma.truncate is not None,
                                 spec=report)
            ceil = vs[0.0][qm.key]
            trunc = {k: v for k, v in vs.items() if isinstance(k, str)}
            extra0 = "".join(f" {m.key}={vs[0.0][m.key]:.4f}"
                             for m in report.extra if m.key in vs[0.0])
            print(f"step {step} VAL[{qm.key}] ceiling(σ=0)={ceil:.4f}"
                  + extra0 + "  "
                  + " ".join(f"σ{f:.2f}={vs[f][qm.key]:.4f}"
                             for f in SIG_FRACS[1:])
                  + ("  " + " ".join(f"{k}={v[qm.key]:.4f}"
                                     for k, v in trunc.items())
                     if trunc else ""), flush=True)
            if use_wandb:
                log = {f"val/{qm.key}_sigma{f:.2f}": vs[f][qm.key]
                       for f in SIG_FRACS}
                log[f"val/ceiling_{qm.key}"] = ceil
                for m in report.extra:
                    log.update({f"val/{m.key}_sigma{f:.2f}": vs[f][m.key]
                                for f in SIG_FRACS if m.key in vs[f]})
                log.update({f"val/{qm.key}_{k}": v[qm.key]
                            for k, v in trunc.items()})
                # recon panels are meaningless for near-pixel windows
                if (report.has_figure("recon")
                        and step % cfg.train.fig_interval == 0
                        and cfg.model.window >= 16):
                    fig = stage1_recon_figure(model, source, val_s,
                                              sigma_max, step,
                                              preview=report.preview)
                    log["figs/recon"] = wandb.Image(fig); plt.close(fig)
                wandb.log(log, step=step)
        if step % cfg.train.save_interval == 0 or step == cfg.train.steps:
            save_checkpoint(os.path.join(run_dir, "model.pt"), model, cfg,
                            step, {"quality_key": qm.key,
                                   f"{qm.key}_sigma0": vs[0.0].get(qm.key)
                                   if step >= cfg.train.val_interval else None})
    _fetcher.shutdown(wait=False, cancel_futures=True)
    pool.shutdown()
    print(f"done -> {run_dir}", flush=True)
    return run_dir
