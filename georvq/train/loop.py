"""The pyramid trainer: one config-driven loop for every quantized variant
(plain pyramid, rate-model, conditioned).

Precision structure (deliberate): the encoder and decoder run under bf16
autocast; quantization, the recon/commit losses and the rate term run in
fp32 outside it. VQ state must never see reduced precision.
"""
from __future__ import annotations

import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from georvq.config import Config, VQConfig
from georvq.data.pool import ChipPool, fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import build_model
from georvq.model.codec import (monotone_tuples, path_tuples,
                                sample_monotone_depths,
                                sample_path_depths, tuple_bits)
from georvq.model.vq import kmeans_init_rvq
from georvq.model.vqinit import residual_init_rvq
from georvq.repro import (run_metadata, save_checkpoint, seed_all,
                          write_resolved_config)
from georvq.train.figures import (codebook_figure, cosdist_figure,
                                  error_hist_harvest, error_hist_render,
                                  rd_err_harvest, rd_err_render, rd_figure,
                                  recon_harvest, recon_render, reset_figure)
from georvq.sources.report import resolve_report
from georvq.train.validate import canonical_tuples, load_rd_ref, tname, validate
from georvq.train import dist as D
from georvq.train.warmstart import (decoder_tail_names, freeze_books,
                                    load_partial, set_encoder_frozen,
                                    set_params_frozen, trainable_params)


def _reset_offset(reset_interval, val_interval):
    """Global step at which the first reset should fire.

    Reset steps land at {r + k*R} which, mod the validation interval V,
    takes values spaced by g = gcd(R, V). Landing on a validation means it
    scores books whose codes were moved in that very step; landing halfway
    between two is the safest. For R=2500, V=1000 that is g=500 and an
    offset of 250, so resets sit at 250 and 750 mod 1000 -- always 250
    steps clear of a validation."""
    from math import gcd
    return max(1, gcd(reset_interval, val_interval) // 2)


def resolve_device(gpu):
    if torch.cuda.is_available() and gpu is not None and gpu >= 0:
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def setup_data(cfg: Config, source, device):
    """Store, split, fixed val tokens, training pool.

    Under data parallelism the resident pool is split across ranks so the
    host RAM footprint is unchanged (two full pools would not fit). Every
    rank still draws from ALL training chips and refreshes from disk at
    the configured rate, so this costs no corpus coverage -- it only
    halves how much each rank holds in RAM at any moment. Seeds are
    offset by rank so ranks hold and sample different chips.

    val_frac=0 is closed-distribution mode (compress a fixed corpus):
    train on every chip and let "val" measure training-set fit."""
    store = open_store(cfg.data.store_dir)
    train_rows, val_rows = store.split(cfg.data.val_frac,
                                       seed=cfg.data.split_seed)
    if cfg.data.val_frac == 0:
        val_rows = train_rows
        if D.is_main():
            print(f"data: {len(store.index)} chips, closed distribution "
                  "(val tokens drawn from train chips)", flush=True)
    elif D.is_main():
        print(f"data: {len(store.index)} chips -> {len(train_rows)} train / "
              f"{len(val_rows)} val", flush=True)
    if cfg.data.limit_chips:
        order = np.random.default_rng(cfg.data.split_seed).permutation(
            len(train_rows))
        train_rows = train_rows.iloc[
            order[:cfg.data.limit_chips]].reset_index(drop=True)
        print(f"data: limit_chips -> training on {len(train_rows)} chips",
              flush=True)
    val_s = fixed_val_tokens(store, val_rows, token=cfg.model.window)
    if D.is_main():
        print(f"data: {val_s.shape[0]} fixed val tokens", flush=True)
    p = cfg.data.pool
    period = store.grid_period or getattr(source, "grid_period", 1)
    pool_size = max(1, p.size // D.world_size())
    if D.world_size() > 1 and D.is_main():
        print(f"data: pool split {D.world_size()} ways -> {pool_size} slots "
              f"per rank (all ranks draw from all {len(train_rows)} chips)",
              flush=True)
    pool = ChipPool(store, train_rows, pool_size, device, workers=p.workers,
                    refresh_chips_per_s=p.refresh_per_s,
                    seed=cfg.train.seed + 1000 * D.rank(),
                    slot_rows=p.slot_rows,
                    pool_device=("cpu" if p.device == "cpu" else device),
                    cycle=p.cycle, period=period)
    return store, train_rows, val_rows, val_s.to(device), pool


def train(cfg: Config, gpu=0):
    seed_all(cfg.train.seed)          # identical model init on every rank
    ddp_device, ddp_rank, ddp_world = D.setup(cfg)
    device = ddp_device or resolve_device(gpu)
    if cfg.train.tf32:
        torch.set_float32_matmul_precision("high")
    amp_on = cfg.train.amp and device.type == "cuda"

    specs = cfg.model.tiers
    mins = cfg.model.min_depths
    on_path = cfg.model.depth_sampling == "path"
    if on_path:
        # Path sampling trains only the strict fill order (each tier
        # saturated before the next opens), so validation and the R-D
        # front enumerate that path and nothing else -- scoring a tuple
        # like 15-10-10 would put never-trained operating points on the
        # pareto.
        tuples = [list(t) for t in path_tuples(specs, mins)]
    else:
        tuples = [t for t in monotone_tuples(specs)
                  if all(L >= m for L, m in zip(t, mins))]
    canon = canonical_tuples(specs, mins, path=on_path)
    ref_fronts = [load_rd_ref(s) for s in cfg.eval.rd_refs]
    run_dir = os.path.join(cfg.log.out_dir, cfg.log.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    write_resolved_config(run_dir, cfg)
    print(f"pyramid: {[str(s) for s in specs]} -> {len(tuples)} operating "
          f"points, {tuple_bits(specs, canon['min']):.0f}.."
          f"{tuple_bits(specs, canon['full']):.0f} bits/window", flush=True)

    model = build_model(cfg, device)
    if model.side_bits():
        print(f"stats channel: +{model.side_bits():.0f} bits/window "
              f"({model.side_bits() / cfg.model.window ** 2:.3f} b/px) "
              f"charged at every depth", flush=True)
    source = model.source
    # Resolve BEFORE the store opens: a bad log.report.* selection should
    # fail in the first second, not at the first val_interval.
    report = resolve_report(source, cfg)
    qm = report.quality
    if D.is_main():
        print(f"report: quality={qm.key} "
              f"extra={[m.key for m in report.extra]} "
              f"baselines={[b.key for b in report.baselines]} "
              f"figures={list(report.figures)}", flush=True)
    store, train_rows, val_rows, val_s, pool = setup_data(cfg, source, device)
    val_rows.to_parquet(os.path.join(run_dir, "val_chips.parquet"))
    # geometry and model-space transform resolve by name; the defaults
    # delegate to pool.sample_batch and Source.prep, so a config that
    # names neither runs the historical path exactly
    from georvq.data.sampler import build_sampler
    from georvq.sources.prep import build_prep
    sampler = build_sampler(cfg.data.sampler.name, pool, cfg.model,
                            source=source)
    prep = build_prep(cfg.data.prep.name, source, cfg.model)
    missing = [k for k in prep.requires if k not in sampler.provides]
    assert not missing, (
        f"prep {cfg.data.prep.name!r} needs meta {missing} that sampler "
        f"{type(sampler).__name__} does not provide")

    # location side-info (conditioner / selector)
    needs_loc = model.conditioner is not None or model.selector is not None
    train_loc = val_loc = None
    if needs_loc:
        from georvq.model.location import build_location_tables
        train_loc, val_s, val_loc = build_location_tables(
            cfg, store, train_rows, val_rows, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params", flush=True)
    if cfg.train.init_from:
        load_partial(model, cfg.train.init_from, device)
    enc_frozen = cfg.train.freeze_encoder_steps > 0
    if enc_frozen:
        names = set_encoder_frozen(model, True)
        print(f"froze {len(names)} encoder params for "
              f"{cfg.train.freeze_encoder_steps} steps", flush=True)

    v = cfg.model.vq
    if cfg.train.init_from and v.mature_warmstart and (
            v.ema_decay != VQConfig.model_fields["ema_decay"].default
            or v.reset_mode != "random"):
        # maturing silently wins over both of these, and a run that set
        # them meant them -- say so rather than let the run quietly use
        # decay 0.9995 with resets off
        print(f"note: mature_warmstart is on, so the warm-started books "
              f"run at decay 0.9995 with resets DISABLED -- overriding "
              f"vq.ema_decay={v.ema_decay} and vq.reset_mode="
              f"{v.reset_mode!r}. Set vq.mature_warmstart=false to keep "
              f"them live.", flush=True)
    if cfg.train.init_from and not v.mature_warmstart:
        # undo load_partial's maturing: keep the books live at the
        # configured decay with resets (and therefore reset_mode) active
        for rvq in model.rvqs:
            for lev in rvq.levels:
                lev.decay = v.ema_decay
                lev.reset_interval = 2500
        print(f"warm-started books kept LIVE: decay {v.ema_decay}, "
              f"resets every 2500 steps, reset_mode={v.reset_mode}",
              flush=True)
    if not v.warmup_steps and not cfg.train.init_from:
        # per-tier data init with >=2K vectors per codebook, drawn and
        # encoded in batch-sized chunks (a single draw of 2*max(K) windows
        # would materialize a giant gather-index tensor on the pool device)
        need = [max(1, (2 * sp.codes) // sp.cells) for sp in specs]
        samples = [[] for _ in specs]
        got = [0] * len(specs)
        with torch.no_grad():
            while any(g < n for g, n in zip(got, need)):
                s, m0 = sampler.draw(
                    min(cfg.train.batch_windows, max(need)),
                    align=cfg.data.pool.align_crops)
                x0, v0, _ = prep(s, m0, device)
                # must match the training distribution: with stats
                # conditioning the encoder sees a mean-SUBTRACTED field, so
                # books fitted on un-centred latents would be fitted to
                # vectors the model never produces
                for i, z in enumerate(
                        model.encode_levels(x0, model._stats_of(x0, v0), v0)):
                    if got[i] < need[i]:
                        samples[i].append(z)
                        got[i] += z.shape[0]
            for i, (rvq, sp) in enumerate(zip(model.rvqs, specs)):
                z = torch.cat(samples[i])[:need[i]]
                rvq.all_levels(z)
        del samples
    # Resets fire on a GLOBAL-step schedule. Each level's private counter
    # advances only on batches where depth sampling makes it active, so
    # left to themselves the counters desynchronise (observed on era5_t64:
    # resets at 3000 / 3002 / 3003, and an earlier warm-start case at
    # 464 / 803 / 2963, the last landing 37 steps before a validation,
    # which then scored books with 6828 codes just moved). The loop syncs
    # every counter to the global step each iteration and anchors the
    # schedule to the first point past the VQ warmup; a level inactive at
    # a schedule point fires at its next active batch without shifting
    # the schedule.
    first_reset = _reset_offset(2500, cfg.train.val_interval)
    while first_reset <= v.warmup_steps:
        first_reset += 2500
    for rvq in model.rvqs:
        for lev in rvq.levels:
            if lev.reset_interval > 0:
                lev.next_reset = first_reset

    Ls_full = torch.tensor([sp.depth for sp in specs], device=device)[None]

    if cfg.train.freeze_codec:        # only the selector trains
        model.requires_grad_(False)
        model.eval()                  # disables the VQ EMA update
        if model.selector is not None:
            model.selector.requires_grad_(True)
        print("freeze-codec: codec frozen (eval, no EMA) -- only the "
              "selector trains", flush=True)

    if v.freeze_books:
        n_book = freeze_books(model)
        print(f"vq.freeze_books: {n_book/1e6:.1f}M codebook weights frozen "
              f"(no EMA, no resets, no grad)", flush=True)
    head_names = set()
    if cfg.train.freeze_head_steps > 0:
        head_names = decoder_tail_names(model, cfg.train.freeze_head_blocks)
        set_params_frozen(model, head_names, True)
        print(f"froze {len(head_names)} decoder-tail tensors "
              f"(last {cfg.train.freeze_head_blocks} blocks + head) for "
              f"{cfg.train.freeze_head_steps} steps", flush=True)

    if ddp_world > 1:
        # Absorb prefill skew explicitly before the first real collective:
        # ranks reach this point many minutes apart because each fills its
        # own CPU chip pool from the same array.
        D.barrier()
        # A warm start leaves some modules freshly initialised (the depth
        # cond_mlp), so equal seeds are not a safe guarantee of equal
        # weights -- copy rank 0's state outright. Only AFTER that do we
        # diverge the RNG, so each rank draws different crops and depth
        # tuples; identical streams would make the extra rank pure waste.
        n = D.broadcast_state(model)
        seed_all(cfg.train.seed + 1000 * ddp_rank)
        assert cfg.train.batch_windows % ddp_world == 0, \
            "train.batch_windows must divide by the number of ranks"
        if D.is_main():
            print(f"ddp: {ddp_world} ranks, broadcast {n} tensors from "
                  f"rank 0; global batch {cfg.train.batch_windows} "
                  f"-> {cfg.train.batch_windows // ddp_world} per rank",
                  flush=True)

    # a fresh rate model gets its own (higher) LR so it can learn while a
    # warm-started codec fine-tunes at the small base LR. The optimizer owns
    # every param that trains at ANY point in the run — temporarily frozen
    # params (encoder, decoder tail) must be present from the start or
    # unfreezing them later would never take effect.
    def _trains_eventually(n, p):
        if v.freeze_books and n.startswith("rvqs."):
            return False              # frozen for the whole run
        if cfg.train.freeze_codec:
            return n.startswith("selector.")
        return p.requires_grad or n in head_names or \
            (enc_frozen and n.startswith(("tokenizer.", "stem.")))
    rate_p = [p for n, p in trainable_params(model, _trains_eventually)
              if n.startswith("rate.")]
    base_p = [p for n, p in trainable_params(model, _trains_eventually)
              if not n.startswith("rate.")]
    groups = [{"params": base_p, "lr": cfg.train.optim.lr}] if base_p else []
    if rate_p and cfg.model.rate is not None:
        groups.append({"params": rate_p, "lr": cfg.model.rate.lr})
        print(f"optimizer: {sum(p.numel() for p in rate_p)/1e6:.2f}M "
              f"rate-model params @ lr={cfg.model.rate.lr}", flush=True)
    opt = torch.optim.AdamW(groups, lr=cfg.train.optim.lr,
                            weight_decay=cfg.train.optim.wd)
    warmup, lr0 = cfg.train.optim.warmup, cfg.train.optim.lr0_steps

    const_lr = cfg.train.optim.schedule == "constant"

    def lr_fn(s):
        if s < lr0:                   # AdamW warms its moments at lr=0 first
            return 0.0
        s -= lr0
        ramp = min(1.0, (s + 1) / warmup)
        if const_lr:                  # stoppable at any step
            return ramp
        total = max(cfg.train.steps - lr0, 1)
        return ramp * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    if cfg.train.init_from:
        ck = torch.load(cfg.train.init_from, map_location=device,
                        weights_only=False)
        if "opt" in ck:
            try:
                # moments only -- the lr SCHEDULE stays what this run's
                # config says (resume recipes set their warmup on
                # purpose); Adam's per-param step counters travel with
                # the state, so bias correction does not restart either
                opt.load_state_dict(ck["opt"])
                print(f"restored optimizer state from "
                      f"{cfg.train.init_from}", flush=True)
            except (ValueError, KeyError) as e:
                print(f"optimizer state in {cfg.train.init_from} does not "
                      f"match this model ({e}); starting FRESH -- expect "
                      f"full-lr updates while the moments warm, consider "
                      f"optim.lr0_steps", flush=True)
        else:
            print(f"no optimizer state in {cfg.train.init_from}: fresh "
                  f"AdamW. Zero second moments give every parameter "
                  f"full-lr updates at first (the measured storm "
                  f"trigger); consider optim.lr0_steps to warm the "
                  f"moments at lr=0.", flush=True)
        del ck

    trap = None
    if cfg.train.trap is not None and D.is_main():
        from georvq.train.trap import Trap
        assert cfg.model.vq is not None and cfg.model.vq.clip_norm, \
            "trap needs model.vq.clip_norm for the gnorm"
        trap = Trap(cfg.train.trap, model, run_dir)
        print(f"trap armed: abs>{cfg.train.trap.gnorm_abs} and "
              f"ratio>{cfg.train.trap.gnorm_ratio}x EMA, "
              f"{cfg.train.trap.n_dumps} dumps, hold="
              f"{cfg.train.trap.hold}", flush=True)

    use_wandb = cfg.log.wandb and D.is_main()
    reporter = None
    if use_wandb:
        import wandb
        wandb.init(project=cfg.log.wandb_project, name=cfg.log.run_tag,
                   config=cfg.model_dump(mode="json") | run_metadata(cfg))
        if cfg.log.async_report:
            # figures render on a background worker after the training
            # thread has moved past `step`, so they log against their own
            # monotone metric instead of the global step
            from georvq.train.report import AsyncReporter
            reporter = AsyncReporter()
            wandb.define_metric("figs/step")
            wandb.define_metric("figs/*", step_metric="figs/step")


    select_weight = (cfg.model.conditioning.weight
                     if cfg.model.conditioning is not None
                     and cfg.model.conditioning.kind == "select" else 0.0)
    commit_weight = v.commit_weight
    # batch_windows is the GLOBAL batch: splitting it keeps the effective
    # batch, the LR schedule and the step count identical to a single-GPU
    # run, so extra ranks buy wall-clock rather than changing the recipe.
    batch_full = cfg.train.batch_windows // ddp_world
    batch = batch_full
    _ramp = cfg.train.batch_ramp
    if _ramp is not None:
        assert batch_full % _ramp.divisor == 0, \
            "train.batch_windows must divide by batch_ramp.divisor"
        print(f"batch ramp: {batch_full // _ramp.divisor} windows for the "
              f"first {_ramp.until} steps, then {batch_full}", flush=True)
    align = cfg.data.pool.align_crops
    t0, data_ms = time.time(), 0.0
    best_q = -math.inf if qm.higher_better else math.inf
    nonfinite_steps = 0
    best_is_quantized = False   # warmup selects on the ceiling; see below
    best_ceil_guard = -math.inf if report.quality.higher_better else math.inf
    vs, cbt = {}, {}
    for step in range(1, cfg.train.steps + 1):
        if enc_frozen and step > cfg.train.freeze_encoder_steps:
            set_encoder_frozen(model, False)
            enc_frozen = False
            print(f"step {step}: unfroze encoder (co-adaptation begins)",
                  flush=True)

        if head_names and step > cfg.train.freeze_head_steps:
            set_params_frozen(model, head_names, False)
            print(f"step {step}: unfroze decoder tail", flush=True)
            head_names = set()
        if v.reset_freeze_frac and \
                step == int(cfg.train.steps * v.reset_freeze_frac):
            for rvq in model.rvqs:
                for lev in rvq.levels:
                    lev.reset_interval = 1 << 30
                    lev.next_reset = None    # no pending schedule point
            print(f"step {step}: dead-code resets frozen", flush=True)
        # keep every level's counter on the trainer's clock (see the
        # schedule anchor above the loop)
        for rvq in model.rvqs:
            for lev in rvq.levels:
                lev.step.fill_(step)
        if v.ema_decay_final is not None:
            s0, s1 = v.decay_ramp
            f = min(1.0, max(0.0, (step - s0) / max(s1 - s0, 1)))
            d = 1 - (1 - v.ema_decay) * \
                ((1 - v.ema_decay_final) / (1 - v.ema_decay)) ** f
            for rvq in model.rvqs:
                for lev in rvq.levels:
                    lev.decay = d
        if v.warmup_steps and step == v.warmup_steps + 1:
            with torch.no_grad():
                # Per-tier budgets when init_per_code is set. Sizing the
                # init by batch count couples the tiers: the coarse tier
                # yields 1 vector per window and the finest 256, so a
                # batch count that gives the coarse tier enough data
                # gives the finest tier tens of GiB of it.
                if v.init_per_code:
                    want = [v.init_per_code * sp.codes for sp in specs]
                else:
                    nb = max(v.kmeans_sample // batch, 1)
                    want = [nb * batch * sp.cells for sp in specs]
                samples = [[] for _ in specs]
                got = [0] * len(specs)
                while any(g < w for g, w in zip(got, want)):
                    s_, m_ = sampler.draw(batch, align=align)
                    xs, vs, _ = prep(s_, m_, device)
                    for i, z in enumerate(
                            model.encode_levels(xs, model._stats_of(xs, vs),
                                                vs)):
                        if got[i] >= want[i]:
                            continue
                        zf = z.reshape(-1, z.shape[-1])
                        zf = zf[:want[i] - got[i]]
                        samples[i].append(zf)
                        got[i] += zf.shape[0]
                for i, rvq in enumerate(model.rvqs):
                    zi = torch.cat(samples[i])
                    samples[i] = None
                    residual_init_rvq(rvq, zi, mode=v.init_mode,
                                      iters=v.kmeans_iters,
                                      expected_batch=batch * specs[i].cells)
                    if hasattr(rvq, "project_books"):    # b-bit QAT books
                        rvq.project_books()
                    if hasattr(rvq, "spherize_books"):   # gain-shape books
                        rvq.spherize_books(zi)
                    print(f"  tier {i}: {zi.shape[0]:,} vectors "
                          f"({zi.shape[0]/specs[i].codes:.0f}/codeword)",
                          flush=True)
                    del zi
                torch.cuda.empty_cache()
            print(f"step {step}: codebooks fit via residual "
                  f"{v.init_mode} init", flush=True)

        td = time.time()
        if _ramp is not None:
            batch = (batch_full // _ramp.divisor if step <= _ramp.until
                     else batch_full)
        loc = None
        if needs_loc:
            s, meta = sampler.draw(batch, align=align, return_rows=True)
            loc = train_loc[meta["row"]]
        else:
            s, meta = sampler.draw(batch, align=align)
        x, valid, meta = prep(s, meta, device)
        st = model._stats_of(x, valid)   # None unless stats_cond is on
        data_ms = 0.9 * data_ms + 0.1 * (time.time() - td) * 1000
        amp = torch.autocast(device.type, torch.bfloat16, enabled=amp_on)
        select = torch.zeros((), device=device)
        pt = None
        if v.warmup_steps and step <= v.warmup_steps:
            with amp:
                zs = model.encode_levels(x, st, valid)
                Lw = Ls_full.expand(x.shape[0], -1)
                if v.warmup_truncate and len(specs) > 1:
                    # keep a random PREFIX of tiers, zero the rest. Depth
                    # within a tier is meaningless here (nothing is
                    # quantized), so truncation is per-tier presence --
                    # the same "true informational absence" the stage-1
                    # path uses. Conditioning says full depth for the
                    # tiers that are present and 0 for the absent ones,
                    # which is exactly what the decoder is being handed.
                    keep = torch.randint(1, len(specs) + 1,
                                         (x.shape[0],), device=device)
                    ti = torch.arange(len(specs), device=device)[None]
                    present = ti < keep[:, None]              # (B, tiers)
                    Lw = torch.where(present, Ls_full,
                                     torch.zeros_like(Ls_full))
                    zs = [z * present[:, i].view(-1, 1, 1).to(z.dtype)
                          for i, z in enumerate(zs)]
                out = model.decode(zs, Lw, st, valid)
            out = out.float()
            commit = torch.zeros((), device=device)
            Ls_used = Lw
        else:
            if cfg.model.depth_sampling == "path":
                Ls = sample_path_depths(x.shape[0], specs, device, mins=mins,
                                        full_frac=cfg.model.full_frac,
                                        path_frac=1.0 - cfg.model.full_frac
                                        - v.passthrough_frac)
            else:
                Ls = sample_monotone_depths(x.shape[0], specs, device,
                                            mins=mins,
                                            full_frac=cfg.model.full_frac)
            # quantization pass-through: a slice of the batch is decoded
            # from the raw latents at full depth, so the continuous path
            # keeps training alongside the quantized one. Conditioning is
            # set to full depth FIRST -- the decoder has to be told the
            # tiers are all present, or it compensates for an absence
            # that isn't there.
            pt = None
            if v.passthrough_frac > 0:
                pt = torch.rand(x.shape[0], device=device) \
                    < v.passthrough_frac
                Ls = torch.where(pt[:, None], Ls_full.expand_as(Ls), Ls)
            with amp:
                zs = model.encode_levels(x, st, valid)
            zqs, commit, select = model.quantize(zs, Ls, loc=loc)
            if pt is not None:
                # straight replacement, not straight-through: these rows
                # carry a clean gradient to the encoder
                m3 = pt[:, None, None]
                zqs = [torch.where(m3, z.to(zq.dtype), zq)
                       for z, zq in zip(zs, zqs)]
            with amp:
                out = model.decode(zqs, Ls, st, valid)
            out = out.float()
            Ls_used = Ls
        recon_loss, m = source.recon_stats(out, x, valid)
        # ladder decoder: every exit a row's tuple licenses is supervised
        # (1:1:1); `out` carries the deepest, the shallower ones are here
        aux = getattr(model.decoder, "last_aux", None)
        aux_vals = []
        if aux:
            for rows_i, amap in aux:
                am = amap.float()
                if model.stats_cond is not None \
                        and model.stats_cond.subtract_mean:
                    # decode() uncenters only the returned output; the aux
                    # exits live in the same centred space and must be
                    # restored before scoring against the full field
                    from georvq.model.stats_cond import uncenter
                    am = uncenter(am, st[rows_i].float(), valid[rows_i])
                al, _ = source.recon_stats(am, x[rows_i], valid[rows_i])
                aux_vals.append(al.detach())   # tensor; float at log time
                recon_loss = recon_loss + al
            # mean over exit terms, not sum: keeps the exits at 1:1:1 while
            # holding the loss (and gradient) scale of a single-exit run,
            # so clip_norm keeps its meaning
            recon_loss = recon_loss / (1 + len(aux))
            model.decoder.last_aux = None
        loss = recon_loss + commit_weight * commit + select_weight * select
        # rate term: entropy-model bits/px over a sub-batch (fp32, outside
        # autocast), ramped in over rate.warmup. Shares the encoder with the
        # recon path; the sub-batch bounds the K-way softmax memory.
        rate_bpp = None
        if model.rate is not None and \
                not (v.warmup_steps and step <= v.warmup_steps):
            r = cfg.model.rate
            rb = min(r.batch, x.shape[0])
            rate_bpp = model.train_rate_bits([z[:rb].float() for z in zs],
                                             r.tau)
            rate_w = r.weight * min(1.0, step / max(r.warmup, 1))
            loss = loss + rate_w * rate_bpp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if ddp_world > 1:
            # average (not sum) so the LR keeps its single-GPU meaning
            D.average_grads(base_p + rate_p)
        # clip AFTER the all-reduce, so the norm is the one actually applied
        gnorm = None
        if v.clip_norm:
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                base_p + rate_p, v.clip_norm))
            if trap is not None:
                # observe (and possibly dump) BEFORE the non-finite skip
                # zeroes the grads and before opt.step() moves the
                # weights: a dump holds the exact pre-transition state.
                # Scalars only on stats-cadence or loud steps: each
                # float() is a host sync (clip has already synced once,
                # but ~12 conversions per step still add up)
                tscal = {}
                if (gnorm > 0.5 * trap.cfg.gnorm_abs
                        or not math.isfinite(gnorm)
                        or step % max(trap.cfg.stats_every, 1) == 0):
                    tscal = {"loss": float(loss.detach()),
                             "recon": float(recon_loss.detach()),
                             "commit": float(commit.detach()),
                             "select": float(select.detach()),
                             **{f"aux{i}": float(a)
                                for i, a in enumerate(aux_vals)},
                             **{f"act_{k}": float(av) for k, av in
                                getattr(model.decoder, "act_max",
                                        {}).items()}}
                if trap.observe(step, gnorm, opt, tscal):
                    trap.dump(step, model, opt, sched,
                              batch={"x": x, "valid": valid, "st": st,
                                     "Ls": Ls_used, "loc": loc, "pt": pt},
                              scalars={"gnorm": gnorm, **tscal})
                    if trap.cfg.hold and trap.dumps == 1:
                        # hold only at the first trip (the onset); later
                        # trips in the same storm dump and move on
                        trap.hold({"trap": trap})
            if not math.isfinite(gnorm):
                # a non-finite gradient would write NaN/inf into Adam's
                # moments and poison every later step, so drop it entirely
                opt.zero_grad(set_to_none=True)
                nonfinite_steps += 1
                if nonfinite_steps <= 10 and D.is_main():
                    print(f"step {step}: non-finite grad norm, step SKIPPED "
                          f"({nonfinite_steps} so far)", flush=True)
                sched.step()
                continue
        opt.step(); sched.step()
        if trap is not None:
            trap.after_step(step, model, opt,
                            {"x": x, "valid": valid, "st": st,
                             "Ls": Ls_used, "loc": loc, "pt": pt})

        rst = {}
        for i, rvq in enumerate(model.rvqs):
            for j, lev in enumerate(rvq.levels):
                st = getattr(lev, "last_reset", None)
                if st is None:
                    continue
                lev.last_reset = None
                for k, val in st.items():
                    if k != "step":
                        rst[f"reset/pyr{i}_L{j+1}_{k}"] = val
        if rst:
            n_tot = sum(val for k, val in rst.items()
                        if k.endswith("_n_reset"))
            if use_wandb:
                # per-level detail as ONE figure, not 168 scalar
                # panels; only the aggregates stay as scalars so they
                # are still plottable over time
                fr = (reset_figure(rst, specs, step)
                      if report.has_figure("reset") else None)
                agg = {"reset/total_codes_reset": n_tot,
                       "reset/mean_resid_gain": float(np.nanmean(
                           [v for k, v in rst.items()
                            if k.endswith("_resid_gain")])),
                       "reset/mean_perp_gain": float(np.nanmean(
                           [v for k, v in rst.items()
                            if k.endswith("_perp_gain")]))}
                if fr is not None:
                    agg["figs/reset"] = wandb.Image(fr)
                wandb.log(agg, step=step)
                if fr is not None:
                    plt.close(fr)
            worst = max((val, k) for k, val in rst.items()
                        if k.endswith("_n_reset"))
            print(f"step {step}: reset {n_tot:.0f} codes "
                  f"({v.reset_mode}); most in "
                  f"{worst[1].split('/')[1].rsplit('_', 2)[0]} "
                  f"({worst[0]:.0f})", flush=True)

        if step % 50 == 0 and D.is_main():
            dt = time.time() - t0; t0 = time.time()
            tok_s = 50 * batch * ddp_world / dt
            log = {"train/loss": loss.item(),
                   "train/commit": commit.item(),
                   "train/select": select.item(),
                   "train/lr": sched.get_last_lr()[0],
                   "perf/tokens_per_s": tok_s,
                   "perf/data_wait_ms": data_ms,
                   "perf/pool_chips_loaded": pool.chips_loaded}
            if gnorm is not None:
                # PRE-clip norm: the diagnostic that would have shown the
                # step-1900 blow-up coming
                log["train/grad_norm"] = gnorm
                log["train/grad_clipped"] = float(
                    gnorm > v.clip_norm)
                log["train/nonfinite_steps"] = nonfinite_steps
            log.update({f"train/{k}": float(val) for k, val in m.items()})
            # per-stage activation maxima (ladder): the latch precursor
            for k_, v_ in getattr(model.decoder, "act_max", {}).items():
                log[f"train/act_max_{k_}"] = float(v_)
            for i_, a_ in enumerate(aux_vals):
                log[f"train/aux_exit{i_}"] = float(a_)
            if pool.cycle:
                log["data/epochs"] = pool.epochs
                log["data/tiles_streamed"] = pool.tiles_streamed
                log["data/reuse_per_patch"] = pool.reuse_estimate
            for i, rvq in enumerate(model.rvqs):
                log[f"rvq/pyr{i}_ema_usage"] = float(rvq.ema_usage())
            if rate_bpp is not None:
                log["train/rate_bpp"] = rate_bpp.item()
                log["train/rate_w"] = rate_w
            if step % 200 == 0:
                rstr = (f" rate={rate_bpp.item():.2f}bpp"
                        if rate_bpp is not None else "")
                cstr = (f" {qm.key}={m[qm.key].item():.4f}"
                        if qm.key in m else "")
                print(f"step {step}: loss={loss.item():.5f}{cstr}{rstr} "
                      f"tok/s={tok_s:,.0f}", flush=True)
            if use_wandb:
                wandb.log(log, step=step)

        if (step % cfg.train.val_interval == 0
                or step == cfg.train.steps) and D.is_main():
            canon_ts = list(canon.values())
            if step == cfg.train.steps and len(tuples) <= 64:
                val_ts = tuples           # full enumeration at the end
            else:
                others = [t for t in tuples if t not in canon_ts]
                vrng = np.random.default_rng(step)
                k = min(max(cfg.train.val_tuples - len(canon_ts), 0),
                        len(others))
                val_ts = canon_ts + [others[i] for i in
                                     vrng.choice(len(others), k,
                                                 replace=False)]
            canon_names = {tname(t) for t in canon.values()}
            # During the VQ-free warmup the books are still whatever
            # nn.Embedding was initialised with, so every quantized tuple
            # would score the model against random vectors. Measure the
            # ceiling only -- it is the sole meaningful number until the
            # books are fitted, and it also keeps model_best.pt from
            # tracking noise.
            vq_live = not (v.warmup_steps and step <= v.warmup_steps)
            vs, cbt = validate(model, source, val_s, val_ts, canon_names,
                               val_loc=val_loc, quantized=vq_live,
                               spec=report, prep=prep)
            if cfg.train.freeze_codec:    # keep the codec frozen (no EMA)
                model.eval()
            if v.freeze_books:            # validate() leaves the model in
                freeze_books(model)       # train() -- re-assert the freeze
            ebt = vs.pop("_extra", {})
            extra_str = "".join(
                f" {m.key}_ceil={vs[f'{m.key}_ceiling']:.4f}"
                for m in report.extra if f"{m.key}_ceiling" in vs)
            base_str = "".join(
                f" {b.key}={vs[f'{qm.key}_{b.key}']:.4f}"
                for b in report.baselines if f"{qm.key}_{b.key}" in vs)
            print(f"step {step} VAL[{qm.key}]: "
                  + ("" if vq_live else "[warmup: ceiling only] ")
                  + " ".join(f"{k}({tname(t)})={cbt[tname(t)]:.4f}"
                             for k, t in canon.items() if tname(t) in cbt)
                  + f" ceil={vs[f'{qm.key}_ceiling']:.4f}"
                  + extra_str + base_str,
                  flush=True)
            if use_wandb:
                scalars = {f"val/{qm.key}_{k}": cbt[tname(t)]
                           for k, t in canon.items() if tname(t) in cbt}
                scalars.update({k: val for k, val in vs.items()
                                if k.startswith("sel_")})
                for mk, by_t in ebt.items():
                    scalars.update({f"val/{mk}_{k}": by_t[tname(t)]
                                    for k, t in canon.items()
                                    if tname(t) in by_t})
                # per-(tier, level) codebook health as SCALARS, not just
                # the figure. rvq/pyr{i}_ema_usage is a poor proxy: it
                # counts entries above the dead threshold, so a book whose
                # mass is concentrated in a few codes still reads ~100%
                # while its effective size is a fraction of K. Perplexity
                # is the honest number.
                # per-level codebook health stays in figs/codebook; as
                # scalars it is 72 more panels for no extra information
                # only the ceilings and baselines this source declares:
                # a scalar field publishes r2_ceiling and no block-mean,
                # rather than inheriting an embedding field's panels
                for m in report.metrics:
                    if f"{m.key}_ceiling" in vs:
                        scalars[f"val/{m.key}_ceiling"] = \
                            vs[f"{m.key}_ceiling"]
                for b in report.baselines:
                    bk = f"{qm.key}_{b.key}"
                    if bk in vs:
                        scalars[f"val/{bk}"] = vs[bk]
                # Every figure below is a view of QUANTIZED behaviour --
                # the R-D front, per-level codebook health, reconstructions
                # at depth tuples. During the VQ-free warmup they would all
                # be drawn against unfitted books, so log the scalars and
                # skip them.
                if not vq_live:
                    wandb.log(scalars, step=step)
                else:
                    sb = model.side_bits()
                    fig_step = (step % cfg.train.fig_interval == 0
                                or step == cfg.train.steps)
                    rt = ([[int(x_) for x_ in t.split("-")]
                           for t in cfg.eval.recon_tuples]
                          if cfg.eval.recon_tuples else None)
                    # HARVEST: every decode the figures need runs now,
                    # while the model is at hand; the payloads are plain
                    # subsampled CPU tensors and the model is not touched
                    # again after this point
                    payloads = {}
                    if report.has_figure("recon") and fig_step:
                        payloads["recon"] = recon_harvest(
                            model, source, val_s, mins=mins, tuples=rt,
                            val_loc=val_loc)
                    if (report.has_figure("rd_err") and fig_step
                            and source.normalizer(cfg.model) is not None):
                        payloads["rd_err"] = rd_err_harvest(
                            model, source, val_s, val_ts,
                            px=cfg.model.window ** 2, side_bits=sb,
                            val_loc=val_loc)
                    if report.has_figure("error_hist") and fig_step:
                        payloads["hist"] = error_hist_harvest(
                            model, source, val_s, tuples=rt, mins=mins,
                            val_loc=val_loc)
                    # host RSS per val: two 60k runs died on silent ~1
                    # GB/val growth (glibc arena bloat under the report
                    # worker's large transient allocs); this makes any
                    # recurrence a visible slope instead of an OOM kill
                    try:
                        with open("/proc/self/status") as f:
                            for ln in f:
                                if ln.startswith("VmRSS"):
                                    scalars["perf/rss_gb"] = \
                                        int(ln.split()[1]) / 2 ** 20
                                    break
                    except OSError:
                        pass
                    wandb.log(scalars, step=step)  # scalars stay sync

                    # RENDER: CPU only (percentiles, matplotlib, upload).
                    # With async_report it runs on the worker while the
                    # GPU goes back to training; otherwise inline.
                    vs_snap, cbt_snap = dict(vs), dict(cbt)
                    ts_snap = [list(t) for t in val_ts]

                    def _render(step=step, payloads=payloads, vs=vs_snap,
                                cbt=cbt_snap, val_ts=ts_snap, sb=sb,
                                deferred=reporter is not None):
                        figs = {"figs/step": step} if deferred else {}
                        open_figs = []

                        def add(key, f):
                            figs[key] = wandb.Image(f)
                            open_figs.append(f)
                        if report.has_figure("rd"):
                            add("figs/rd", rd_figure(
                                cbt, vs, specs, val_ts, step, report,
                                ref_fronts=ref_fronts,
                                px=cfg.model.window ** 2, side_bits=sb))
                        if report.has_figure("rd_log"):
                            add("figs/rd_log", rd_figure(
                                cbt, vs, specs, val_ts, step, report,
                                ref_fronts=ref_fronts,
                                px=cfg.model.window ** 2, dist_log=True,
                                side_bits=sb))
                        if report.has_figure("codebook"):
                            add("figs/codebook", codebook_figure(
                                vs, specs, step))
                        if "cosdist" in vs and report.has_figure("cosdist"):
                            add("figs/cosdist_by_level", cosdist_figure(
                                vs["cosdist"], None, specs, step))
                        if "recon" in payloads:
                            add("figs/recon", recon_render(
                                payloads["recon"], step,
                                preview=report.preview))
                        if "rd_err" in payloads:
                            # rate vs physical-unit error (MAE/RMSE/
                            # p90/p99) over the R-D front's points
                            add("figs/rd_err", rd_err_render(
                                payloads["rd_err"], step,
                                unit=report.units))
                        if "hist" in payloads:
                            add("figs/error_hist", error_hist_render(
                                payloads["hist"], step, unit=report.units))
                        if deferred:
                            wandb.log(figs)
                        else:
                            wandb.log(figs, step=step)
                        for f in open_figs:
                            plt.close(f)
                    if reporter is not None:
                        reporter.submit(_render)
                    else:
                        _render()

            # Collapse guard: a stage latch is a catastrophic ceiling drop
            # that never recovers; trip on the FIRST validation past it.
            ceil_q = vs[f"{qm.key}_ceiling"]
            if cfg.train.guard_drop is not None:
                worse = (best_ceil_guard - ceil_q if qm.higher_better
                         else ceil_q - best_ceil_guard)
                if worse > cfg.train.guard_drop:
                    save_checkpoint(
                        os.path.join(run_dir, "model_collapsed.pt"),
                        model, cfg, step,
                        {"quality_key": qm.key,
                         f"{qm.key}_ceiling": ceil_q,
                         "best_ceiling": best_ceil_guard})
                    print(f"step {step}: COLLAPSE GUARD TRIPPED -- "
                          f"{qm.key}_ceiling {ceil_q:.4f} vs running best "
                          f"{best_ceil_guard:.4f} (guard_drop="
                          f"{cfg.train.guard_drop}). Forensic state in "
                          f"model_collapsed.pt; resume from model_best.pt "
                          f"or the last periodic save.", flush=True)
                    raise SystemExit(3)
                best_ceil_guard = (max(best_ceil_guard, ceil_q)
                                   if qm.higher_better
                                   else min(best_ceil_guard, ceil_q))

            # Track the ceiling during warmup: cos_full would be the
            # model scored against unfitted books, so model_best.pt would
            # be selected on noise. The two scales differ, so reset the
            # running best when the books go live and the criterion
            # switches.
            if vq_live and not best_is_quantized:
                best_q = -math.inf if qm.higher_better else math.inf
                best_is_quantized = True
            full_q = (cbt[tname(canon["full"])] if vq_live
                      else vs[f"{qm.key}_ceiling"])
            metrics = {"quality_key": qm.key,
                       f"{qm.key}_full": full_q,
                       f"{qm.key}_ceiling": vs[f"{qm.key}_ceiling"]}
            improved = (full_q > best_q if qm.higher_better
                        else full_q < best_q)
            if improved:
                best_q = full_q
                save_checkpoint(os.path.join(run_dir, "model_best.pt"),
                                model, cfg, step, metrics,
                                prep_state=prep.state())
        if (step % cfg.train.save_interval == 0
                or step == cfg.train.steps) and D.is_main():
            save_checkpoint(os.path.join(run_dir, "model.pt"), model, cfg,
                            step, {"quality_key": qm.key,
                                   f"{qm.key}_full": best_q},
                            opt=opt if cfg.train.save_optim else None,
                            sched=sched if cfg.train.save_optim else None,
                            prep_state=prep.state())

    if D.is_main():
        # one writer: every rank reaching this line would race on the
        # same path, and the ranks do not all hold the same `vs`
        with open(os.path.join(run_dir, "final_val.json"), "w") as f:
            blob = {"scalars": {k: val for k, val in vs.items()
                                if not k.startswith("_")},
                    "quality_key": qm.key,
                    "quality_by_tuple": cbt}
            if qm.key == "cos":
                # readers of pre-report runs (load_rd_ref, analysis/) key
                # off this name; keep it where it still means the same
                blob["cos_by_tuple"] = cbt
            json.dump(blob, f, indent=1)
    if reporter is not None and not reporter.close():
        print("warning: report worker did not flush before timeout",
              flush=True)
    pool.shutdown()
    D.cleanup()
    print(f"done -> {run_dir}", flush=True)
    return run_dir
