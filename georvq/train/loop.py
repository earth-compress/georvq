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

from georvq.config import Config
from georvq.data.pool import ChipPool, fixed_val_tokens
from georvq.data.store import ChipStore
from georvq.factory import build_model
from georvq.model.codec import (monotone_tuples, path_tuples,
                                sample_monotone_depths,
                                sample_path_depths, tuple_bits)
from georvq.model.vq import kmeans_init_rvq
from georvq.model.vqinit import residual_init_rvq
from georvq.repro import (run_metadata, save_checkpoint, seed_all,
                          write_resolved_config)
from georvq.train.figures import (codebook_figure, cosdist_figure,
                                   rd_figure, recon_figure, refit_figure,
                                   reset_figure)
from georvq.train.validate import canonical_tuples, load_rd_ref, tname, validate
from georvq.train import dist as D
from georvq.train.drift import DriftProbe
from georvq.train.refit import collect_latents, refit_phase
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
    offset by rank so ranks hold and sample different chips."""
    store = ChipStore(cfg.data.store_dir)
    train_rows, val_rows = store.split(cfg.data.val_frac,
                                       seed=cfg.data.split_seed)
    if D.is_main():
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
    tuples = [t for t in monotone_tuples(specs)
              if all(L >= m for L, m in zip(t, mins))]
    canon = canonical_tuples(specs, mins)
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
    store, train_rows, val_rows, val_s, pool = setup_data(cfg, source, device)
    val_rows.to_parquet(os.path.join(run_dir, "val_chips.parquet"))

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
    if cfg.train.init_from and not v.mature_warmstart:
        # undo load_partial's maturing: keep the books live at the
        # configured decay with resets (and therefore reset_mode) active
        for rvq in model.rvqs:
            for lev in rvq.levels:
                lev.decay = v.ema_decay
                lev.reset_interval = 2500
                # Zero the inherited step counters. They come from the
                # source checkpoint -- already tens of thousands of steps
                # in, and desynchronised across levels by skip_masked --
                # so resets would fire at arbitrary offsets in this run.
                # Observed: resets at 464 / 803 / 2963, the last landing 37
                # steps before the validation at 3000, which then scored
                # books with 6828 codes just moved (cos 0.9572 -> 0.9550).
                lev.step.fill_(2500 - _reset_offset(2500,
                                                    cfg.train.val_interval))
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
                s = pool.sample_batch(
                    min(cfg.train.batch_windows, max(need)),
                    cfg.model.window, align=cfg.data.pool.align_crops)
                x0, v0 = source.prep(s, device)
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

    use_wandb = cfg.log.wandb and D.is_main()
    if use_wandb:
        import wandb
        wandb.init(project=cfg.log.wandb_project, name=cfg.log.run_tag,
                   config=cfg.model_dump(mode="json") | run_metadata(cfg))

    probe = None
    if v.drift_interval:
        # a FIXED batch, drawn once: drift must be measured on the same
        # windows every time or it is confounded with sampling noise
        # NB: `batch` / `align` are bound below, so read the config here
        px = pool.sample_batch(
            min(v.drift_windows, cfg.train.batch_windows), cfg.model.window,
            align=cfg.data.pool.align_crops)
        probe_x, _ = source.prep(px, device)
        probe = DriftProbe(
            probe_x, v.drift_interval,
            tier_codes=[sp.codes for sp in specs],
            vecs_per_step=[(cfg.train.batch_windows // max(ddp_world, 1))
                           * sp.cells for sp in specs])
        print(f"drift probe: {probe_x.shape[0]} fixed windows every "
              f"{v.drift_interval} steps", flush=True)

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
    best_cos = -1.0
    nonfinite_steps = 0
    best_is_quantized = False   # warmup selects on the ceiling; see below
    vs, cbt = {}, {}
    for step in range(1, cfg.train.steps + 1):
        if enc_frozen and step > cfg.train.freeze_encoder_steps:
            set_encoder_frozen(model, False)
            enc_frozen = False
            print(f"step {step}: unfroze encoder (co-adaptation begins)",
                  flush=True)
        # ---- periodic offline refit phase ----------------------------
        # Fires one step AFTER a validation, so a validation never scores
        # a book that was replaced moments earlier.
        if v.refit_interval and step % v.refit_interval == v.refit_offset:
            t_rf = time.time()
            model.eval()
            zs_fit, n_b = collect_latents(model, source, pool, cfg, device,
                                          per_code=v.refit_per_code)
            model.train()
            if v.freeze_books:
                freeze_books(model)
            rst = refit_phase(model, zs_fit, cfg,
                              lloyd_iters=v.refit_lloyd_iters,
                              dead_frac=v.refit_dead_frac,
                              recycle_frac=v.refit_recycle_frac)
            n_rec = sum(val for k, val in rst.items()
                        if k.endswith("_n_recycled"))
            dt_rf = time.time() - t_rf
            if D.is_main():
                print(f"step {step}: REFIT on {n_b} batches "
                      f"({sum(z.shape[0] for z in zs_fit):,} vectors), "
                      f"{n_rec:.0f} codes recycled, {dt_rf:.0f}s",
                      flush=True)
            if use_wandb:
                ff = refit_figure(rst, specs, step)
                wandb.log({"figs/refit": wandb.Image(ff),
                           "refit/total_recycled": n_rec,
                           "refit/seconds": dt_rf,
                           "refit/vectors": sum(z.shape[0] for z in zs_fit)},
                          step=step)
                plt.close(ff)
            del zs_fit
            torch.cuda.empty_cache()
            t0 = time.time()          # don't bill the refit to tokens/s

        if head_names and step > cfg.train.freeze_head_steps:
            set_params_frozen(model, head_names, False)
            print(f"step {step}: unfroze decoder tail", flush=True)
            head_names = set()
        if v.reset_freeze_frac and \
                step == int(cfg.train.steps * v.reset_freeze_frac):
            for rvq in model.rvqs:
                for lev in rvq.levels:
                    lev.reset_interval = 1 << 30
            print(f"step {step}: dead-code resets frozen", flush=True)
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
                    xs, vs = source.prep(
                        pool.sample_batch(batch, cfg.model.window, align=align),
                        device)
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
            s, rows = pool.sample_batch(batch, cfg.model.window, align=align,
                                        return_rows=True)
            loc = train_loc[rows]
        else:
            s = pool.sample_batch(batch, cfg.model.window, align=align)
        x, valid = source.prep(s, device)
        st = model._stats_of(x, valid)   # None unless stats_cond is on
        data_ms = 0.9 * data_ms + 0.1 * (time.time() - td) * 1000
        amp = torch.autocast(device.type, torch.bfloat16, enabled=amp_on)
        select = torch.zeros((), device=device)
        if v.warmup_steps and step <= v.warmup_steps:
            with amp:
                zs = model.encode_levels(x, st, valid)
                out = model.decode(zs, Ls_full.expand(x.shape[0], -1),
                                   st, valid)
            out = out.float()
            commit = torch.zeros((), device=device)
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
        recon_loss, m = source.recon_stats(out, x, valid)
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
        if cfg.train.optim.grad_clip > 0:
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                base_p + rate_p, cfg.train.optim.grad_clip))
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

        if probe is not None:
            dstat = probe(model, step)
            if dstat and use_wandb:
                wandb.log(dstat, step=step)
            if dstat and step % (v.drift_interval * 10) == 0 and D.is_main():
                k0 = "drift/pyr0_per_step_over_sigma"
                w0 = "drift/pyr0_optimal_ema_decay"
                if k0 in dstat:
                    print(f"step {step}: drift {dstat[k0]:.2e} sigma/step "
                          f"(tier0), optimal ema_decay "
                          f"{dstat.get(w0, float('nan')):.5f}", flush=True)

        # reset diagnostics: emitted only on steps where a reset fired
        if D.is_main() and not v.freeze_books:
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
                    fr = reset_figure(rst, specs, step)
                    agg = {"reset/total_codes_reset": n_tot,
                           "reset/mean_resid_gain": float(np.nanmean(
                               [v for k, v in rst.items()
                                if k.endswith("_resid_gain")])),
                           "reset/mean_perp_gain": float(np.nanmean(
                               [v for k, v in rst.items()
                                if k.endswith("_perp_gain")])),
                           "figs/reset": wandb.Image(fr)}
                    wandb.log(agg, step=step)
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
                    gnorm > cfg.train.optim.grad_clip)
                log["train/nonfinite_steps"] = nonfinite_steps
            log.update({f"train/{k}": float(val) for k, val in m.items()})
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
                cstr = (f" cos={m['cos'].item():.4f}" if "cos" in m else "")
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
                               val_loc=val_loc, quantized=vq_live)
            if cfg.train.freeze_codec:    # keep the codec frozen (no EMA)
                model.eval()
            if v.freeze_books:            # validate() leaves the model in
                freeze_books(model)       # train() -- re-assert the freeze
            ccbt = vs.pop("_ccos", {})
            print(f"step {step} VAL: "
                  + ("" if vq_live else "[warmup: ceiling only] ")
                  + " ".join(f"{k}({tname(t)})={cbt[tname(t)]:.4f}"
                             for k, t in canon.items() if tname(t) in cbt)
                  + f" ceil={vs['cos_ceiling']:.4f}"
                  + f" ccos_ceil={vs['ccos_ceiling']:.3f}"
                  + f" blockmean={vs['cos_blockmean']:.4f}"
                  + (f" ccos_full={ccbt.get(tname(canon['full']), 0):.3f}"
                     if vq_live else ""),
                  flush=True)
            if use_wandb:
                scalars = {f"val/cos_{k}": cbt[tname(t)]
                           for k, t in canon.items() if tname(t) in cbt}
                scalars.update({k: val for k, val in vs.items()
                                if k.startswith("sel_")})
                scalars.update({f"val/ccos_{k}": ccbt[tname(t)]
                                for k, t in canon.items()
                                if tname(t) in ccbt})
                # per-(tier, level) codebook health as SCALARS, not just
                # the figure. rvq/pyr{i}_ema_usage is a poor proxy: it
                # counts entries above the dead threshold, so a book whose
                # mass is concentrated in a few codes still reads ~100%
                # while its effective size is a fraction of K. Perplexity
                # is the honest number.
                # per-level codebook health stays in figs/codebook; as
                # scalars it is 72 more panels for no extra information
                scalars["val/cos_ceiling"] = vs["cos_ceiling"]
                scalars["val/ccos_ceiling"] = vs["ccos_ceiling"]
                scalars["val/cos_blockmean"] = vs["cos_blockmean"]
                # Every figure below is a view of QUANTIZED behaviour --
                # the R-D front, per-level codebook health, reconstructions
                # at depth tuples. During the VQ-free warmup they would all
                # be drawn against unfitted books, so log the scalars and
                # skip them.
                if not vq_live:
                    wandb.log(scalars, step=step)
                else:
                    sb = model.side_bits()
                    f1 = rd_figure(cbt, vs, specs, val_ts, step,
                                   ref_fronts=ref_fronts,
                                   px=cfg.model.window ** 2, side_bits=sb)
                    f1b = rd_figure(cbt, vs, specs, val_ts, step,
                                    ref_fronts=ref_fronts,
                                    px=cfg.model.window ** 2, dist_log=True,
                                    side_bits=sb)
                    f2 = codebook_figure(vs, specs, step)
                    scalars["figs/rd"] = wandb.Image(f1)
                    scalars["figs/rd_log"] = wandb.Image(f1b)
                    scalars["figs/codebook"] = wandb.Image(f2)
                    f4 = None
                    if "cosdist" in vs:
                        f4 = cosdist_figure(vs["cosdist"], None, specs, step)
                        scalars["figs/cosdist_by_level"] = wandb.Image(f4)
                    wandb.log(scalars, step=step)
                    plt.close(f1); plt.close(f1b); plt.close(f2)
                    if f4 is not None:
                        plt.close(f4)
                    if step % cfg.train.fig_interval == 0 \
                            or step == cfg.train.steps:
                        rt = ([[int(x_) for x_ in t.split("-")]
                               for t in cfg.eval.recon_tuples]
                              if cfg.eval.recon_tuples else None)
                        f3 = recon_figure(model, source, val_s, step,
                                          mins=mins, tuples=rt,
                                          val_loc=val_loc)
                        wandb.log({"figs/recon": wandb.Image(f3)}, step=step)
                        plt.close(f3)

            # Track the ceiling during warmup: cos_full would be the
            # model scored against unfitted books, so model_best.pt would
            # be selected on noise. The two scales differ, so reset the
            # running best when the books go live and the criterion
            # switches.
            if vq_live and not best_is_quantized:
                best_cos, best_is_quantized = -1.0, True
            full_cos = (cbt[tname(canon["full"])] if vq_live
                        else vs["cos_ceiling"])
            metrics = {"cos_full": full_cos,
                       "cos_ceiling": vs["cos_ceiling"]}
            if full_cos > best_cos:
                best_cos = full_cos
                save_checkpoint(os.path.join(run_dir, "model_best.pt"),
                                model, cfg, step, metrics)
        if (step % cfg.train.save_interval == 0
                or step == cfg.train.steps) and D.is_main():
            save_checkpoint(os.path.join(run_dir, "model.pt"), model, cfg,
                            step, {"cos_full": best_cos})

    with open(os.path.join(run_dir, "final_val.json"), "w") as f:
        json.dump({"scalars": {k: val for k, val in vs.items()
                               if not k.startswith("_")},
                   "cos_by_tuple": cbt}, f, indent=1)
    pool.shutdown()
    print(f"done -> {run_dir}", flush=True)
    return run_dir
