"""The pyramid codec: multi-scale latent tiers, per-tier residual VQ,
matryoshka variable-rate training, attention decoder.

`PyramidCodec` is a fixed composition — optional capabilities are the
presence of optional submodules (rate model, conditioner, selector), chosen
by config, never runtime flag routing:

    source.stem -> ViTTokenizer -> per-tier ResidualVQ -> PyramidAttnDecoder
                                   [conditioner]  [selector]  [rate model]

A trained model decodes at any monotone per-tier depth tuple: training
samples random tuples (`sample_monotone_depths`) and the decoder is
conditioned on the tuple, so one checkpoint serves the whole R-D front.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.config import ModelConfig, TierSpec  # noqa: F401  (re-exported)
from georvq.model.decoder import PyramidAttnDecoder
from georvq.model.layers import thermometer
from georvq.model.rate import PyramidRateModel
from georvq.model.tokenizer import ViTTokenizer
from georvq.model.attnmask import build_local_mask
from georvq.model.vq import ResidualVQ


# --------------------------------------------------------------------------
# matryoshka depth-tuple utilities
# --------------------------------------------------------------------------

def sample_monotone_depths(batch, specs, device, mins=None, full_frac=0.0):
    """(B, n_tiers) ints, monotone non-increasing; optional per-tier
    minimum depths via `mins`; full_frac of rows forced to the full
    cascade (extra training traffic for the deepest RVQ levels)."""
    mins = mins or [1] + [0] * (len(specs) - 1)
    Ls = torch.empty(batch, len(specs), dtype=torch.long, device=device)
    Ls[:, 0] = torch.randint(mins[0], specs[0].depth + 1, (batch,),
                             device=device)
    for i in range(1, len(specs)):
        lo = mins[i]
        hi = Ls[:, i - 1].clamp(max=specs[i].depth).clamp_min(lo)
        Ls[:, i] = (lo + (torch.rand(batch, device=device)
                          * (hi - lo + 1).float()).long()) \
            .clamp(max=specs[i].depth)
    if full_frac > 0:
        full, prev = [], specs[0].depth
        for sp in specs:
            prev = min(prev, sp.depth); full.append(prev)
        sel = torch.rand(batch, device=device) < full_frac
        Ls[sel] = torch.tensor(full, device=device)
    return Ls


def path_tuples(specs, mins=None):
    """The canonical monotone truncation PATH: add one codebook at a time.

    Fill the coarsest tier to its depth, then the next, and so on --
    (1,0,0), (2,0,0) ... (L,0,0), (L,1,0) ... (L,L,0), (L,L,1) ... So each
    step adds exactly one book, and the path has sum(depths) points.

    `mins[0]` sets where the path STARTS -- there is no point training a
    1-0-0 point, which spends 8 bits on a 64x64 patch of 128-d vectors.

    Sampling uniformly along this path gives every codebook an equal share
    of the training signal. Nested sampling does not: it draws the coarse
    depth uniformly and each finer tier conditionally, which makes deep
    fine-tier levels vanishingly rare -- measured at P(active)=0.006 for
    tier-2 level 14, i.e. those books train on 0.6% of rows.
    """
    mins = mins or [0] * len(specs)
    cur = [mins[i] for i in range(len(specs))]
    out = []
    if cur[0] > 0:
        # include the starting point itself, so mins[0]=4 means the path
        # BEGINS at (4,0,0) rather than at (5,0,0)
        out.append(tuple(cur))
    for i, sp in enumerate(specs):
        while cur[i] < sp.depth:
            cur[i] += 1
            # keep the tuple monotone: a finer tier can never exceed a
            # coarser one, so clamp downstream entries as we grow
            t = list(cur)
            for j in range(1, len(t)):
                t[j] = min(t[j], t[j - 1])
            out.append(tuple(t))
    return out


def sample_path_depths(batch, specs, device, mins=None, full_frac=0.15,
                       path_frac=0.70):
    """Depth tuples: `full_frac` at full depth, the rest spread uniformly
    along the monotone path so every codebook trains equally often.

    (The remaining 1 - full_frac - path_frac of the batch is left for the
    caller's unquantized pass-through rows, which never reach here.)
    """
    path = path_tuples(specs, mins)
    P = torch.tensor(path, device=device, dtype=torch.long)
    full = torch.tensor([sp.depth for sp in specs], device=device,
                        dtype=torch.long)
    u = torch.rand(batch, device=device)
    denom = max(full_frac + path_frac, 1e-9)
    is_full = u < (full_frac / denom)
    pick = torch.randint(0, P.shape[0], (batch,), device=device)
    Ls = P[pick]
    Ls[is_full] = full
    return Ls


def monotone_tuples(specs):
    """All (L_0..L_n) with L_0>=1 and 0 <= L_i <= min(L_{i-1}, d_i)."""
    tuples = [[L] for L in range(1, specs[0].depth + 1)]
    for sp in specs[1:]:
        tuples = [t + [L] for t in tuples
                  for L in range(0, min(t[-1], sp.depth) + 1)]
    return tuples


def tuple_bits(specs, ls, side_bits=0.0):
    """Nominal (uncoded) bits/window for one depth tuple.

    `side_bits` is any fixed per-window side channel outside the pyramid's
    index stream -- currently the stats-conditioning mean. It is charged at
    EVERY depth, including the coarsest, because it is always transmitted;
    leaving it out was how the ring-space S2 work ended up with a stats
    channel silently taking 34% of the bitstream at L=1."""
    return float(sum(sp.bits(L) for sp, L in zip(specs, ls))) + side_bits


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------

class PyramidCodec(nn.Module):
    """Multi-scale latent pyramid codec over a (window x window) input."""

    def __init__(self, cfg: ModelConfig, source):
        super().__init__()
        self.cfg = cfg
        self.window = cfg.window
        self.bands = cfg.bands
        self.specs = list(cfg.tiers)
        self.norm_latents = cfg.norm_latents
        self.continuous = cfg.vq is None
        self.norm_latents = cfg.norm_latents

        self.stem = source.build_stem(cfg)
        if cfg.local_attn and cfg.local_attn[-1] is not None:
            sr = cfg.local_attn[-1]
        self.tokenizer = ViTTokenizer(
            self.specs, cfg.window, cfg.bands, dim=cfg.tokenizer.dim,
            depth=cfg.tokenizer.depth, ff_ratio=cfg.ff_ratio,
            late_join=cfg.tokenizer.late_join)
        if cfg.local_attn is not None:
            m = build_local_mask(self.specs, cfg.window, cfg.local_attn)
            # computed once; non-persistent so checkpoints stay unchanged
            self.tokenizer.register_buffer("attn_mask", m, persistent=False)

        if not self.continuous:
            v = cfg.vq
            from functools import partial
            if v.book_bits:
                from georvq.model.vq_quant import QuantizedResidualVQ
                rvq_cls = partial(QuantizedResidualVQ, book_bits=v.book_bits)
            else:
                rvq_cls = ResidualVQ
            self.rvqs = nn.ModuleList([
                rvq_cls(num_levels=sp.depth, num_codes=sp.codes,
                        dim=sp.dim, decay=v.ema_decay,
                        dead_threshold=v.dead_threshold,
                        grad_codebook=v.grad_codebook,
                        bf16_scores=v.bf16_scores,
                        reset_mode=v.reset_mode)
                for sp in self.specs])
            self._skip_masked = v.skip_masked
            self._base_dead = v.dead_threshold
            self._part_ema = {}      # (tier, level) -> EMA of active fraction
            for i, rvq in enumerate(self.rvqs):
                for j, lev in enumerate(rvq.levels):
                    lev._tag = f"tier{i}/L{j}"   # for reset telemetry

        self.rate = None
        if cfg.rate is not None:
            self.rate = PyramidRateModel(self.specs, model_dim=cfg.rate.model_dim,
                                         n_blocks=cfg.rate.blocks,
                                         ctx_dim=cfg.rate.ctx_dim)

        # conditioning submodules (predict-and-subtract / codebook select)
        self.conditioner = None
        self.selector = None
        if cfg.conditioning is not None:
            from georvq.model.conditioning import build_conditioning
            build_conditioning(self, cfg.conditioning)

        self.stats_cond = cfg.stats_cond
        if cfg.stats_cond is not None:
            from georvq.model.stats_cond import StatsEmbed
            sw = cfg.bands * (2 if cfg.stats_cond.use_std else 1)
            self.stats_enc = (StatsEmbed(sw, cfg.tokenizer.dim)
                              if cfg.stats_cond.cond_encoder else None)
            self.stats_dec = (StatsEmbed(sw, cfg.decoder.dim)
                              if cfg.stats_cond.cond_decoder else None)
        else:
            self.stats_enc = self.stats_dec = None

        if cfg.decoder.kind == "tiersum":
            assert cfg.decoder.map_px is not None, \
                "decoder.kind=tiersum requires decoder.map_px"
            assert cfg.sigma is None and cfg.local_attn is None, \
                "tiersum supports neither stage-1 sigma mode nor local attention"
            assert cfg.depth_sampling == "path", \
                "tiersum groups rows by tier-presence prefix, which only " \
                "fill-order (depth_sampling=path) training guarantees"
            from georvq.model.tiersum import TierSumDecoder
            self.decoder = TierSumDecoder(
                self.specs, cfg.window, cfg.bands, dim=cfg.decoder.dim,
                depth=cfg.decoder.depth, heads=cfg.decoder.heads,
                ff_ratio=cfg.ff_ratio, map_px=cfg.decoder.map_px,
                solo_hidden=cfg.decoder.solo_hidden,
                solo_layers=cfg.decoder.solo_layers)
        else:
            head_in = PyramidAttnDecoder.head_in_dim(
                self.specs, cfg.decoder.dim, cfg.decoder.cls_bridge)
            self.decoder = PyramidAttnDecoder(
                self.specs, head=source.build_head(cfg, head_in),
                dim=cfg.decoder.dim, depth=cfg.decoder.depth,
                heads=cfg.decoder.heads, cls_bridge=cfg.decoder.cls_bridge,
                ff_ratio=cfg.ff_ratio, sigma_cond=cfg.sigma is not None,
                sigma_freqs=cfg.sigma.freqs if cfg.sigma else 8,
                late_join=cfg.decoder.late_join)
        if cfg.local_attn is not None:
            self.decoder.register_buffer(
                "attn_mask", build_local_mask(self.specs, cfg.window,
                                              cfg.local_attn),
                persistent=False)

    # ---- encode / decode ----

    def encode_levels(self, x, stats=None, valid=None):
        """x: (B, C, W, W) -> list of per-tier latents [(B, cells, dim)].

        With `norm_latents`, tiers come out in the SAME normalized
        geometry `encode_stage1` produces, so a stage-2 run quantizes and
        decodes exactly what stage-1 trained on. Without it, a decoder
        warm-started from stage-1 is handed raw latents whose norms are a
        fraction of sqrt(dim) (measured on w32hd: 5.3/6.0/9.7 vs 19.6),
        and codebooks fitted on stage-1 latents are 2-4x too long for the
        vectors they are matched against -- which leaves the 8-level RVQ
        residual ABOVE the input energy on the coarse tiers, i.e. worse
        than transmitting nothing.

        Default off: this changes what the encoder emits, so every
        existing checkpoint and bitstream keeps its current behaviour."""
        self._require_stats(stats, "encode_levels")
        if self.stats_cond is not None and self.stats_cond.subtract_mean:
            from georvq.model.stats_cond import center
            x = center(x, stats, valid)
        bias = None if self.stats_enc is None else self.stats_enc(stats)
        zs = self.tokenizer(self.stem(x), stats_bias=bias)
        if self.norm_latents:
            zs = [F.normalize(z, dim=-1, eps=1e-6) * (z.shape[-1] ** 0.5)
                  for z in zs]
        return zs

    def decode(self, zqs, cond, stats=None, valid=None):
        """zqs: list of (B, cells, dim); cond: (B, n_tiers) depths (or sigmas
        in stage-1 mode). stats: (B, bands) per-window mean when stats
        conditioning is on -- REQUIRED then, because silently omitting it
        would return a mean-removed field and quietly wreck every metric
        rather than raising."""
        self._require_stats(stats, "decode")
        emb = None if self.stats_dec is None else self.stats_dec(stats)
        out = self.decoder(zqs, cond, stats=emb)
        if self.stats_cond is not None and self.stats_cond.subtract_mean:
            from georvq.model.stats_cond import uncenter
            out = uncenter(out, stats.to(out.dtype), valid)
        return out

    def _require_stats(self, stats, where):
        if self.stats_cond is not None and stats is None:
            raise ValueError(
                f"{where}: model.stats_cond is enabled, so the per-window "
                "stats must be passed (model.patch_stats(x, valid)). "
                "Omitting them would silently produce a stats-removed "
                "field.")

    def side_bits(self):
        """Fixed per-window bits outside the index stream (stats channel)."""
        if self.stats_cond is None:
            return 0.0
        from georvq.model.stats_cond import channel_bits
        return float(channel_bits(self.stats_cond, self.bands))

    def _stats_of(self, x, valid=None):
        """The stats argument the encode/decode pair needs, or None when
        stats conditioning is off. Keeps every eval helper a one-argument
        call at the call site while staying explicit inside the model."""
        if self.stats_cond is None:
            return None
        return self.patch_stats(x, valid)

    def patch_stats(self, x, valid=None):
        from georvq.model.stats_cond import patch_stats
        sc = self.stats_cond
        return patch_stats(x, valid, use_std=sc.use_std,
                           std_floor=sc.std_floor)

    def patch_mean(self, x, valid=None):
        from georvq.model.stats_cond import patch_mean
        return patch_mean(x, valid)

    # ---- quantization (the one routing point) ----

    def quantize(self, zs, Ls, loc=None):
        """Quantize per-tier latents at the sampled depth tuple.

        zs: list of (B, cells, dim) encoder outputs (any dtype; VQ runs fp32).
        Ls: (B, n_tiers) depth tuple. loc: per-window location embedding
        (required by conditioner/selector when configured).

        Returns (zqs, commit, select): decoder inputs, commitment loss and
        selector loss (zero when no selector)."""
        device = zs[0].device
        commit = torch.zeros((), device=device)
        select = torch.zeros((), device=device)

        if self.selector is not None:
            scores = self.selector(loc)
            zqs = []
            for i, (z, rvq, sp) in enumerate(zip(zs, self.rvqs, self.specs)):
                per, _, c, sl = rvq.forward_gated(
                    z.float(), scores[i], self.selector.subset_k(sp),
                    sel_n=self.selector.sel_n, temp=self.selector.temp,
                    tau=self.selector.tau)
                m = thermometer(Ls[:, i], sp.depth, device)[:, None, :, None]
                zqs.append((per * m).sum(-2))
                commit = commit + c
                select = select + sl
            return zqs, commit / len(self.specs), select / len(self.specs)

        if self.conditioner is not None:
            zqs, commit = self.conditioner.quantize(
                self.rvqs, self.specs, [z.float() for z in zs], loc, Ls)
            return zqs, commit, select

        zqs = []
        for i, (z, rvq, sp) in enumerate(zip(zs, self.rvqs, self.specs)):
            if self._skip_masked:
                # run each RVQ level only on the rows whose sampled depth
                # reaches it; scale the level's dead threshold by its EMA'd
                # traffic fraction so rare deep levels aren't mass-reset
                zf = z.float()
                residual = zf.clone()
                zq = torch.zeros_like(zf)
                for j, lev in enumerate(rvq.levels):
                    act = Ls[:, i] > j
                    frac = float(act.float().mean())
                    if frac == 0.0:
                        continue
                    key = (i, j)
                    self._part_ema[key] = 0.95 * self._part_ema.get(key, frac) \
                        + 0.05 * frac
                    lev.dead_threshold = self._base_dead * self._part_ema[key]
                    sub = residual[act]
                    q, _, c = lev(sub)
                    zq[act] += q
                    residual[act] = sub - q
                    commit = commit + c * frac / sp.depth
                zqs.append(zq)
            else:
                per, _, c = rvq.all_levels(z.float())  # VQ stays fp32
                m = thermometer(Ls[:, i], sp.depth, device)[:, None, :, None]
                zqs.append((per * m).sum(-2))
                commit = commit + c
        return zqs, commit / len(self.specs), select

    def forward_train(self, x, Ls, loc=None, stats=None, valid=None):
        """Convenience end-to-end pass (the trainer orchestrates the three
        stages itself to control autocast per stage)."""
        zs = self.encode_levels(x, stats, valid)
        zqs, commit, select = self.quantize(zs, Ls, loc)
        return self.decode(zqs, Ls, stats, valid), commit, select

    # ---- stage-1: continuous, noise-robust, sigma-conditioned ----

    def encode_stage1(self, x):
        """Continuous per-tier latents, L2-normalized per cell then scaled by
        sqrt(dim) so per-element std ~= 1 in EVERY tier. This (a) bounds the
        latent scale so an absolute sigma is meaningful (the encoder can't
        escape the noise by inflating magnitudes), and (b) makes one sigma
        corrupt all tiers equally despite their different dims."""
        return [F.normalize(z, dim=-1, eps=1e-6) * (z.shape[-1] ** 0.5)
                for z in self.encode_levels(x)]

    def forward_stage1(self, x, sigmas, drop=None):
        """Encode -> normalize -> add per-tier additive-uniform noise
        U(-s/2, s/2) -> sigma-conditioned decode. sigmas: (B, n_tiers) >= 0.
        drop: optional (B, n_tiers) bool — tiers zeroed to TRUE informational
        absence (matryoshka truncation; heavy noise still leaks signal). The
        sigma conditioning for dropped tiers should be sigma_max so the
        decoder can tell."""
        zs = self.encode_stage1(x)
        noisy = [z + (torch.rand_like(z) - 0.5) * sigmas[:, i].view(-1, 1, 1)
                 for i, z in enumerate(zs)]
        if drop is not None:
            noisy = [nz * (~drop[:, i]).view(-1, 1, 1).to(nz.dtype)
                     for i, nz in enumerate(noisy)]
        return self.decoder(noisy, sigmas)

    # ---- rate model ----

    def train_rate_bits(self, zs, tau_rel=0.5):
        """Differentiable achievable rate (bits/pixel); see PyramidRateModel."""
        return self.rate.rate_bits(zs, self.rvqs, self.window, tau_rel)

    @torch.no_grad()
    def entropy_level_bits(self, x):
        """Honest eval-time entropy-coded bits, per tier (B, depth)."""
        return self.rate.level_bits(self.encode_levels(x), self.rvqs)

    # ---- eval ----

    @torch.no_grad()
    def level_cumsums(self, x, loc=None, valid=None):
        """Encode once; per tier return (B, cells, depth+1, dim) cumulative
        quantized sums so any depth tuple is a gather + decode."""
        zs = self.encode_levels(x, self._stats_of(x, valid), valid)
        cums = []
        if self.selector is not None:
            scores = self.selector(loc)
            for i, (z, rvq, sp) in enumerate(zip(zs, self.rvqs, self.specs)):
                per = rvq.all_levels_gated(z.float(), scores[i],
                                           self.selector.subset_k(sp))[0]
                zero = torch.zeros_like(per[:, :, :1])
                cums.append(torch.cat([zero, per.cumsum(-2)], dim=-2))
            return zs, cums
        if self.conditioner is not None:
            return zs, self.conditioner.cumsums(
                self.rvqs, self.specs, [z.float() for z in zs], loc)
        for z, rvq in zip(zs, self.rvqs):
            per = rvq.all_levels(z)[0]
            zero = torch.zeros_like(per[:, :, :1])
            cums.append(torch.cat([zero, per.cumsum(-2)], dim=-2))
        return zs, cums

    @torch.no_grad()
    def eval_tuple(self, cums, ls, batch_ref, valid=None):
        """Decode one depth tuple from cached cumulative sums."""
        B = batch_ref.shape[0]
        Ls = torch.tensor(ls, device=batch_ref.device)[None].expand(B, -1)
        zqs = [c[:, :, L] for c, L in zip(cums, ls)]
        return self.decode(zqs, Ls, self._stats_of(batch_ref, valid), valid)

    @torch.no_grad()
    def codebook_health(self, x, loc=None):
        """Per-(tier, level) codebook usage / entropy / perplexity on one
        batch: 'pyr{i}_usage_L{l}', 'pyr{i}_entropy_L{l}',
        'pyr{i}_perplexity_L{l}'."""
        out = {}
        zs = self.encode_levels(x, self._stats_of(x), None)

        def _tally(i, sp, per_level_idx):
            """Per-level codebook diagnostics for one tier: fraction of codes
            used, index entropy, and its perplexity.

            Usage and perplexity are health hints only, never quality measures:
            a codebook can reach near-full utilization while explaining almost
            none of the variance (see lab/findings/perplexity-is-gameable.md). Judge books by
            rate-distortion; read these to spot a level that has collapsed.
            """
            for j in range(sp.depth):
                p = torch.bincount(per_level_idx[j].flatten(),
                                   minlength=sp.codes).float()
                out[f"pyr{i}_usage_L{j+1}"] = float((p > 0).float().mean())
                p = p / p.sum()
                out[f"pyr{i}_entropy_L{j+1}"] = float(
                    -(p[p > 0] * p[p > 0].log2()).sum())
                out[f"pyr{i}_perplexity_L{j+1}"] = float(
                    2 ** out[f"pyr{i}_entropy_L{j+1}"])
                # EMA-based (training-traffic) counterparts: not limited
                # to one val batch, and what the dead-reset logic sees
                lev = self.rvqs[i].levels[j]
                q = lev.ema_count / lev.ema_count.sum().clamp_min(1e-12)
                out[f"pyr{i}_ema_alive_L{j+1}"] = float(
                    (lev.ema_count > lev.dead_threshold).float().mean())
                He = float(-(q[q > 0] * q[q > 0].log2()).sum())
                out[f"pyr{i}_ema_perplexity_L{j+1}"] = float(2 ** He)

        if self.selector is not None:
            scores = self.selector(loc)
            for i, (z, rvq, sp) in enumerate(zip(zs, self.rvqs, self.specs)):
                _, idxs = rvq.all_levels_gated(z.float(), scores[i],
                                               self.selector.subset_k(sp))
                _tally(i, sp, idxs)
        elif self.conditioner is not None:
            idxs = self.conditioner.full_depth_indices(
                self.rvqs, self.specs, [z.float() for z in zs], loc)
            for i, sp in enumerate(self.specs):
                _tally(i, sp, idxs[i])
        else:
            for i, (z, rvq, sp) in enumerate(zip(zs, self.rvqs, self.specs)):
                residual = z
                idxs = []
                for lev in rvq.levels:
                    q, idx, _ = lev(residual)
                    residual = residual - q
                    idxs.append(idx)
                _tally(i, sp, idxs)
        return out

    @torch.no_grad()
    def eval_ceiling(self, x, valid=None):
        """Decode the un-quantized latents: the encoder/decoder ceiling."""
        st = self._stats_of(x, valid)
        zs = self.encode_levels(x, st, valid)
        Ls = torch.tensor([sp.depth for sp in self.specs],
                          device=x.device)[None].expand(x.shape[0], -1)
        return self.decode(zs, Ls, st, valid)
