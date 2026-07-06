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


def monotone_tuples(specs):
    """All (L_0..L_n) with L_0>=1 and 0 <= L_i <= min(L_{i-1}, d_i)."""
    tuples = [[L] for L in range(1, specs[0].depth + 1)]
    for sp in specs[1:]:
        tuples = [t + [L] for t in tuples
                  for L in range(0, min(t[-1], sp.depth) + 1)]
    return tuples


def tuple_bits(specs, ls):
    """Nominal (uncoded) bits/window for one depth tuple."""
    return float(sum(sp.bits(L) for sp, L in zip(specs, ls)))


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
        self.continuous = cfg.vq is None

        self.stem = source.build_stem(cfg)
        self.tokenizer = ViTTokenizer(
            self.specs, cfg.window, cfg.bands, dim=cfg.tokenizer.dim,
            depth=cfg.tokenizer.depth, ff_ratio=cfg.ff_ratio)

        if not self.continuous:
            v = cfg.vq
            self.rvqs = nn.ModuleList([
                ResidualVQ(num_levels=sp.depth, num_codes=sp.codes,
                           dim=sp.dim, decay=v.ema_decay,
                           dead_threshold=v.dead_threshold,
                           grad_codebook=v.grad_codebook,
                           bf16_scores=v.bf16_scores)
                for sp in self.specs])
            self._skip_masked = v.skip_masked
            self._base_dead = v.dead_threshold
            self._part_ema = {}      # (tier, level) -> EMA of active fraction

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

        head_in = PyramidAttnDecoder.head_in_dim(
            self.specs, cfg.decoder.dim, cfg.decoder.cls_bridge)
        self.decoder = PyramidAttnDecoder(
            self.specs, head=source.build_head(cfg, head_in),
            dim=cfg.decoder.dim, depth=cfg.decoder.depth,
            heads=cfg.decoder.heads, cls_bridge=cfg.decoder.cls_bridge,
            ff_ratio=cfg.ff_ratio, sigma_cond=cfg.sigma is not None,
            sigma_freqs=cfg.sigma.freqs if cfg.sigma else 8)

    # ---- encode / decode ----

    def encode_levels(self, x):
        """x: (B, C, W, W) -> list of per-tier latents [(B, cells, dim)]."""
        return self.tokenizer(self.stem(x))

    def decode(self, zqs, cond):
        """zqs: list of (B, cells, dim); cond: (B, n_tiers) depths (or sigmas
        in stage-1 mode)."""
        return self.decoder(zqs, cond)

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

    def forward_train(self, x, Ls, loc=None):
        """Convenience end-to-end pass (the trainer orchestrates the three
        stages itself to control autocast per stage)."""
        zs = self.encode_levels(x)
        zqs, commit, select = self.quantize(zs, Ls, loc)
        return self.decode(zqs, Ls), commit, select

    # ---- stage-1: continuous, noise-robust, sigma-conditioned ----

    def encode_stage1(self, x):
        """Continuous per-tier latents, L2-normalized per cell then scaled by
        sqrt(dim) so per-element std ~= 1 in EVERY tier. This (a) bounds the
        latent scale so an absolute sigma is meaningful (the encoder can't
        escape the noise by inflating magnitudes), and (b) makes one sigma
        corrupt all tiers equally despite their different dims."""
        return [F.normalize(z, dim=-1, eps=1e-6) * (z.shape[-1] ** 0.5)
                for z in self.encode_levels(x)]

    def forward_stage1(self, x, sigmas):
        """Encode -> normalize -> add per-tier additive-uniform noise
        U(-s/2, s/2) -> sigma-conditioned decode. sigmas: (B, n_tiers) >= 0."""
        zs = self.encode_stage1(x)
        noisy = [z + (torch.rand_like(z) - 0.5) * sigmas[:, i].view(-1, 1, 1)
                 for i, z in enumerate(zs)]
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
    def level_cumsums(self, x, loc=None):
        """Encode once; per tier return (B, cells, depth+1, dim) cumulative
        quantized sums so any depth tuple is a gather + decode."""
        zs = self.encode_levels(x)
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
    def eval_tuple(self, cums, ls, batch_ref):
        """Decode one depth tuple from cached cumulative sums."""
        B = batch_ref.shape[0]
        Ls = torch.tensor(ls, device=batch_ref.device)[None].expand(B, -1)
        zqs = [c[:, :, L] for c, L in zip(cums, ls)]
        return self.decode(zqs, Ls)

    @torch.no_grad()
    def codebook_health(self, x, loc=None):
        """Per-(tier, level) codebook usage / entropy / perplexity on one
        batch: 'pyr{i}_usage_L{l}', 'pyr{i}_entropy_L{l}',
        'pyr{i}_perplexity_L{l}'."""
        out = {}
        zs = self.encode_levels(x)

        def _tally(i, sp, per_level_idx):
            for j in range(sp.depth):
                p = torch.bincount(per_level_idx[j].flatten(),
                                   minlength=sp.codes).float()
                out[f"pyr{i}_usage_L{j+1}"] = float((p > 0).float().mean())
                p = p / p.sum()
                out[f"pyr{i}_entropy_L{j+1}"] = float(
                    -(p[p > 0] * p[p > 0].log2()).sum())
                out[f"pyr{i}_perplexity_L{j+1}"] = float(
                    2 ** out[f"pyr{i}_entropy_L{j+1}"])

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
    def eval_ceiling(self, x):
        """Decode the un-quantized latents: the encoder/decoder ceiling."""
        zs = self.encode_levels(x)
        Ls = torch.tensor([sp.depth for sp in self.specs],
                          device=x.device)[None].expand(x.shape[0], -1)
        return self.decode(zs, Ls)
