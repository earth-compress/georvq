"""Learned conditional entropy model (RDVQ) for the pyramid codec.

`ContextRateModel` predicts each cell's code distribution over K for one
tier, conditioned on: the cumulative within-tier reconstruction z_q^{(<l)}
(causal: what the decoder already holds before level l), the level index
(thermometer, AdaLN), and optionally coarser-tier reconstruction tokens.

`PyramidRateModel` owns the per-tier models plus the cross-tier context
projections and implements both the differentiable training rate
(`rate_bits`, R = -(softmax(-d/tau) . log q)) and the honest eval-time
hard-index bits (`level_bits`) — the same model is the arithmetic coder's
distribution at deployment.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from georvq.model.layers import AdaLNBlock, thermometer


class ContextRateModel(nn.Module):
    """Per-tier conditional entropy model. forward(z_prev, level, coarse) ->
    (B, cells, K) code logits for the current level's cells."""

    def __init__(self, cells: int, dim: int, num_codes: int, num_levels: int,
                 model_dim: int = 256, n_blocks: int = 4, heads: int = 8,
                 coarse_dim: int = 0, head_per_level: bool = False,
                 bottleneck: int = 0, head_layers: int = 2):
        super().__init__()
        self.cells = cells
        self.num_codes = num_codes
        self.num_levels = num_levels
        self.in_proj = nn.Linear(dim, model_dim)
        self.pos = nn.Parameter(torch.randn(1, cells, model_dim) * 0.02)
        # cross-tier context tokens (already pooled/concatenated by the
        # wrapper), projected to model_dim and prepended to the sequence.
        self.coarse_proj = nn.Linear(coarse_dim, model_dim) if coarse_dim else None
        self.blocks = nn.ModuleList(
            [AdaLNBlock(model_dim, heads, num_levels) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.head_per_level = head_per_level
        # optional narrow bottleneck: squeeze context to `bottleneck` dims
        # before the per-level readout (regularizer).
        self.bottleneck = bottleneck
        hin = bottleneck if bottleneck else model_dim
        if bottleneck:
            self.bneck = nn.Linear(model_dim, bottleneck)

        def _mk_head():           # hin -> (2*hin ->)^head_layers -> num_codes
            hw = 2 * hin
            seq = [nn.Linear(hin, hw), nn.GELU()]
            for _ in range(head_layers - 1):
                seq += [nn.Linear(hw, hw), nn.GELU()]
            seq += [nn.Linear(hw, num_codes)]
            return nn.Sequential(*seq)
        if head_per_level:        # independent MLP readout per level
            self.heads = nn.ModuleList([_mk_head() for _ in range(num_levels)])
        else:
            self.head = nn.Linear(hin, num_codes)

    def forward(self, z_prev, level, coarse=None):
        """z_prev: (B, cells, dim) cumulative recon of levels < `level`
        (detached by caller). level: (B,) long (thermometer marks the levels
        already decoded). coarse: (B, n_coarse, coarse_dim) coarser-tier
        context tokens, or None."""
        x = self.in_proj(z_prev) + self.pos                      # (B, cells, M)
        n_pre = 0
        if self.coarse_proj is not None and coarse is not None:
            ctx = self.coarse_proj(coarse)                       # (B, n_coarse, M)
            x = torch.cat([ctx, x], dim=1)
            n_pre = ctx.shape[1]
        cond = thermometer(level, self.num_levels)               # (B, num_levels)
        for blk in self.blocks:
            x = blk(x, cond)
        x = self.norm(x[:, n_pre:])                              # current-tier cells
        if self.bottleneck:
            x = self.bneck(x)                                    # (B, cells, bottleneck)
        if self.head_per_level:        # uniform level per call -> index by scalar
            return self.heads[int(level.reshape(-1)[0])](x)      # (B, cells, K)
        return self.head(x)                                      # (B, cells, K)


def soft_rate(d: torch.Tensor, logits: torch.Tensor, tau: torch.Tensor):
    """Differentiable rate (bits) for one level.
    d: (N, K) VQ distances (N = B*cells). logits: (B, cells, K) from the
    entropy model. tau: scalar temperature. Returns mean bits/cell.
    R = -(p_soft * log q), p_soft = softmax(-d/tau) (grad to encoder+codebook),
    log q = log_softmax(logits) (grad to the entropy model)."""
    K = d.shape[-1]
    p_soft = torch.softmax(-d / tau, dim=1)                      # (N, K)
    log_q = torch.log_softmax(logits.reshape(-1, K), dim=-1)     # (N, K)
    nats = -(p_soft * log_q).sum(1)                              # (N,)
    return nats.mean() / 0.6931471805599453                      # nats -> bits


def calib_tau(d: torch.Tensor, tau_rel: float = 0.5):
    """tau ~ tau_rel * mean over rows of the min distance — keeps
    softmax(-d/tau) neither one-hot nor uniform."""
    return d.min(dim=1).values.mean().detach() * tau_rel + 1e-9


class PyramidRateModel(nn.Module):
    """Per-tier entropy models + cross-tier context projections for a
    pyramid. Consumes encoder outputs `zs` and the codec's `rvqs`."""

    def __init__(self, specs, model_dim=256, n_blocks=4, ctx_dim=96):
        super().__init__()
        self.specs = specs
        self.ctx_dim = ctx_dim
        self.models = nn.ModuleList([
            ContextRateModel(sp.cells, sp.dim, sp.codes, sp.depth,
                             model_dim=model_dim, n_blocks=n_blocks,
                             coarse_dim=(ctx_dim if i else 0))
            for i, sp in enumerate(specs)])
        # per consumer tier i: a Linear(dim_j -> ctx_dim) for each coarser j
        self.ctx_proj = nn.ModuleList([
            nn.ModuleList([nn.Linear(specs[j].dim, ctx_dim)
                           for j in range(i)])
            for i in range(len(specs))])

    def coarse_ctx(self, recons, i):
        """Cross-tier context tokens for tier i: each coarser tier j<i's full
        (detached) recon (B, cells_j, dim_j) projected to ctx_dim and
        concatenated along the token axis. None for the coarsest tier."""
        if i == 0:
            return None
        return torch.cat([self.ctx_proj[i][j](recons[j])
                          for j in range(i)], dim=1)

    def rate_bits(self, zs, rvqs, window, tau_rel=0.5):
        """Achievable rate (bits/pixel) summed over all tiers & levels.
        zs: list of (B, cells, dim) encoder outputs (caller sub-batches to
        bound memory; pass fp32). Differentiable: dR/dz via softmax(-d/tau),
        dR/dq via the entropy model, dR/dC via the live codebook in
        `distances` (grad_codebook)."""
        B = zs[0].shape[0]
        R_bits = zs[0].new_zeros(())           # bits per window (sum over tiers)
        recons = []                            # full per-tier recon (detached)
        for i, (z, rvq, sp, rm) in enumerate(
                zip(zs, rvqs, self.specs, self.models)):
            levs = rvq.levels_with_rate(z)     # all sp.depth levels (with d, ctx)
            coarse = self.coarse_ctx(recons, i)
            for l, lv in enumerate(levs):
                level = torch.full((B,), l, device=z.device, dtype=torch.long)
                logits = rm(lv["ctx"], level, coarse)        # (B, cells, K)
                tau = calib_tau(lv["d"], tau_rel)
                # soft_rate = mean bits/cell; * cells -> bits/window this level
                R_bits = R_bits + soft_rate(lv["d"], logits, tau) * sp.cells
            recons.append(torch.stack([lv["q"] for lv in levs], -2)
                          .sum(-2).detach())
        return R_bits / (window ** 2)          # bits / pixel

    @torch.no_grad()
    def level_bits(self, zs, rvqs):
        """Eval-time entropy-coded cost. Per tier returns (B, depth)
        HARD-index bits summed over cells: -log2 q(chosen_idx | within-tier
        recon^{<l}, full coarse recon, level) — the honest range-coder rate
        (not the soft train surrogate). One full-depth pass; any tuple's
        rate = sum of its included levels. Coarse context uses the FULL
        coarse reconstruction -> EXACT for operating points with full coarse
        depth, a mild approximation at shallow coarse depths."""
        recons, out = [], []
        for i, (z, rvq, sp, rm) in enumerate(
                zip(zs, rvqs, self.specs, self.models)):
            coarse = self.coarse_ctx(recons, i)
            B = z.shape[0]
            residual, cum, tier_bits = z, torch.zeros_like(z), []
            for l, lev in enumerate(rvq.levels):
                q, idx, _ = lev(residual)                # hard assign (eval: no EMA)
                level = torch.full((B,), l, device=z.device, dtype=torch.long)
                logits = rm(cum, level, coarse)          # ctx = recon^{<l}
                log_q = torch.log_softmax(logits, dim=-1)
                bits = -log_q.gather(-1, idx.unsqueeze(-1)).squeeze(-1)   # (B,cells) nats
                tier_bits.append(bits.sum(-1) / 0.6931471805599453)      # (B,) bits
                residual = residual - q
                cum = cum + q
            out.append(torch.stack(tier_bits, -1))       # (B, depth)
            recons.append(cum)                           # full recon of this tier
        return out                                       # list of (B, depth) per tier
