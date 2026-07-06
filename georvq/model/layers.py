"""Shared low-level model components: depth/noise conditioning embeddings
and the AdaLN-Zero transformer block used by the decoder and entropy model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def thermometer(L: torch.Tensor, depth: int, device=None) -> torch.Tensor:
    """L: (B,) ints in [0, depth] -> (B, depth) cumulative encoding.
    Monotone — matches RVQ semantics (level k contains level k-1)."""
    device = L.device if device is None else device
    ramp = torch.arange(depth, device=device).unsqueeze(0)   # (1, depth)
    return (ramp < L.unsqueeze(1)).float()


def modulate(x, shift, scale):
    # x:(B,T,d)  shift/scale:(B,d)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNBlock(nn.Module):
    """Pre-norm transformer block with DiT-style AdaLN-Zero conditioning."""

    def __init__(self, dim, heads, cond_dim, ff_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        ff = int(dim * ff_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
        # 6 modulation params: (shift,scale,gate) × (attn, mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)   # → gates 0, scales 0 ⇒ block = identity at init

    def forward(self, x, c, mask=None):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.ada(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s_msa, sc_msa)
        x = x + g_msa.unsqueeze(1) * self.attn(
            h, h, h, need_weights=False, attn_mask=mask)[0]
        h = modulate(self.norm2(x), s_mlp, sc_mlp)
        x = x + g_mlp.unsqueeze(1) * self.mlp(h)
        return x


def sigma_embed(sigma, n_freq):
    """Continuous per-tier noise-level conditioning (stage-1): Fourier
    features of log-sigma. sigma: (B, T) >= 0 -> (B, T * 2 * n_freq). Replaces
    the discrete depth thermometer with a smooth, codebook-agnostic
    corruption-level embedding (diffusion-timestep style)."""
    ls = torch.log(sigma.clamp_min(1e-6))                      # (B, T)
    freqs = (2.0 ** torch.arange(n_freq, device=sigma.device)).float()
    ang = ls.unsqueeze(-1) * freqs                             # (B, T, n_freq)
    return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)
