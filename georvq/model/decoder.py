"""Pyramid attention decoder: full self-attention over all tier tokens with
AdaLN-Zero conditioning on the matryoshka depth vector (or a continuous
sigma embedding in stage-1 mode), followed by a per-source pixel head.

The pixel head is pluggable per data source (`DenseHead` for single-group
dense fields like AlphaEarth; multi-resolution heads for sources like
Sentinel-2). The bitstream is decoder-independent: this module can be
upgraded or replaced without re-encoding (see docs/BITSTREAM.md).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.model.layers import AdaLNBlock, sigma_embed, thermometer


class DenseHead(nn.Module):
    """Fine tokens -> a single dense pixel plane.

    Per-fine-token linear patch projection then a small conv stack that
    smooths patch seams; optional per-pixel L2 normalization for unit-norm
    embedding fields (AlphaEarth)."""

    def __init__(self, in_dim, grid, p, bands, head_feat=64, normalize=True):
        super().__init__()
        self.grid, self.p, self.normalize = grid, p, normalize
        self.to_patch = nn.Linear(in_dim, head_feat * p * p)
        self.conv = nn.Sequential(
            nn.Conv2d(head_feat, head_feat, 3, 1, 1), nn.GELU(),
            nn.Conv2d(head_feat, head_feat, 3, 1, 1), nn.GELU(),
            nn.Conv2d(head_feat, bands, 3, 1, 1))

    def forward(self, fine):                       # fine: (B, g*g, in_dim)
        B, g, p = fine.shape[0], self.grid, self.p
        f = self.to_patch(fine).view(B, g, g, -1, p, p)
        f = f.permute(0, 3, 1, 4, 2, 5).reshape(B, -1, g * p, g * p)
        out = self.conv(f)
        return F.normalize(out, dim=1, eps=1e-6) if self.normalize else out


class PyramidAttnDecoder(nn.Module):
    """Mixing decoder over [tier-0 .. tier-n cells] tokens with AdaLN-Zero
    conditioning on the depth vector, then a per-source pixel head.

    sigma_cond: condition on a per-tier continuous noise level (B, n_tiers)
    via a Fourier embedding instead of the per-tier depth thermometer
    (stage-1: a noise-robust, codebook-agnostic decoder)."""

    def __init__(self, specs, head, dim=384, depth=8, heads=8,
                 cls_bridge=False, ff_ratio=4.0, sigma_cond=False,
                 sigma_freqs=8):
        super().__init__()
        self.cls_bridge = cls_bridge
        self.sigma_cond = sigma_cond
        self.sigma_freqs = sigma_freqs
        self.specs = specs
        self.fine = specs[-1]
        n_tokens = sum(sp.cells for sp in specs)
        self.projs = nn.ModuleList([nn.Linear(sp.dim, dim) for sp in specs])
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        cond_in = (len(specs) * 2 * sigma_freqs if sigma_cond
                   else sum(sp.depth for sp in specs))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_in, dim), nn.SiLU(),
                                      nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([
            AdaLNBlock(dim, heads, cond_dim=dim, ff_ratio=ff_ratio)
            for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)
        if cls_bridge and len(specs) > 2:
            # fine cell (r,c) -> ancestor cell index per intermediate tier
            g_f = specs[-1].grid
            maps = []
            for sp in specs[1:-1]:
                f = g_f // sp.grid
                idx = [(r // f) * sp.grid + (c // f)
                       for r in range(g_f) for c in range(g_f)]
                maps.append(torch.tensor(idx))
            self.register_buffer("anc_maps", torch.stack(maps))
        self.head = head

    @staticmethod
    def head_in_dim(specs, dim, cls_bridge):
        """Input width of the pixel head: fine token, plus (with cls_bridge)
        every ancestor token concatenated per fine cell."""
        return dim * (len(specs) if cls_bridge else 1)

    def forward(self, zqs, cond_arg):
        """zqs: list of (B, cells_i, dim_i). cond_arg: (B, n_tiers) integer
        depths (depth mode), or (B, n_tiers) per-tier sigma (sigma_cond)."""
        toks = torch.cat([proj(z) for proj, z in zip(self.projs, zqs)], 1)
        x = toks + self.pos
        if self.sigma_cond:
            cond = sigma_embed(cond_arg, self.sigma_freqs)
        else:
            Ls = cond_arg
            cond = torch.cat([thermometer(Ls[:, i], sp.depth, Ls.device)
                              for i, sp in enumerate(self.specs)], -1)
        c = self.cond_mlp(cond)
        for blk in self.blocks:
            x = blk(x, c)
        xo = self.norm_out(x)
        fine = xo[:, -self.fine.cells:]                    # (B, cells, dim)
        if self.cls_bridge:    # lineage: concat all ancestors per fine cell
            parts = [fine, xo[:, :1].expand(-1, self.fine.cells, -1)]
            off = 1
            for li, sp in enumerate(self.specs[1:-1]):
                toks = xo[:, off:off + sp.cells]
                parts.append(toks[:, self.anc_maps[li]])
                off += sp.cells
            fine = torch.cat(parts, -1)
        return self.head(fine)
