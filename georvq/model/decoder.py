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

    def __init__(self, in_dim, grid, p, bands, head_feat=64, normalize=True,
                 dw=False):
        super().__init__()
        self.grid, self.p, self.normalize = grid, p, normalize
        self.to_patch = nn.Linear(in_dim, head_feat * p * p)

        def block(cin, cout):
            if not dw:
                return [nn.Conv2d(cin, cout, 3, 1, 1)]
            # depthwise-separable: 3x3 per-channel then 1x1 mix (~8x fewer
            # MACs at these widths)
            return [nn.Conv2d(cin, cin, 3, 1, 1, groups=cin),
                    nn.Conv2d(cin, cout, 1)]
        self.conv = nn.Sequential(
            *block(head_feat, head_feat), nn.GELU(),
            *block(head_feat, head_feat), nn.GELU(),
            *block(head_feat, bands))

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
                 sigma_freqs=8, late_join=0, suffix_dim=None,
                 suffix_radius=2):
        super().__init__()
        assert 0 <= late_join < depth or (suffix_dim and late_join > 0)
        assert not (suffix_dim and not late_join), \
            "suffix_dim needs late_join (number of narrow cross blocks)"
        self.cls_bridge = cls_bridge
        self.sigma_cond = sigma_cond
        self.sigma_freqs = sigma_freqs
        self.late_join = late_join
        self.suffix_dim = suffix_dim
        self.grad_ckpt = False        # set by the factory from train cfg
        self.specs = specs
        self.fine = specs[-1]
        fdim = suffix_dim or dim
        n_tokens = sum(sp.cells for sp in specs)
        self.projs = nn.ModuleList([
            nn.Linear(sp.dim, fdim if sp is specs[-1] and suffix_dim else dim)
            for sp in specs])
        if suffix_dim:
            n_base = n_tokens - specs[-1].cells
            self.pos = nn.Parameter(torch.randn(1, n_base, dim) * 0.02)
            self.pos_f = nn.Parameter(
                torch.randn(1, specs[-1].cells, fdim) * 0.02)
            from georvq.model.suffix import CrossAdaLNBlock
            self.suffix = nn.ModuleList([
                CrossAdaLNBlock(fdim, dim, grid=specs[-1].grid,
                                radius=suffix_radius, heads=heads,
                                cond_dim=dim, ff_ratio=ff_ratio)
                for _ in range(late_join)])
            self.norm_out_f = nn.LayerNorm(fdim, eps=1e-6)
        else:
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
    def head_in_dim(specs, dim, cls_bridge, suffix_dim=None):
        """Input width of the pixel head: fine token, plus (with cls_bridge)
        every ancestor token concatenated per fine cell."""
        fdim = suffix_dim or dim
        return fdim + (dim * (len(specs) - 1) if cls_bridge else 0)

    def forward(self, zqs, cond_arg):
        """zqs: list of (B, cells_i, dim_i). cond_arg: (B, n_tiers) integer
        depths (depth mode), or (B, n_tiers) per-tier sigma (sigma_cond)."""
        if self.suffix_dim:
            return self._forward_suffix(zqs, cond_arg)
        toks = torch.cat([proj(z) for proj, z in zip(self.projs, zqs)], 1)
        x = toks + self.pos
        if self.sigma_cond:
            cond = sigma_embed(cond_arg, self.sigma_freqs)
        else:
            Ls = cond_arg
            cond = torch.cat([thermometer(Ls[:, i], sp.depth, Ls.device)
                              for i, sp in enumerate(self.specs)], -1)
        c = self.cond_mlp(cond)

        def run(h, blocks):
            m = getattr(self, "attn_mask", None)
            if m is not None:
                m = m[:h.shape[1], :h.shape[1]]    # late-join prefix slice
            if self.grad_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint
                for blk in blocks:
                    h = checkpoint(blk, h, c, m, use_reentrant=False)
            else:
                for blk in blocks:
                    h = blk(h, c, m)
            return h

        if self.late_join:
            nb = x.shape[1] - self.fine.cells
            base = run(x[:, :nb], self.blocks[:len(self.blocks) - self.late_join])
            x = run(torch.cat([base, x[:, nb:]], 1),
                    self.blocks[len(self.blocks) - self.late_join:])
        else:
            x = run(x, self.blocks)
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

    def _cond(self, cond_arg):
        if self.sigma_cond:
            return self.cond_mlp(sigma_embed(cond_arg, self.sigma_freqs))
        Ls = cond_arg
        return self.cond_mlp(torch.cat(
            [thermometer(Ls[:, i], sp.depth, Ls.device)
             for i, sp in enumerate(self.specs)], -1))

    def _forward_suffix(self, zqs, cond_arg):
        """Narrow-suffix path: base tiers through the full AdaLN stack (few
        tokens), fine tier through neighborhood cross blocks at suffix_dim."""
        from torch.utils.checkpoint import checkpoint
        ckpt = self.grad_ckpt and self.training
        c = self._cond(cond_arg)
        base = torch.cat([proj(z) for proj, z in
                          zip(self.projs[:-1], zqs[:-1])], 1) + self.pos
        for blk in self.blocks:
            base = checkpoint(blk, base, c, use_reentrant=False) if ckpt \
                else blk(base, c)
        fine = self.projs[-1](zqs[-1]) + self.pos_f
        for blk in self.suffix:
            fine = checkpoint(blk, fine, base, c, use_reentrant=False) \
                if ckpt else blk(fine, base, c)
        xb = self.norm_out(base)
        xf = self.norm_out_f(fine)
        if self.cls_bridge:
            parts = [xf, xb[:, :1].expand(-1, self.fine.cells, -1)]
            off = 1
            for li, sp in enumerate(self.specs[1:-1]):
                toks = xb[:, off:off + sp.cells]
                parts.append(toks[:, self.anc_maps[li]])
                off += sp.cells
            xf = torch.cat(parts, -1)
        return self.head(xf)
