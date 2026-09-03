"""Tier-sum decoder: per-tier pixel maps, upsampled and summed.

The attention decoder reads pixels only from the finest tier's token
positions, so a coarse-only decode must move everything it knows into
256 informationally-absent relay tokens before the head can see it --
and those empty tokens still occupy attention. Here every PRESENT tier
gets its own projection head after the shared transformer: tier i's
tokens are projected to a map of `map_px[i]` pixels on a side,
bilinearly upsampled to the window, and the maps are summed. A tier at
depth 0 contributes no tokens, no attention traffic and no map.

Under fill-order depth sampling, tier presence is always a prefix, so a
batch splits into at most n_tiers groups and the transformer (which is
length-agnostic) is shared across them. The coarse-only group skips the
transformer entirely: with a single token there is nothing to attend
to, so the quantized coarse vector is piped through a small MLP with
its own projection weights -- separate from the transformer-path coarse
head, whose input is a token shaped by interaction with the finer
tiers.

With `sum_feat`, the tiers communicate after the sum: each head emits a
`sum_feat`-channel feature map instead of bands, the summed features go
through a small shared conv stack that renders bands, and the stack is
FiLM-conditioned on the depth-thermometer (+stats) vector so it knows
how quantized the composite it is looking at is. The renderer is shared
by every presence group, the solo path included, so the coarse-only
readout stops being a fully separate head. Zero-init keeps the FiLM
neutral at the start.

Limitations (asserted at build time): no cls_bridge, no local attention
mask, no sigma/stage-1 mode, and the output is a plain dense band map
(no per-source head, no unit-norm fields).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.model.layers import AdaLNBlock
from georvq.model.layers import thermometer


class TierSumDecoder(nn.Module):
    def __init__(self, specs, window, bands, dim=256, heads=8, depth=8,
                 ff_ratio=2.0, map_px=(8, 8, 64), solo_hidden=512,
                 solo_layers=3, sum_feat=None):
        super().__init__()
        assert len(map_px) == len(specs), "map_px needs one entry per tier"
        for sp, mp in zip(specs, map_px):
            assert mp % sp.grid == 0 and mp <= window, \
                f"map_px {mp} must be a multiple of grid {sp.grid} <= window"
        self.specs = specs
        self.window = window
        self.bands = bands
        self.sum_feat = sum_feat
        self.ch = sum_feat or bands      # channels the tier maps carry
        self.map_px = tuple(map_px)
        self.grad_ckpt = False        # set by the factory from train cfg
        n_tokens = sum(sp.cells for sp in specs)
        self.projs = nn.ModuleList(
            [nn.Linear(sp.dim, dim) for sp in specs])
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        cond_in = sum(sp.depth for sp in specs)
        self.cond_mlp = nn.Sequential(nn.Linear(cond_in, dim), nn.SiLU(),
                                      nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([
            AdaLNBlock(dim, heads, cond_dim=dim, ff_ratio=ff_ratio)
            for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)
        # per-tier map heads (transformer path); each token emits its
        # (map_px/grid)^2-pixel tile of the tier's map
        self.heads = nn.ModuleList([
            nn.Linear(dim, (mp // sp.grid) ** 2 * self.ch)
            for sp, mp in zip(specs, map_px)])
        # coarse-only path: quantized coarse vector (+ conditioning)
        # straight to the coarse map, no attention
        seq, d_in = [], specs[0].dim + dim
        for _ in range(solo_layers - 1):
            seq += [nn.Linear(d_in, solo_hidden), nn.GELU()]
            d_in = solo_hidden
        seq += [nn.Linear(d_in, map_px[0] ** 2 * self.ch)]
        self.solo = nn.Sequential(*seq)
        if sum_feat:
            # shared post-sum renderer: the tiers' feature maps interact
            # here, FiLM-conditioned on the depth/stats vector (zero-init
            # -> neutral modulation at the start)
            self.rc1 = nn.Conv2d(sum_feat, sum_feat, 3, 1, 1)
            self.rc2 = nn.Conv2d(sum_feat, sum_feat, 3, 1, 1)
            self.rc3 = nn.Conv2d(sum_feat, bands, 3, 1, 1)
            self.film = nn.Linear(dim, 4 * sum_feat)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def _cond(self, Ls, stats=None):
        c = self.cond_mlp(torch.cat(
            [thermometer(Ls[:, i], sp.depth, Ls.device)
             for i, sp in enumerate(self.specs)], -1))
        return c if stats is None else c + stats

    def _tile(self, flat, grid, mp):
        """(B, cells, p*p*ch) token tiles -> (B, ch, mp, mp) map."""
        B, p = flat.shape[0], mp // grid
        f = flat.view(B, grid, grid, self.ch, p, p)
        return f.permute(0, 3, 1, 4, 2, 5).reshape(B, self.ch, mp, mp)

    def _render(self, m, c):
        """Summed feature maps -> bands, FiLM'd on the conditioning."""
        g1, b1, g2, b2 = self.film(c).chunk(4, -1)
        h = self.rc1(m)
        h = F.gelu(h * (1 + g1[:, :, None, None]) + b1[:, :, None, None])
        h = self.rc2(h)
        h = F.gelu(h * (1 + g2[:, :, None, None]) + b2[:, :, None, None])
        return self.rc3(h)

    def _up(self, m):
        if m.shape[-1] == self.window:
            return m
        return F.interpolate(m, size=(self.window, self.window),
                             mode="bilinear", align_corners=False)

    def _run_blocks(self, h, c):
        if self.grad_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            for blk in self.blocks:
                h = checkpoint(blk, h, c, None, use_reentrant=False)
        else:
            for blk in self.blocks:
                h = blk(h, c, None)
        return h

    def _attn_group(self, zqs, rows, k, c):
        """Decode the rows whose present tiers are exactly specs[:k]."""
        toks, off, spans = [], 0, []
        for i in range(k):
            t = self.projs[i](zqs[i][rows])
            toks.append(t + self.pos[:, off:off + t.shape[1]])
            spans.append((off, off + t.shape[1]))
            off += self.specs[i].cells
        h = self.norm_out(self._run_blocks(torch.cat(toks, 1), c[rows]))
        out, off = None, 0
        for i in range(k):
            cells = self.specs[i].cells
            m = self._up(self._tile(self.heads[i](h[:, off:off + cells]),
                                    self.specs[i].grid, self.map_px[i]))
            out = m if out is None else out + m
            off += cells
        return out

    def forward(self, zqs, cond_arg, stats=None):
        """zqs: list of (B, cells_i, dim_i); cond_arg: (B, n_tiers) depths.
        Rows are grouped by their present-tier PREFIX (fill-order depth
        sampling guarantees presence is a prefix; a nonzero depth after a
        zero would decode that tier as absent here)."""
        Ls = cond_arg
        B, dev = Ls.shape[0], Ls.device
        c = self._cond(Ls, stats)
        n_present = (Ls > 0).long().sum(1)          # prefix length per row
        out = zqs[0].new_zeros(B, self.ch, self.window, self.window)

        solo = n_present <= 1
        if bool(solo.any()):
            z0 = zqs[0][solo].reshape(solo.sum(), -1).to(c.dtype)
            flat = self.solo(torch.cat([z0, c[solo]], -1))
            out[solo] = self._up(self._tile(
                flat.unsqueeze(1), 1, self.map_px[0])).to(out.dtype)
        for k in range(2, len(self.specs) + 1):
            rows = n_present == k
            if bool(rows.any()):
                out[rows] = self._attn_group(zqs, rows, k, c).to(out.dtype)
        if self.sum_feat:
            out = self._render(out, c)
        return out
