"""Ladder decoder: staged decoding with one exit per rate-ladder prefix.

Stage 0 processes the coarse token alone through residual MLP blocks (no
attention: one token has nothing to attend to) and exits through a
learned-upsampling head. Stage 1 mixes coarse+mid tokens through
`depth` transformer blocks and exits from the mid tokens; stage 2 adds
the fine tokens for another `depth` blocks and exits from those. A row
runs exactly the stages its present-tier prefix licenses, so coarse-only
decoding costs a few MLPs and coarse+mid runs attention over 17 tokens
rather than 273.

Deep supervision: during training every exit a row's tuple licenses is
computed and stored in `last_aux` for the trainer to add to the loss
(weights 1:1:1). Under fill-order sampling every non-final active tier
is saturated by construction, so each computed exit sits exactly on a
rate-ladder rung -- the coarse exit trains on every row, which is the
direct gradient pushing high-value low-frequency content into the
coarse code.

Conditioning is PER TIER: stage i sees its own tier's depth thermometer
(plus the stats embedding) and nothing else. From an exit's viewpoint
the finer depths are identically zero and the coarser ones identically
saturated, so cross-tier conditioning would carry no information.

Every exit uses the same head architecture (`PSHead`): per-token linear
to a feature tile at the tier's grid, pixel-shuffle conv stages up to
the window, then a conv render to bands. The fine tier's instance has
zero upsampling stages and degenerates to the classic dense head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.model.layers import AdaLNBlock, modulate, thermometer


class FiLMMLPBlock(nn.Module):
    """Residual MLP block with AdaLN-Zero-style conditioning (the
    attention-free sibling of AdaLNBlock, for the single-token stem)."""

    def __init__(self, dim, cond_dim, ff_ratio=2.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        ff = int(dim * ff_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, ff), nn.GELU(),
                                 nn.Linear(ff, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)   # identity at init

    def forward(self, x, c):
        s, sc, g = self.ada(c).chunk(3, dim=-1)
        return x + g.unsqueeze(1) * self.mlp(
            modulate(self.norm(x), s, sc))


class PSHead(nn.Module):
    """Unified exit head: tokens -> feature tiles at `base_px` -> learned
    pixel-shuffle upsampling to `window` -> conv render to bands."""

    def __init__(self, in_dim, grid, base_px, window, bands, feat=64):
        super().__init__()
        assert base_px % grid == 0 and window % base_px == 0
        stages = int(math.log2(window // base_px))
        assert 2 ** stages * base_px == window, \
            "window / base_px must be a power of two"
        self.grid, self.feat = grid, feat
        self.p = base_px // grid
        self.base_px = base_px
        self.to_patch = nn.Linear(in_dim, feat * self.p ** 2)
        ups = []
        for _ in range(stages):
            ups += [nn.Conv2d(feat, 4 * feat, 3, 1, 1),
                    nn.PixelShuffle(2), nn.GELU()]
        self.up = nn.Sequential(*ups)
        self.render = nn.Sequential(
            nn.Conv2d(feat, feat, 3, 1, 1), nn.GELU(),
            nn.Conv2d(feat, bands, 3, 1, 1))

    def forward(self, toks):                     # (B, grid*grid, in_dim)
        B, g, p = toks.shape[0], self.grid, self.p
        f = self.to_patch(toks).view(B, g, g, self.feat, p, p)
        f = f.permute(0, 3, 1, 4, 2, 5).reshape(B, self.feat,
                                                self.base_px, self.base_px)
        return self.render(self.up(f))


class LadderDecoder(nn.Module):
    def __init__(self, specs, window, bands, dim=256, depth=8, heads=8,
                 ff_ratio=2.0, map_px=(8, 16, 64), head_feat=64,
                 stem_layers=3):
        super().__init__()
        assert len(specs) == len(map_px)
        self.specs = specs
        self.window = window
        self.grad_ckpt = False        # set by the factory from train cfg
        self.projs = nn.ModuleList(
            [nn.Linear(sp.dim, dim) for sp in specs])
        self.pos = nn.Parameter(
            torch.randn(1, sum(sp.cells for sp in specs), dim) * 0.02)
        # per-tier conditioning MLPs over that tier's own thermometer
        self.cond_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(sp.depth, dim), nn.SiLU(),
                          nn.Linear(dim, dim)) for sp in specs])
        self.stem = nn.ModuleList(
            [FiLMMLPBlock(dim, dim, ff_ratio) for _ in range(stem_layers)])
        self.stages = nn.ModuleList([
            nn.ModuleList([AdaLNBlock(dim, heads, cond_dim=dim,
                                      ff_ratio=ff_ratio)
                           for _ in range(depth)])
            for _ in specs[1:]])
        self.heads = nn.ModuleList([
            PSHead(dim, sp.grid, mp, window, bands, feat=head_feat)
            for sp, mp in zip(specs, map_px)])
        self.last_aux = None

    def _cond(self, i, Ls, stats):
        c = self.cond_mlps[i](
            thermometer(Ls[:, i], self.specs[i].depth, Ls.device))
        return c if stats is None else c + stats

    def _run(self, blocks, h, c):
        if self.grad_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            for blk in blocks:
                h = checkpoint(blk, h, c, None, use_reentrant=False)
        else:
            for blk in blocks:
                h = blk(h, c, None)
        return h

    def forward(self, zqs, cond_arg, stats=None):
        """Returns each row's DEEPEST licensed exit; during training the
        shallower exits land in `last_aux` as (row_index_tensor, map)."""
        Ls = cond_arg
        B = Ls.shape[0]
        n_present = (Ls > 0).long().sum(1).clamp_min(1)
        out = None
        aux = []
        act = {}                       # per-stage activation maxima

        # stage 0: every row, single token, no attention
        c0 = self._cond(0, Ls, stats)
        h = self.projs[0](zqs[0]) + self.pos[:, :1]
        for blk in self.stem:
            h = blk(h, c0)
        act["stem"] = float(h.detach().abs().max())
        a_map = self.heads[0](h)
        out = a_map.new_zeros(B, a_map.shape[1], self.window, self.window)
        final0 = n_present == 1
        out[final0] = a_map[final0]
        deeper = (~final0).nonzero(as_tuple=True)[0]
        if self.training and deeper.numel():
            aux.append((deeper, a_map[deeper]))

        rows = torch.arange(B, device=Ls.device)   # rows alive in `h`
        off = 1
        for s, sp in enumerate(self.specs[1:], start=1):
            keep = n_present[rows] >= s + 1        # continue past stage s-1
            rows = rows[keep]
            if rows.numel() == 0:
                break
            h = h[keep]
            t = self.projs[s](zqs[s][rows]) + self.pos[:, off:off + sp.cells]
            off += sp.cells
            h = self._run(self.stages[s - 1], torch.cat([h, t], 1),
                          self._cond(s, Ls, stats)[rows])
            act[f"stage{s}"] = float(h.detach().abs().max())
            m = self.heads[s](h[:, -sp.cells:])
            final = n_present[rows] == s + 1
            out[rows[final]] = m[final].to(out.dtype)
            if self.training and (~final).any():
                aux.append((rows[~final], m[~final]))
        self.last_aux = aux if self.training else None
        # activation telemetry: the latch failure mode (activations blow
        # up, norms saturate, the stage freezes) shows here as a rising
        # max long before the val ceiling moves
        self.act_max = act
        return out
