"""ViT tokenizer encoder: per-tier conv patch stems feeding a shared
transformer, one latent stream out per pyramid tier.

The coarsest tier (grid=1) reads from a CLS token; every finer tier gets
its own Conv2d patch embed at that tier's grid. Source-specific band
handling happens before this tokenizer, in a `Source.build_stem` module.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ViTTokenizer(nn.Module):
    """x: (B, C, W, W) -> list of per-tier latents [(B, cells_i, dim_i)]."""

    def __init__(self, specs, window, bands, dim=256, depth=9, ff_ratio=4.0,
                 grad_ckpt=False, late_join=0, suffix_dim=None,
                 suffix_radius=2):
        super().__init__()
        assert specs[0].cells == 1, "coarsest tier must be grid=1 (CLS token)"
        assert 0 <= late_join < depth or (suffix_dim and late_join > 0)
        assert not (suffix_dim and not late_join), \
            "suffix_dim needs late_join (number of narrow cross layers)"
        self.specs = specs
        self.grad_ckpt = grad_ckpt
        self.late_join = late_join
        self.suffix_dim = suffix_dim
        self.n_base = 1 + sum(sp.cells for sp in specs[1:-1])
        fdim = suffix_dim or dim                # finest tier's stem/pos width
        self.single = len(specs) == 1
        if self.single:
            # single-vector autoencoder: per-pixel input tokens feed the
            # trunk, the CLS token is the only readout
            self.patches = nn.ModuleList([nn.Conv2d(bands, dim, 1)])
            fdim = dim
        else:
            self.patches = nn.ModuleList([
                nn.Conv2d(bands, fdim if sp is specs[-1] else dim,
                          window // sp.grid, window // sp.grid)
                for sp in specs[1:]])
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        n_tok = 1 + (window * window if self.single
                     else sum(sp.cells for sp in specs[1:]))
        if suffix_dim:
            self.pos = nn.Parameter(
                torch.randn(1, self.n_base, dim) * 0.02)
            self.pos_f = nn.Parameter(
                torch.randn(1, specs[-1].cells, fdim) * 0.02)
            from georvq.model.suffix import CrossSuffixLayer
            self.suffix = nn.ModuleList([
                CrossSuffixLayer(fdim, dim, grid=specs[-1].grid,
                                 radius=suffix_radius, heads=8,
                                 ff_ratio=ff_ratio)
                for _ in range(late_join)])
        else:
            self.pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, 8, int(dim * ff_ratio), dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.vit = nn.TransformerEncoder(layer, depth)
        self.out = nn.ModuleList([
            nn.Linear(fdim if sp is specs[-1] else dim, sp.dim)
            for sp in specs])

    def _run(self, h, layers):
        m = getattr(self, "attn_mask", None)
        if m is not None:
            m = m[:h.shape[1], :h.shape[1]]    # late-join prefix slice
        if self.grad_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            for layer in layers:
                h = checkpoint(layer, h, m, use_reentrant=False)
        else:
            for layer in layers:
                h = layer(h, m)
        return h

    def _run_suffix(self, fine, base):
        if self.grad_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            for layer in self.suffix:
                fine = checkpoint(layer, fine, base, use_reentrant=False)
        else:
            for layer in self.suffix:
                fine = layer(fine, base)
        return fine

    def forward(self, x):
        streams = [self.cls.expand(x.shape[0], -1, -1)]
        streams += [pe(x).flatten(2).transpose(1, 2) for pe in self.patches]
        if self.suffix_dim:
            # base tiers: full self-attention stack on few tokens; finest
            # tier: narrow cross+local suffix reading the finished base
            base = torch.cat(streams[:-1], 1) + self.pos
            base = self._run(base, self.vit.layers)
            fine = self._run_suffix(streams[-1] + self.pos_f, base)
            zs, o = [], 0
            for sp, lo in zip(self.specs[:-1], self.out[:-1]):
                zs.append(lo(base[:, o:o + sp.cells]))
                o += sp.cells
            zs.append(self.out[-1](fine))
            return zs
        h = torch.cat(streams, 1) + self.pos
        layers = self.vit.layers
        if self.late_join:
            # coarse tiers run the prefix alone; the finest tier's tokens
            # (patch embed + pos only) join for the last `late_join` layers
            base, ultra = h[:, :self.n_base], h[:, self.n_base:]
            base = self._run(base, layers[:len(layers) - self.late_join])
            h = self._run(torch.cat([base, ultra], 1),
                          layers[len(layers) - self.late_join:])
        else:
            h = self._run(h, layers)
        zs, o = [], 0
        for sp, lo in zip(self.specs, self.out):
            zs.append(lo(h[:, o:o + sp.cells]))
            o += sp.cells
        return zs
