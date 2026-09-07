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
                 grad_ckpt=False, late_join=0, embed_norm=False):
        super().__init__()
        assert specs[0].cells == 1, "coarsest tier must be grid=1 (CLS token)"
        assert 0 <= late_join < depth
        self.specs = specs
        self.grad_ckpt = grad_ckpt
        self.late_join = late_join
        self.embed_norm = embed_norm
        # tokens before the finest tier: the late-join prefix runs on
        # these alone, so the split point is a property of the geometry
        self.n_base = 1 + sum(sp.cells for sp in specs[1:-1])
        fdim = dim
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

    def forward(self, x, stats_bias=None):
        # stats_bias: (B, dim) added to the CLS token, as ring-space's
        # `stats_proj` does. The CLS token is the coarsest tier's readout,
        # so this is where knowing the patch mean is most directly useful.
        streams = [self.cls.expand(x.shape[0], -1, -1)]
        if stats_bias is not None:
            streams[0] = streams[0] + stats_bias.unsqueeze(1)
        streams += [pe(x).flatten(2).transpose(1, 2) for pe in self.patches]
        h = torch.cat(streams, 1) + self.pos
        if self.embed_norm:
            # parameter-free: bounds layer 0's pre-norm backward (1/std
            # of this stream), the measured spike-amplification site
            h = nn.functional.layer_norm(h, (h.shape[-1],))
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
