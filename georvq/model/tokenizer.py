"""ViT tokenizer encoder: per-tier conv patch stems feeding a shared
transformer, one latent stream out per pyramid tier.

The coarsest tier (grid=1) reads from a CLS token; every finer tier gets
its own Conv2d patch embed at that tier's grid. Source-specific band
handling happens before this tokenizer, in a `Source.build_stem` module.

With `stem_pyramid`, the independent patch embeds are replaced by one
conv pyramid: the finest tier is embedded from the stem output, and each
coarser tier is a strided conv over the tier below it. The coarsest tier
then arrives with a convolutional read of the image instead of being a
CLS parameter that starts identical for every window and can only learn
through attention. Off by default, because it changes the parameter names
the tokenizer emits and every existing checkpoint carries the other set.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ViTTokenizer(nn.Module):
    """x: (B, C, W, W) -> list of per-tier latents [(B, cells_i, dim_i)]."""

    def __init__(self, specs, window, bands, dim=256, depth=9, ff_ratio=4.0,
                 grad_ckpt=False, stem_pyramid=False, embed_norm=False):
        super().__init__()
        assert specs[0].cells == 1, "coarsest tier must be grid=1 (CLS token)"
        if len(specs) < 2:
            raise ValueError("the tokenizer needs at least two tiers: the "
                             "grid=1 tier is a readout token with no input "
                             "of its own")
        self.specs = specs
        self.grad_ckpt = grad_ckpt
        self.embed_norm = embed_norm
        self.pyramid = bool(stem_pyramid)
        if self.pyramid:
            for a, b in zip(specs, specs[1:]):
                if b.grid % a.grid:
                    raise ValueError(
                        "stem_pyramid needs each grid to divide the next; "
                        f"got {a.grid} -> {b.grid}")
            if window % specs[-1].grid:
                raise ValueError(
                    f"stem_pyramid needs window {window} divisible by the "
                    f"finest grid {specs[-1].grid}")
            # finest tier embedded from the stem output, then one strided
            # conv per step up the pyramid. Built fine-to-coarse and read
            # back coarse-to-fine so the token order is unchanged.
            p = window // specs[-1].grid
            self.stages = nn.ModuleList([nn.Conv2d(bands, dim, p, p)])
            for a, b in zip(specs[:-1], specs[1:]):
                r = b.grid // a.grid
                self.stages.append(nn.Conv2d(dim, dim, r, r))
        else:
            self.patches = nn.ModuleList([
                nn.Conv2d(bands, dim, window // sp.grid, window // sp.grid)
                for sp in specs[1:]])
        if not self.pyramid:
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        n_tok = sum(sp.cells for sp in specs)
        self.pos = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, 8, int(dim * ff_ratio), dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.vit = nn.TransformerEncoder(layer, depth)
        self.out = nn.ModuleList([
            nn.Linear(dim, sp.dim)
            for sp in specs])

    def _run(self, h, layers):
        if self.grad_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            for layer in layers:
                h = checkpoint(layer, h, use_reentrant=False)
        else:
            for layer in layers:
                h = layer(h)
        return h

    def forward(self, x, stats_bias=None):
        # stats_bias: (B, dim) added to the CLS token, as ring-space's
        # `stats_proj` does. The CLS token is the coarsest tier's readout,
        # so this is where knowing the patch mean is most directly useful.
        if self.pyramid:
            f = self.stages[0](x)                  # finest tier map
            maps = [f]
            for st in self.stages[1:]:
                f = st(f)
                maps.append(f)
            # maps run fine -> coarse; tokens go coarse -> fine
            streams = [m.flatten(2).transpose(1, 2) for m in reversed(maps)]
            if stats_bias is not None:
                streams[0] = streams[0] + stats_bias.unsqueeze(1)
        else:
            streams = [self.cls.expand(x.shape[0], -1, -1)]
            if stats_bias is not None:
                streams[0] = streams[0] + stats_bias.unsqueeze(1)
            streams += [pe(x).flatten(2).transpose(1, 2)
                        for pe in self.patches]
        h = torch.cat(streams, 1) + self.pos
        if self.embed_norm:
            # parameter-free: bounds layer 0's pre-norm backward (1/std
            # of this stream), the measured spike-amplification site
            h = nn.functional.layer_norm(h, (h.shape[-1],))
        h = self._run(h, self.vit.layers)
        zs, o = [], 0
        for sp, lo in zip(self.specs, self.out):
            zs.append(lo(h[:, o:o + sp.cells]))
            o += sp.cells
        return zs
