"""Tessera embedding-field source: 128-band vectors stored as per-pixel
max-abs-quantized int8 chips (nodata = all-bands-zero).

Tessera tiles ship as int8 quantized embeddings plus a per-pixel float32
scale (dequant: x = q * scale). The scale is a positive per-pixel scalar,
so the unit *direction* of each vector is fully determined by the int8
data alone; like AlphaEarth we model the unit-norm field, so `prep` just
normalizes the raw int8 and the scale plane never enters the model (the
store keeps it as a float16 sidecar for amplitude-aware experiments).

Stem is the identity and the head is a dense plane head with per-pixel L2
normalization, exactly as for AlphaEarth.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from georvq.metrics import masked_recon_stats
from georvq.model.decoder import DenseHead
from georvq.sources import Source


class TesseraSource(Source):
    name = "tessera"
    # 10 m native output (no upsampled lattice): trivial period-1 grid
    grid_period = 1

    def prep(self, raw, device):
        """int8 (B, 128, W, W) -> (x, valid). Keeps the first `bands`
        channels, re-normalized to unit vectors; nodata (all-zero vector)
        pixels stay zero and are masked out."""
        s = raw.to(device, non_blocking=True)
        valid = s.ne(0).any(dim=1)
        x = F.normalize(s[:, :self.cfg.bands].float(), dim=1, eps=1e-6)
        return x, valid

    def build_stem(self, model_cfg) -> nn.Module:
        return nn.Identity()

    def build_head(self, model_cfg, in_dim) -> nn.Module:
        fine = model_cfg.tiers[-1]
        return DenseHead(in_dim, grid=fine.grid,
                         p=model_cfg.window // fine.grid,
                         bands=model_cfg.bands,
                         head_feat=model_cfg.decoder.head_feat,
                         normalize=True)

    def recon_stats(self, out, x, valid):
        mse, cos = masked_recon_stats(out, x, valid)
        return mse, {"mse": mse, "cos": cos}
