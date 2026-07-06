"""AlphaEarth embedding-field source: 64-band unit-norm vectors stored as
signed-square-quantized int8 chips (nodata = -128).

The field is already unit-norm per pixel, so there is no stats side-channel;
the stem is the identity (the 64-band plane feeds the tokenizer directly)
and the head is a dense plane head with per-pixel L2 normalization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.metrics import masked_recon_stats
from georvq.model.decoder import DenseHead
from georvq.sources import Source

NODATA = -128


def dequant(s: torch.Tensor) -> torch.Tensor:
    """Signed-square de-quantisation: int8 -> unit-norm float vectors."""
    x = s.float() / 127.5
    return x.square() * torch.sign(x)


class AESource(Source):
    name = "ae"

    def prep(self, raw, device):
        """int8 (B, 64, W, W) -> (x, valid). Keeps the first `bands`
        channels, re-normalized to unit vectors; nodata pixels zeroed."""
        s = raw.to(device, non_blocking=True)
        valid = s[:, 0] != NODATA
        x = dequant(s) * valid.unsqueeze(1)
        x = F.normalize(x[:, :self.cfg.bands], dim=1, eps=1e-6)
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
