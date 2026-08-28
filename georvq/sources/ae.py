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
    # the published embeddings are ~20m data 2x-upsampled to a 10m grid, so
    # the pixel lattice has period-2 structure (used by aligned cropping)
    grid_period = 2

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

    def build_head(self, model_cfg, in_dim, n_tiers=None,
                   out_px=None) -> nn.Module:
        """Dense pixel head, L2-normalized: AlphaEarth embeddings are
        unit-norm by construction, so the head should not be free to emit
        vectors of any other length. Compare `TesseraSource.build_head`,
        which sets `normalize=False` because that field is not unit-norm.
        """
        grid, p = self.head_geom(model_cfg, n_tiers, out_px)
        return DenseHead(in_dim, grid=grid, p=p,
                         bands=model_cfg.bands,
                         head_feat=model_cfg.decoder.head_feat,
                         normalize=True,
                         dw=model_cfg.decoder.head_dw)

    def recon_stats(self, out, x, valid):
        mse, cos = masked_recon_stats(out, x, valid)
        return mse, {"mse": mse, "cos": cos}

    def report(self, model_cfg):
        """Cosine on a unit-norm field, with the block-mean floor: the
        published embeddings are exactly unit-norm, so the best constant
        direction is the honest zero-rate reference."""
        from georvq.sources.report import embedding_field_spec
        return embedding_field_spec()
