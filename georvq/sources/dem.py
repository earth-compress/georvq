"""Copernicus DEM (GLO-30) source: one elevation band in metres.

Absolute elevation spans -400..8800m and dominates the raw signal, so each
window is standardized on its valid pixels before encoding — the same
per-tile stats side-channel pattern as Sentinel-2, here just (mean, std).
The model codes local relief; the two floats restore metres after decoding.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from georvq.metrics import masked_r2, masked_recon_stats
from georvq.model.decoder import DenseHead
from georvq.sources import Source

NODATA = -32767.0
VALID_MIN = -1000.0  # deepest COPDEM land ~-450 m; below = fill/warp smear
STD_FLOOR = 0.1     # metres; keeps near-flat windows from amplifying noise


class DEMNormalizer:
    """Per-window masked mean/std over valid pixels: normalize before
    encode, restore after decode. `stats_vector` is [mean, std] (B, 2)."""

    per_sample = True       # denormalize needs the batch's own stats

    def normalize(self, x, valid):
        vm = valid.unsqueeze(1).float()
        n = vm.sum((2, 3)).clamp_min(1)
        mu = (x * vm).sum((2, 3)) / n
        var = ((x - mu[:, :, None, None]) ** 2 * vm).sum((2, 3)) / n
        sd = var.sqrt().clamp_min(STD_FLOOR)
        out = (x - mu[:, :, None, None]) / sd[:, :, None, None] * vm
        return out, (mu, sd)

    def denormalize(self, x, stats):
        mu, sd = stats
        return x * sd[:, :, None, None] + mu[:, :, None, None]

    @staticmethod
    def stats_vector(stats):
        return torch.cat(stats, 1)                    # (B, 2)


class DEMSource(Source):
    name = "dem"

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self._norm = DEMNormalizer()

    def prep(self, raw, device):
        """float32 (B, 1, W, W) metres -> standardized floats + validity."""
        x = raw.to(device, non_blocking=True).float()
        # fill is -32767 (documented) or -32768 (some mosaics), and warp
        # resampling smears the land/fill boundary into arbitrary large
        # negatives — mask by physical plausibility, not value equality
        # (COPDEM land minimum is ~-450 m)
        valid = x[:, 0] > VALID_MIN
        xn, self._last_stats = self._norm.normalize(x, valid)
        return xn, valid

    def last_stats(self):
        return self._last_stats

    def build_stem(self, model_cfg) -> nn.Module:
        return nn.Identity()

    def build_head(self, model_cfg, in_dim, n_tiers=None,
                   out_px=None) -> nn.Module:
        grid, p = self.head_geom(model_cfg, n_tiers, out_px)
        return DenseHead(in_dim, grid=grid, p=p,
                         bands=model_cfg.bands,
                         head_feat=model_cfg.decoder.head_feat,
                         normalize=False)

    def normalizer(self, model_cfg):
        return self._norm

    def side_bits(self, cfg) -> float:
        """The window's mean and std, sent as two float32 values, which is
        how decode uses them. A cheaper encoding is a separate decision:
        float16 spaces values 8 m apart above 8,192 m, which would land
        directly in the reconstructed elevations."""
        return 64.0

    def recon_stats(self, out, x, valid):
        mse, _ = masked_recon_stats(out, x, valid)
        return mse, {"mse": mse, "r2": masked_r2(out, x, valid)}

    def report(self, model_cfg):
        """Standardized relief, one band: R² (variance explained) is the
        honest quality measure -- cosine over a single band collapses to
        sign agreement -- with RMSE alongside it and a grayscale preview
        (a 3-component PCA-RGB projection of one band cannot be drawn).

        No block-mean baseline: the per-window mean and std are already
        transmitted as a side channel, so scoring against the window mean
        would be scoring against part of the bitstream rather than
        against a rate-free reference.

        The error histogram earns its place here because the normalizer
        can put the second panel in METRES, which is the unit the result
        is actually judged in.
        """
        from georvq.sources.report import scalar_field_spec
        return scalar_field_spec(units="metres")
