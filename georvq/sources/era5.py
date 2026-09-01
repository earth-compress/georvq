"""ERA5 single-variable source: one atmospheric variable on all of its
pressure levels, levels as bands, one hourly slice per chip.

The store (georvq/data/era5.py) serves physical units; `prep`
standardizes each level with fixed global moments from
era5_stats.json, measured once over 256 random slices of the train
store (2026-08-31). Like tessera's XNORM, the moments are part of the
model-space definition: changing them invalidates trained checkpoints.

The loss is per-pixel MSE on the standardized block, which weights
every level equally; the several-fold per-level variance differences
are exactly what the standardization removes. Levels of one variable
share units and smoothly varying structure, so no per-band weighting
beyond the standardization is applied. Quality is reported as pooled R2
over the standardized block, the DEM convention for physical fields
(cosine over a temperature profile would mostly measure the profile's
fixed shape).
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import torch
import torch.nn as nn

from georvq.metrics import masked_r2, masked_recon_stats, masked_rmse
from georvq.model.decoder import DenseHead
from georvq.sources import Source

_STATS_PATH = os.path.join(os.path.dirname(__file__), "era5_stats.json")

# Units of the DEQUANTIZED store values, per variable (ECMWF conventions).
_UNITS = {"t": "K", "z": "m2 s-2", "u": "m s-1", "v": "m s-1",
          "q": "kg kg-1", "msl": "Pa", "t2m": "K", "u10": "m s-1",
          "v10": "m s-1"}


class _FixedNormalizer:
    """Per-level affine with FIXED global moments, so `stats` is unused.
    Exists so the error histogram can draw its second panel in physical
    units; there is no per-tile side channel to encode."""

    def __init__(self, mu, sd):
        self._mu, self._sd = mu, sd

    def denormalize(self, x, stats):
        return x * self._sd.to(x.device) + self._mu.to(x.device)


class Era5Source(Source):
    name = "era5"
    grid_period = 1

    def __init__(self, model_cfg, var="t"):
        super().__init__(model_cfg)
        self.var = var
        with open(_STATS_PATH) as f:
            stats = json.load(f)
        if var not in stats:
            raise KeyError(
                f"no standardization moments for era5 variable {var!r} in "
                f"{_STATS_PATH}; have {sorted(stats)}")
        s = stats[var]
        if len(s["mean"]) != model_cfg.bands:
            raise ValueError(
                f"era5:{var} has {len(s['mean'])} levels but model.bands="
                f"{model_cfg.bands}")
        self._mu = torch.tensor(s["mean"]).view(1, -1, 1, 1)
        self._sd = torch.tensor(s["std"]).clamp_min(1e-12).view(1, -1, 1, 1)

    def prep(self, raw, device):
        """float32 (B, levels, W, W) physical units -> per-level
        standardized floats. ERA5 is gap-free, so validity is all-true."""
        x = raw.to(device, non_blocking=True).float()
        if self._mu.device != x.device:
            self._mu = self._mu.to(x.device)
            self._sd = self._sd.to(x.device)
        x = (x - self._mu) / self._sd
        valid = torch.ones(x.shape[0], *x.shape[2:], dtype=torch.bool,
                           device=x.device)
        return x, valid

    def build_stem(self, model_cfg) -> nn.Module:
        return nn.Identity()

    def build_head(self, model_cfg, in_dim) -> nn.Module:
        fine = model_cfg.tiers[-1]
        return DenseHead(in_dim, grid=fine.grid,
                         p=model_cfg.window // fine.grid,
                         bands=model_cfg.bands,
                         head_feat=model_cfg.decoder.head_feat,
                         normalize=False,
                         dw=model_cfg.decoder.head_dw)

    def normalizer(self, model_cfg):
        return _FixedNormalizer(self._mu, self._sd)

    def last_stats(self):
        return None

    def _rmse_phys(self, out, x, valid):
        """RMSE in the variable's physical unit. The per-level sd converts
        each level's standardized error back before pooling; the mean
        cancels in the difference, so only the scale is applied."""
        sd = self._sd.to(out.device)
        return masked_rmse(out * sd, x * sd, valid)

    def recon_stats(self, out, x, valid):
        mse, _ = masked_recon_stats(out, x, valid)
        return mse, {"mse": mse, "r2": masked_r2(out, x, valid),
                     "rmse_phys": self._rmse_phys(out, x, valid)}

    def report(self, model_cfg):
        """R2 with RMSE, as for DEM, but a PCA-RGB preview: with levels as
        bands there is a full profile per pixel to project, and a single
        gray level would show one arbitrary pressure surface. rmse_phys
        pools the de-standardized error over all levels, so it reads in
        the variable's own unit (K for t)."""
        from georvq.sources.report import (Metric, PCA_RGB, replace_extra,
                                           scalar_field_spec)
        unit = _UNITS.get(self.var, "source units")
        spec = scalar_field_spec(units=unit)
        phys = Metric("rmse_phys", f"RMSE ({unit})", self._rmse_phys,
                      higher_better=False, bounded=False)
        return replace(replace_extra(spec, add=(phys,)), preview=PCA_RGB)
