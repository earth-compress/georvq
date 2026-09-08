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


def _boxcox(x, lam):
    """Per-level Box-Cox: log at lam=0, (x^lam - 1)/lam elsewhere.
    `x` strictly positive and OWNED by the caller (mutated in place);
    `lam` broadcast as (1, C, 1, 1). In-place: at batch 1024 each
    full-batch temporary is ~620 MB on the GPU and the caching
    allocator retains the peak, so the count of temporaries is the
    footprint."""
    logx = torch.log_(x)
    lam_z = lam < 1e-12
    y = logx.mul(lam).expm1_().div_(lam.clamp_min(1e-12))
    return torch.where(lam_z.expand_as(y), logx, y)


def _boxcox_inv(y, lam):
    lam_z = lam < 1e-12
    x = y.mul(lam).add_(1).clamp_min_(1e-30).log_() \
         .div_(lam.clamp_min(1e-12)).exp_()
    return torch.where(lam_z.expand_as(x), torch.exp(y), x)


class _BoxCoxNormalizer:
    """Inverse of the transform pipeline for physical-unit reporting:
    unstandardize, invert the per-level Box-Cox, remove the eps shift."""

    def __init__(self, mu, sd, lam, eps):
        self._mu, self._sd, self._lam, self._eps = mu, sd, lam, eps

    def denormalize(self, x, stats):
        y = x * self._sd.to(x.device) + self._mu.to(x.device)
        return _boxcox_inv(y, self._lam.to(x.device)) - self._eps


# Named level subsets, as indices into the store's ascending 37-level
# axis [1..1000 hPa]. "std13" is the classic 13-level set (50, 100,
# 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa) most
# reanalysis users work with. Selected as "era5:<var>@<subset>".
_LEVEL_SUBSETS = {
    "std13": [8, 10, 12, 14, 16, 17, 19, 21, 23, 25, 30, 33, 36],
}


class Era5Source(Source):
    name = "era5"
    grid_period = 1

    def __init__(self, model_cfg, var="t"):
        super().__init__(model_cfg)
        self._levels = None
        self._bc = None
        if var.endswith("+boxcox"):
            # per-level Box-Cox ahead of standardization, for
            # heavy-tailed variables (q). The exponents and the
            # post-transform moments are model-space constants stored
            # beside the raw moments; the raw entry is untouched, so
            # "era5:q" checkpoints are unaffected.
            var = var[:-len("+boxcox")]
            self._bc = True
        if "@" in var:
            var, subset = var.split("@", 1)
            if subset not in _LEVEL_SUBSETS:
                raise KeyError(f"unknown era5 level subset {subset!r}; "
                               f"have {sorted(_LEVEL_SUBSETS)}")
            self._levels = _LEVEL_SUBSETS[subset]
        self.var = var
        with open(_STATS_PATH) as f:
            stats = json.load(f)
        if var not in stats:
            raise KeyError(
                f"no standardization moments for era5 variable {var!r} in "
                f"{_STATS_PATH}; have {sorted(stats)}")
        s = stats[var]
        if self._bc:
            if "boxcox" not in s:
                raise KeyError(
                    f"era5:{var}+boxcox requested but {_STATS_PATH} has "
                    f"no 'boxcox' block for {var!r}; fit and store the "
                    f"per-level exponents first")
            b = s["boxcox"]
            mean, std, lam = b["mean"], b["std"], b["lam"]
            self._bc_eps = float(b["eps"])
        else:
            mean, std = s["mean"], s["std"]
            lam = None
        if self._levels is not None:
            mean = [mean[i] for i in self._levels]
            std = [std[i] for i in self._levels]
            if lam is not None:
                lam = [lam[i] for i in self._levels]
        if len(mean) != model_cfg.bands:
            raise ValueError(
                f"era5:{var} has {len(mean)} levels but model.bands="
                f"{model_cfg.bands}")
        self._mu = torch.tensor(mean).view(1, -1, 1, 1)
        self._sd = torch.tensor(std).clamp_min(1e-12).view(1, -1, 1, 1)
        self._lam = (None if lam is None
                     else torch.tensor(lam).view(1, -1, 1, 1))

    def prep(self, raw, device):
        """float32 (B, levels, W, W) physical units -> per-level
        standardized floats. ERA5 is gap-free, so validity is all-true.
        With a level subset, the store still serves all 37 levels and
        the unused ones are dropped here."""
        x = raw.to(device, non_blocking=True).float()
        if self._levels is not None and x.shape[1] != len(self._levels):
            x = x[:, self._levels]
        if self._mu.device != x.device:
            self._mu = self._mu.to(x.device)
            self._sd = self._sd.to(x.device)
            if self._lam is not None:
                self._lam = self._lam.to(x.device)
        if self._lam is not None:
            x = _boxcox(x.clamp_min(0) + self._bc_eps, self._lam)
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
        if self._lam is not None:
            return _BoxCoxNormalizer(self._mu, self._sd, self._lam,
                                     self._bc_eps)
        return _FixedNormalizer(self._mu, self._sd)

    def last_stats(self):
        return None

    def _rmse_phys(self, out, x, valid):
        """RMSE in the variable's physical unit. Affine normalization
        needs only the per-level scale (the mean cancels in the
        difference); the Box-Cox variant is nonlinear, so both fields
        are inverted to physical units before differencing."""
        if self._lam is not None:
            n = self.normalizer(None)
            return masked_rmse(n.denormalize(out, None),
                               n.denormalize(x, None), valid)
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
