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
# Only variables with moments in era5_stats.json can load; add a unit
# here when its moments are added there.
_UNITS = {"t": "K", "z": "m2 s-2", "u": "m s-1", "v": "m s-1",
          "q": "kg kg-1"}


def _finite_mask(x):
    """(x with non-finite values zeroed, valid (B, H, W), any_gaps).
    A gap-free batch is returned as the same tensor, so its values are
    untouched."""
    valid = torch.isfinite(x).all(dim=1)
    if bool(valid.all()):
        return x, valid, False
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), valid, True


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

# "era5:t@single": one level per sample (bands=1). The level is a hash
# of the chip's own values, so a chip keeps its level across calls --
# validation re-preps its chips every pass and must score the same
# slices each time -- while the training stream still covers all 37
# levels. Each level is standardized with its own global moments.
_SINGLE = "single"


class Era5BoxcoxPrep:
    """Named prep `boxcox`: the per-level bounded Box-Cox ahead of
    standardization, from the variable's fitted block in
    era5_stats.json. Equivalent to the legacy "era5:<var>+boxcox"
    source grammar (kept for old resolved configs); new configs write
    `data.prep: {name: boxcox}`."""

    invertible = True

    def __init__(self, source, model_cfg):
        # a boxcox-mode twin of the source does the arithmetic; the
        # outer source (plain moments) keeps serving the model/report.
        # The twin is built from the outer source's full spec so a level
        # subset (@std13) carries over.
        self._src = Era5Source(model_cfg, var=source.spec + "+boxcox")

    def forward(self, raw, meta, device):
        x, valid = self._src.prep(raw, device)
        return x, valid, meta

    def inverse(self, y, meta):
        return self._src.normalizer(None).denormalize(y, None)

    def load_state(self, state):
        """The exponents and moments come from era5_stats.json, which
        the module docstring freezes, so the embedded copy is not read
        back."""

    def state(self):
        return {"boxcox": {"lam": self._src._lam.view(-1).tolist(),
                           "mean": self._src._mu.view(-1).tolist(),
                           "std": self._src._sd.view(-1).tolist(),
                           "eps": self._src._bc_eps}}

    def __call__(self, raw, meta, device):
        return self.forward(raw, meta, device)


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
        self._single = False
        #: "<var>[@subset]" without "+boxcox": what a named prep rebuilds
        #: its twin from
        self.spec = var
        if "@" in var:
            var, subset = var.split("@", 1)
            if subset == _SINGLE:
                self._single = True
            elif subset in _LEVEL_SUBSETS:
                self._levels = _LEVEL_SUBSETS[subset]
            else:
                raise KeyError(f"unknown era5 level subset {subset!r}; "
                               f"have {sorted(_LEVEL_SUBSETS)} + "
                               f"{_SINGLE!r}")
        if self._bc and self._single:
            # the @single path standardizes by the drawn level's moments
            # and never applies the transform, and the Box-Cox inverse
            # needs the level, which is not carried past prep
            raise ValueError(
                f"era5:{self.spec}: Box-Cox is not supported with "
                f"@{_SINGLE}")
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
        if self._single:
            if model_cfg.bands != 1:
                raise ValueError("era5:<var>@single needs model.bands=1")
        elif len(mean) != model_cfg.bands:
            raise ValueError(
                f"era5:{var} has {len(mean)} levels but model.bands="
                f"{model_cfg.bands}")
        self._mu = torch.tensor(mean).view(1, -1, 1, 1)
        self._sd = torch.tensor(std).clamp_min(1e-12).view(1, -1, 1, 1)
        self._lam = (None if lam is None
                     else torch.tensor(lam).view(1, -1, 1, 1))

    def _pick_single(self, raw):
        """(B, L, H, W) -> (B, 1, H, W): one level per sample. Training
        draws fresh per call (pool reuse of a patch then covers many
        levels); validation (validate() sets _prep_deterministic) uses
        a content hash so a chip keeps its level across re-preps. The
        chosen indices land in self._pick_idx for the moment gather."""
        n = raw.shape[1]
        if getattr(self, "_prep_deterministic", False):
            h = raw[:, 0, ::5, ::7].double().nan_to_num().abs().sum(dim=(1, 2))
            idx = (h * 1e3).long() % n
        else:
            idx = torch.randint(n, (raw.shape[0],), device=raw.device)
        self._pick_idx = idx
        b = torch.arange(raw.shape[0], device=raw.device)
        return raw[b, idx].unsqueeze(1)

    def prep(self, raw, device):
        """float32 (B, levels, W, W) physical units -> per-level
        standardized floats. A pixel is valid when every level is finite:
        the store serves NaN where a NaN-coded variable (sst over land)
        has no value, and those pixels are set to 0 in model space. The
        reanalysis variables are gap-free, so for them validity is
        all-true. With a level subset, the store still serves all 37
        levels and the unused ones are dropped here."""
        if self._single:
            self._pick_idx = None
            if raw.shape[1] != 1 and torch.device(device).type != "cpu":
                # legacy 37-band store: select the level BEFORE the
                # transfer so only one plane crosses the bus
                raw = self._pick_single(raw)
        x = raw.to(device, non_blocking=True).float()
        if self._levels is not None and x.shape[1] != len(self._levels):
            x = x[:, self._levels]
        if self._mu.device != x.device:
            self._mu = self._mu.to(x.device)
            self._sd = self._sd.to(x.device)
            if self._lam is not None:
                self._lam = self._lam.to(x.device)
        if self._single:
            if x.shape[1] != 1:            # cpu path: not yet selected
                x = self._pick_single(x)
            x, valid, gaps = _finite_mask(x)
            if self._pick_idx is not None:
                # picked from a 37-band store: standardize by the
                # drawn level. A single-level store (--single-level
                # index) serves planes already in model space.
                idx = self._pick_idx.to(x.device)
                mu = self._mu.view(-1).to(x.device)[idx].view(-1, 1, 1, 1)
                sd = self._sd.view(-1).to(x.device)[idx].view(-1, 1, 1, 1)
                x = (x - mu) / sd
            if gaps:
                x = torch.where(valid[:, None], x, torch.zeros_like(x))
            return x, valid
        x, valid, gaps = _finite_mask(x)
        if self._lam is not None:
            x = _boxcox(x.clamp_min(0) + self._bc_eps, self._lam)
        x = (x - self._mu) / self._sd
        if gaps:
            x = torch.where(valid[:, None], x, torch.zeros_like(x))
        return x, valid

    def named_preps(self):
        return {"boxcox": Era5BoxcoxPrep}

    def build_stem(self, model_cfg) -> nn.Module:
        return nn.Identity()

    def build_head(self, model_cfg, in_dim, n_tiers=None,
                   out_px=None) -> nn.Module:
        grid, p = self.head_geom(model_cfg, n_tiers, out_px)
        return DenseHead(in_dim, grid=grid, p=p,
                         bands=model_cfg.bands,
                         head_feat=model_cfg.decoder.head_feat,
                         normalize=False,
                         dw=model_cfg.decoder.head_dw)

    def normalizer(self, model_cfg):
        if self._lam is not None:
            return _BoxCoxNormalizer(self._mu, self._sd, self._lam,
                                     self._bc_eps)
        if self._single:
            # figure panels only: the per-sample level is not available
            # to the normalizer, so scale by the median level's sd --
            # approximate Kelvin, exact standardized values
            med = self._sd.view(-1).median().view(1, 1, 1, 1)
            return _FixedNormalizer(torch.zeros_like(med), med)
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
        if self._single:
            # the level is drawn per sample in prep and is not carried
            # to the metrics (validate preps every chunk before scoring
            # any), so physical RMSE uses the median level's sd --
            # approximate Kelvin; r2/rmse in model space are exact
            sd = self._sd.view(-1).median().to(out.device)
        else:
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
        from georvq.sources.report import (GRAY, Metric, PCA_RGB,
                                           replace_extra, scalar_field_spec)
        unit = _UNITS.get(self.var, "source units")
        spec = scalar_field_spec(units=unit)
        # declared-physical: the harness inverts out and x through the
        # RUN'S prep before this plain RMSE, so the metric is paired
        # with whatever transform actually made the model space (the
        # train-side rmse_phys in recon_stats keeps the source's own
        # conversion and is approximate under a non-default prep)
        from georvq.metrics import masked_rmse as _mr
        phys = Metric("rmse_phys", f"RMSE ({unit})",
                      lambda o, x, v: _mr(o, x, v),
                      higher_better=False, bounded=False,
                      space="physical")
        # levels-as-bands gets the PCA-RGB profile preview; the @single
        # variant is one plane and draws grayscale like DEM
        pv = GRAY if model_cfg.bands < 3 else PCA_RGB
        return replace(replace_extra(spec, add=(phys,)), preview=pv)
