"""Sentinel-2 L1C source: 13 bands at three native resolutions.

Bands stay at their native ground sample distance end to end — no
resampling of the data itself:

- stem: per-resolution conv stacks run at native GSD, then upsample-and-
  fuse onto the 10m grid to form the single plane the tokenizer eats.
- head: per-resolution patch heads decode the fine tokens back to each
  native grid.
- normalizer: per-tile per-band mean/std are removed before encoding and
  re-applied after decoding; the stats travel in the packet's side channel
  (see georvq.inference.stats_codec).

Batches are dicts {"10m": (B,4,W,W), "20m": (B,6,W/2,W/2),
"60m": (B,3,W/6,W/6)} with W = window (10m pixels, divisible by 6).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.data.pool import GroupBatch
from georvq.sources import Source

BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
BANDS_60M = ["B01", "B09", "B10"]
GROUPS = {"10m": (len(BANDS_10M), 1), "20m": (len(BANDS_20M), 2),
          "60m": (len(BANDS_60M), 6)}
MAX_VALUE = 10000.0     # stored uint16 reflectance scale


class S2NativeStem(nn.Module):
    """Native-resolution conv stems fused onto the 10m grid."""

    def __init__(self, out_ch, width=64):
        super().__init__()
        w = width
        self.enc_10m = nn.Sequential(
            nn.Conv2d(GROUPS["10m"][0], w, 3, 1, 1), nn.GELU(),
            nn.Conv2d(w, w, 3, 1, 1), nn.GELU())
        self.enc_20m = nn.Sequential(
            nn.Conv2d(GROUPS["20m"][0], w, 3, 1, 1), nn.GELU(),
            nn.Conv2d(w, w, 3, 1, 1), nn.GELU())
        self.enc_60m = nn.Sequential(
            nn.Conv2d(GROUPS["60m"][0], w, 3, 1, 1), nn.GELU(),
            nn.Conv2d(w, w, 3, 1, 1), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * w, out_ch, 1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1))

    def forward(self, x: dict):
        f10 = self.enc_10m(x["10m"])
        f20 = F.interpolate(self.enc_20m(x["20m"]), scale_factor=2,
                            mode="nearest")
        f60 = F.interpolate(self.enc_60m(x["60m"]), scale_factor=6,
                            mode="nearest")
        return self.fuse(torch.cat([f10, f20, f60], 1))


class S2MultiResHead(nn.Module):
    """Decoder fine tokens -> one output plane per native resolution."""

    def __init__(self, in_dim, grid, p, head_feat=48):
        super().__init__()
        self.grid, self.p = grid, p
        self.heads = nn.ModuleDict()
        for name, (bands, ratio) in GROUPS.items():
            pr = max(p // ratio, 1)
            self.heads[name] = _PatchHead(in_dim, bands, pr, grid, head_feat)

    def forward(self, fine):                          # (B, g*g, in_dim)
        # GroupBatch, not a bare dict: the trainer calls .float() on the
        # decoder output and validation slices it, and those have to mean
        # the same thing for a multi-resolution source as for a plane.
        return GroupBatch({name: h(fine) for name, h in self.heads.items()})


class _PatchHead(nn.Module):
    """Per-token linear patch + conv smoothing at one resolution."""

    def __init__(self, dim, bands, p, grid, feat=48):
        super().__init__()
        self.p, self.feat, self.grid = p, feat, grid
        self.to_patch = nn.Linear(dim, feat * p * p)
        self.conv = nn.Sequential(
            nn.Conv2d(feat, feat, 3, 1, 1), nn.GELU(),
            nn.Conv2d(feat, bands, 3, 1, 1))

    def forward(self, tok):
        B, g, p, f = tok.shape[0], self.grid, self.p, self.feat
        x = self.to_patch(tok).view(B, g, g, f, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, f, g * p, g * p)
        return self.conv(x)


class S2Normalizer:
    """Per-tile per-band mean/std side channel: normalize before encode,
    restore after decode. `stats_vector` orders [means..., stds...] over
    the 13 bands (the layout the stats codec compresses)."""

    def normalize(self, x: dict):
        """Per-band-group standardization, returning the stats to transmit.

        Sentinel-2 bands differ in physical scale and per-scene
        illumination, so the model sees standardized inputs and the (mu, sd)
        side channel carries what was removed. Embedding-field sources need
        nothing here and return None from `normalizer`.
        """
        stats = {}
        out = GroupBatch()
        for name, t in x.items():
            mu = t.mean((2, 3), keepdim=True)
            sd = t.std((2, 3), keepdim=True).clamp_min(1e-4)
            out[name] = (t - mu) / sd
            stats[name] = (mu, sd)
        return out, stats

    def denormalize(self, x: dict, stats):
        return GroupBatch({name: t * stats[name][1] + stats[name][0]
                           for name, t in x.items()})

    @staticmethod
    def stats_vector(stats):
        mus = torch.cat([stats[n][0].flatten(1) for n in GROUPS], 1)
        sds = torch.cat([stats[n][1].flatten(1) for n in GROUPS], 1)
        return torch.cat([mus, sds], 1)               # (B, 26)


class S2Source(Source):
    name = "s2"
    grid_period = 6      # crop offsets must land on all three native grids

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        p = model_cfg.window // model_cfg.tiers[-1].grid
        assert p % 6 == 0, \
            ("S2 needs the fine-token patch size (window / finest grid) "
             f"divisible by 6 so every native grid decodes exactly; got {p}")
        self._norm = S2Normalizer()

    def prep(self, raw: dict, device):
        """Raw uint16 group dict -> normalized float dict + 10m validity."""
        x = GroupBatch()
        for name, t in raw.items():
            x[name] = t.to(device, non_blocking=True).float() / MAX_VALUE
        valid = x["10m"].sum(1) > 0
        xn, self._last_stats = self._norm.normalize(x)
        return xn, valid

    def last_stats(self):
        return self._last_stats

    def build_stem(self, model_cfg) -> nn.Module:
        return S2NativeStem(model_cfg.bands)

    def build_head(self, model_cfg, in_dim) -> nn.Module:
        fine = model_cfg.tiers[-1]
        return S2MultiResHead(in_dim, grid=fine.grid,
                              p=model_cfg.window // fine.grid,
                              head_feat=model_cfg.decoder.head_feat)

    def normalizer(self, model_cfg):
        return self._norm

    def side_bits(self, cfg) -> float:
        """The 26 per-band mean/std floats `prep` removes.

        With `model.stats_pca` set, this is the MEASURED mean packed size
        of the fitted basis (`scripts/fit_stats_pca_s2.py` stores it in the
        npz, so the rate a run reports is reproducible from the artifact
        rather than re-measured per run). The packer is tolerance-coded, so
        the per-window size varies; the mean is what an expected-rate
        figure wants, and the same file carries the p99 for anyone who
        needs the tail.

        Without a basis it falls back to raw f16, ~5x larger. That is the
        conservative direction, but it is not free: on a 1/4/16 pyramid the
        raw channel can exceed the coarse-only operating point outright,
        which is how a stats channel reaches a third of a bitstream
        without anyone noticing.
        """
        n = 2 * (len(BANDS_10M) + len(BANDS_20M) + len(BANDS_60M))
        path = getattr(cfg, "stats_pca", None) if cfg is not None else None
        if path is None:
            return float(n) * 16.0
        return float(_packed_bits(str(path)))

    def recon_stats(self, out: dict, x: dict, valid):
        losses = {}
        total = 0.0
        for name in GROUPS:
            mse = F.mse_loss(out[name], x[name])
            losses[f"mse_{name}"] = mse
            total = total + mse
        losses["mse"] = total / len(GROUPS)
        # proxy quality metric on the 10m plane (normalized space)
        err = ((out["10m"] - x["10m"]) ** 2).mean(1)
        losses["cos"] = 1.0 - err[valid].mean().clamp(max=1.0)
        return total / len(GROUPS), losses

    def report(self, model_cfg):
        """Reflectance, not an embedding field, so nothing here is
        inherited: quality is the spectral angle (it reads the band RATIOS
        that indices and classifiers use, which a per-band MSE averages
        away), RMSE accompanies it, and the preview is true colour off the
        10m group rather than a PCA projection. There is no block-mean
        floor because the per-tile mean is already transmitted as a side
        channel -- scoring against it would be scoring against part of the
        bitstream.

        CAVEAT ON THE ANGLE. `prep` standardizes each band per tile before
        the model sees anything, so this measures the angle between
        STANDARDIZED ANOMALY vectors, not between reflectance spectra. It
        is still the right shape of metric -- scale-invariant, sensitive to
        the relative pattern across bands rather than to overall
        brightness -- but it is not the spectral angle a remote-sensing
        reader will assume, and the two do not have the same numeric scale
        (random anomaly vectors sit at pi/2; random reflectance spectra,
        being all-positive, sit far below it). Reporting true-reflectance
        SAM means denormalizing through `last_stats()` first; until that
        exists, the label says what is actually computed.
        """
        from georvq.sources.report import (FIGURES, Metric, ReportSpec,
                                           band_rgb_preview)
        sam = Metric("sam", "spectral angle, standardized (rad)", _s2_sam,
                     higher_better=False, bounded=False)
        rmse = Metric("rmse", "RMSE (standardized)", _s2_rmse,
                      higher_better=False, bounded=False)
        # R=B04, G=B03, B=B02 within the 10m group's band order; the
        # preview selects that group out of the model's dict output.
        # no error_hist: it does tensor arithmetic on the whole batch to
        # build one signed-error distribution, which has no single meaning
        # across three native grids. A per-group version would be worth
        # having; it is not this figure.
        return ReportSpec(
            quality=sam, extra=(rmse,), baselines=(),
            figures=tuple(f for f in FIGURES if f != "error_hist"),
            preview=band_rgb_preview(2, 1, 0, select=_s2_10m))


def _s2_10m(t):
    return t["10m"] if isinstance(t, dict) else t


def _s2_sam(out, x, valid):
    from georvq.metrics import spectral_angle
    return spectral_angle(_s2_10m(out), _s2_10m(x), valid)


def _s2_rmse(out, x, valid):
    from georvq.metrics import masked_rmse
    return masked_rmse(_s2_10m(out), _s2_10m(x), valid)


_PACKED_BITS: dict[str, float] = {}


def _packed_bits(path: str) -> float:
    """Mean packed bits/window recorded by the basis fit.

    Cached: `side_bits` is read on every rate figure, and this would
    otherwise re-open the npz each time.
    """
    if path not in _PACKED_BITS:
        import numpy as np
        z = np.load(path)
        if "mean_bytes" not in z.files:
            raise KeyError(
                f"{path}: no 'mean_bytes' -- refit with "
                f"scripts/fit_stats_pca_s2.py, which records the measured "
                f"packed size that the rate figures charge.")
        _PACKED_BITS[path] = float(z["mean_bytes"]) * 8.0
    return _PACKED_BITS[path]
