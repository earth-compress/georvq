"""Sentinel-2 L1C source: 13 bands at three native resolutions.

Bands stay at their native ground sample distance end to end — no
resampling of the data itself:

- stem: per-resolution conv stacks run at native GSD, then upsample-and-
  fuse onto the 10m grid to form the single plane the tokenizer eats.
- head: per-resolution patch heads decode the fine tokens back to each
  native grid.
- normalizer: per-tile per-band mean/std, taken over each band's valid
  (non-zero) pixels, are removed before encoding and re-applied after
  decoding; the stats travel in the packet's side channel
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


def _cbg(cin, cout, k, stride=1, pad=0):
    """Conv, BatchNorm, GELU."""
    return [nn.Conv2d(cin, cout, k, stride, pad), nn.BatchNorm2d(cout),
            nn.GELU()]


class _ResBlock(nn.Module):
    """3x3 conv and 1x1 channel mix with a skip, BatchNorm after each."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        return x + self.bn2(F.gelu(self.conv2(F.gelu(self.bn1(self.conv1(x))))))


class S2ConvTileEncoder(nn.Module):
    """The ring-space beta_v2 encoder: a conv stack that reads the 10 m
    bands, halves to the 20 m grid and joins the 20 m bands, divides by
    three to the 60 m grid and joins the 60 m bands. Each 60 m pixel
    becomes one fine-tier cell (6 px at 10 m) and a pooled branch gives
    the single coarse-tier vector.

    With `mid_dim` and `mid_cell` set there is a third, middle tier
    between the two: each token reads a `mid_cell` x `mid_cell` block of
    the 60 m feature map through a conv whose kernel and stride are that
    block, so it sees its own block of fine cells and nothing outside it,
    then a 1x1 conv and a projection to `mid_dim`. Without them the
    module holds exactly beta_v2's parameters.

    Module names follow beta_v2's `TileVQAutoencoder` so its weights load
    by prefix. `width` is beta's stem width (128 in beta_v2).
    """

    def __init__(self, coarse_dim, fine_dim, width=128, mid_dim=None,
                 mid_cell=None):
        super().__init__()
        w, w2, w4 = width, 2 * width, 4 * width
        n10, n20, n60 = (GROUPS[g][0] for g in ("10m", "20m", "60m"))
        self.enc_10m = nn.Sequential(*_cbg(n10, w, 3, 1, 1),
                                     *_cbg(w, w, 3, 1, 1),
                                     *_cbg(w, w, 3, 2, 1))
        self.enc_fuse_20m = nn.Sequential(*_cbg(w + n20, w2, 1))
        self.enc_20m = nn.Sequential(*_cbg(w2, w2, 3, 1, 1),
                                     *_cbg(w2, w2, 3, 3, 0))
        self.enc_fuse_60m = nn.Sequential(*_cbg(w2 + n60, w4, 1))
        self.enc_res = _ResBlock(w4)
        self.hq_proj = nn.Conv2d(w4, fine_dim, 1)
        self.cls_res = _ResBlock(w4)
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_proj = nn.Linear(w4, coarse_dim)
        # built after beta_v2's modules, so that a two-tier encoder draws
        # the same initial weights from a given seed as before
        if (mid_dim is None) != (mid_cell is None):
            raise ValueError("mid_dim and mid_cell are set together")
        if mid_dim is not None:
            self.mid_down = nn.Sequential(*_cbg(w4, w4, mid_cell, mid_cell),
                                          *_cbg(w4, w4, 1))
            self.mid_proj = nn.Conv2d(w4, mid_dim, 1)
        else:
            self.mid_down = self.mid_proj = None

    def trunk(self, x):
        """The shared stack: the prepped batch -> the 60 m feature map
        (B, 4 * width, W/6, W/6)."""
        h = self.enc_10m(x["10m"])
        h = self.enc_fuse_20m(torch.cat([h, x["20m"]], 1))
        h = self.enc_20m(h)
        return self.enc_res(self.enc_fuse_60m(torch.cat([h, x["60m"]], 1)))

    def mid(self, h):
        """60 m feature map -> middle-tier tokens (B, g_mid**2, mid_dim)."""
        return self.mid_proj(self.mid_down(h)).flatten(2).transpose(1, 2)

    def forward(self, x, stats_bias=None):
        if stats_bias is not None:
            raise ValueError("S2ConvTileEncoder takes no stats conditioning")
        h = self.trunk(x)
        fine = self.hq_proj(h).flatten(2).transpose(1, 2)    # (B, g*g, D)
        coarse = self.cls_proj(self.cls_pool(self.cls_res(h)).flatten(1))
        if self.mid_down is None:
            return [coarse.unsqueeze(1), fine]
        return [coarse.unsqueeze(1), self.mid(h), fine]


def _check_tile_layout(model_cfg, what):
    """The beta_v2 conv encoder needs a whole-window coarse cell and a fine
    cell of one 60 m pixel, with optionally a middle tier between them
    whose cell is a whole number of fine cells.

    Returns the middle tier's cell size in fine cells, or None for two
    tiers."""
    t, win = model_cfg.tiers, model_cfg.window
    g_fine = win // 6
    ok = (len(t) in (2, 3) and win % 6 == 0 and t[0].grid == 1
          and t[-1].grid == g_fine)
    if ok and len(t) == 3:
        g_mid = t[1].grid
        ok = 1 < g_mid < g_fine and g_fine % g_mid == 0
    if not ok:
        raise ValueError(
            f"{what} needs tiers [1:..., window/6:...] or "
            f"[1:..., g_mid:..., window/6:...] with g_mid dividing window/6 "
            f"(window {win}, window/6 = {win / 6:g}); got "
            f"{[str(s) for s in t]}")
    return g_fine // t[1].grid if len(t) == 3 else None


class S2Normalizer:
    """Per-tile per-band mean/std side channel: normalize before encode,
    restore after decode. `stats_vector` orders [means..., stds...] over
    the 13 bands (the layout the stats codec compresses)."""

    per_sample = True       # denormalize needs the batch's own stats

    def normalize(self, x: dict, valid: dict | None = None):
        """Per-band standardization over valid pixels, returning the stats
        to transmit.

        Sentinel-2 bands differ in physical scale and per-scene
        illumination, so the model sees standardized inputs and the (mu, sd)
        side channel carries what was removed. `valid` maps each group to a
        per-band mask (B, C, h, w) of pixels that are not nodata. The mean
        and std of a band are taken over its own valid pixels only, and its
        invalid pixels are set to 0 (the band mean) in the output, so a
        window that straddles a swath edge is coded around the statistics
        of the data it holds rather than those of the DN 0 fill. The std is
        the unbiased estimate, as `Tensor.std` gives, so a fully valid
        window gets the same stats as before masking existed.
        """
        stats = {}
        out = GroupBatch()
        for name, t in x.items():
            if valid is None:
                m = torch.ones_like(t)
            else:
                m = valid[name].to(t.dtype)
            n = m.sum((2, 3), keepdim=True)
            mu = (t * m).sum((2, 3), keepdim=True) / n.clamp_min(1)
            var = ((t - mu) ** 2 * m).sum((2, 3), keepdim=True) \
                / (n - 1).clamp_min(1)
            sd = var.sqrt().clamp_min(1e-4)
            out[name] = (t - mu) / sd * m
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


def pixel_valid(band_valid: dict):
    """(B, W, W) mask on the 10 m grid of pixels where every band of every
    group holds data. The 20 m and 60 m masks are repeated onto the 10 m
    grid, so a 10 m pixel counts as valid only when the coarser pixels
    covering it are valid too."""
    v = None
    for name, (_, ratio) in GROUPS.items():
        g = band_valid[name].all(1)
        if ratio > 1:
            g = g.repeat_interleave(ratio, 1).repeat_interleave(ratio, 2)
        v = g if v is None else v & g
    return v


def group_valid(valid, name):
    """The 10 m pixel mask reduced to one group's grid: a coarse pixel is
    valid when every 10 m pixel it covers is. Returns (B, 1, h, w)."""
    ratio = GROUPS[name][1]
    v = valid.unsqueeze(1).float()
    if ratio > 1:
        v = -F.max_pool2d(-v, ratio)
    return v > 0.5


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
        self._stats_pca_bits = None    # set from a checkpoint's prep state

    def prep_state(self):
        """The PCA stats channel's charged size, for checkpoint embedding,
        so a checkpoint's rate does not depend on finding the artifact
        again or on the artifact staying unchanged."""
        if getattr(self.cfg, "stats_pca", None) is None:
            return {}
        return {"stats_pca_bits": self._pca_bits(self.cfg)}

    def load_prep_state(self, state):
        if "stats_pca_bits" in state:
            self._stats_pca_bits = float(state["stats_pca_bits"])

    def _pca_bits(self, cfg):
        if self._stats_pca_bits is not None:
            return self._stats_pca_bits
        return _packed_bits(resolve_artifact(cfg.stats_pca))

    def prep(self, raw: dict, device):
        """Raw uint16 group dict -> normalized float dict + 10 m validity.

        DN 0 is the L1C nodata value. It is masked per band for the
        statistics, and the returned mask marks the 10 m pixels where all
        13 bands hold data (`pixel_valid`), which is what the loss and the
        metrics score.
        """
        x = GroupBatch()
        band_valid = {}
        for name, t in raw.items():
            t = t.to(device, non_blocking=True)
            band_valid[name] = t != 0
            x[name] = t.float() / MAX_VALUE
        valid = pixel_valid(band_valid)
        xn, self._last_stats = self._norm.normalize(x, band_valid)
        return xn, valid

    def last_stats(self):
        return self._last_stats

    def build_stem(self, model_cfg) -> nn.Module:
        return S2NativeStem(model_cfg.bands)

    def build_head(self, model_cfg, in_dim, n_tiers=None,
                   out_px=None) -> nn.Module:
        grid, p = self.head_geom(model_cfg, n_tiers, out_px)
        return S2MultiResHead(in_dim, grid=grid, p=p,
                              head_feat=model_cfg.decoder.head_feat)

    def build_encoder(self, model_cfg) -> nn.Module:
        mid_cell = _check_tile_layout(model_cfg, "tokenizer.kind=source")
        t = model_cfg.tiers
        return S2ConvTileEncoder(
            t[0].dim, t[-1].dim, width=model_cfg.tokenizer.dim,
            mid_dim=t[1].dim if mid_cell else None, mid_cell=mid_cell)

    def normalizer(self, model_cfg):
        return self._norm

    #: Share of a rung's index bits the stats channel may take before a
    #: cheaper encoding is chosen.
    STATS_BUDGET = 0.02

    def _stats_encodings(self, cfg):
        """(name, bits) cheapest first.

        The PCA cost is the MEASURED mean packed size recorded by
        `scripts/fit_stats_pca_s2.py`, so a run's reported rate is
        reproducible from the artifact rather than from a constant that
        can drift away from the basis. A checkpoint that embedded the
        size (`prep_state`) is charged that value instead. The raw escapes
        are exact: a 1-byte code plus 26 values.
        """
        out = []
        if getattr(cfg, "stats_pca", None) is not None:
            out.append(("pca", self._pca_bits(cfg)))
        n = 2 * (len(BANDS_10M) + len(BANDS_20M) + len(BANDS_60M))
        out.append(("f16", n * 16 + 8.0))
        # No f32 tier. It was measured and buys nothing: at both 16-16-8
        # and full depth, f16 stats reproduce the exact-stats per-band
        # reflectance error to the digit (4.4e-3 and 4.3e-3), because
        # f16's ~3e-4 resolution is already an order of magnitude below
        # the model's own error. The extra 416 bits would be spent on
        # precision nothing can see.
        return out

    def side_bits(self, cfg, depths=None) -> float:
        """Per-window bits for the 26 mean/std values `prep` removes.

        The right encoding DEPENDS ON THE RATE, because the two costs move
        in opposite directions. Tolerance-coded PCA is ~78 bits but leaves
        a reconstruction floor of ~5e-3 reflectance RMSE; sending the
        vector verbatim removes the floor at 424 bits (f16) or 840 (f32).
        At the cheapest rungs 78 bits is already a large share of the
        bitstream and model error dwarfs the floor. At full depth the
        floor EQUALS the entire model error -- measured 4.3e-3 with exact
        stats against 6.5e-3 with fixed-tolerance packing -- while f16 is
        0.8% of the rate and closes it completely.

        So: the most precise encoding costing no more than `STATS_BUDGET`
        of the rung's index bits, falling back to the cheapest when even
        that does not fit. `depths=None` means "a whole packet", i.e. full
        depth.
        """
        enc = self._stats_encodings(cfg)
        if depths is None:
            depths = [sp.depth for sp in cfg.tiers]
        from georvq.model.codec import tuple_bits
        budget = tuple_bits(cfg.tiers, list(depths)) * self.STATS_BUDGET
        affordable = [b for _, b in enc if b <= budget]
        return max(affordable) if affordable else enc[0][1]

    def stats_tolerance(self, cfg, depths=None) -> float:
        """The `StatsPacker` tolerance matching `side_bits`' choice: <= 0
        means send the vector verbatim at f32, and a tolerance below what
        PCA can reach drives the packer's f16 escape."""
        enc = dict(self._stats_encodings(cfg))
        bits = self.side_bits(cfg, depths)
        if "f16" in enc and bits >= enc["f16"]:
            return 1e-4          # tighter than PCA can hit -> f16 escape
        return 0.01

    def recon_stats(self, out: dict, x: dict, valid):
        """Per-group MSE over valid pixels, averaged over the groups.

        Nodata pixels carry no target (the release restores them from the
        mask), so they are left out of the loss as DEM leaves them out."""
        losses = {}
        total = 0.0
        for name in GROUPS:
            mse = _masked_mse(out[name], x[name], group_valid(valid, name))
            losses[f"mse_{name}"] = mse
            total = total + mse
        losses["mse"] = total / len(GROUPS)
        # 1 - MSE over valid 10m pixels (normalized space); a proxy, not
        # a cosine
        err = ((out["10m"] - x["10m"]) ** 2).mean(1)
        losses["one_minus_mse_10m"] = 1.0 - err[valid].mean().clamp(max=1.0) \
            if bool(valid.any()) else err.new_tensor(0.0)
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

        WHY R2 AND NOT THE SPECTRAL ANGLE. The angle is arccos of a
        per-pixel cosine, so it is a DIRECTION-ONLY measure: it discards
        how bright a pixel is relative to its window and reads only the
        pattern across bands. `docs/SOURCE_ONBOARDING.md` makes magnitude
        spread the deciding test -- AlphaEarth is exactly unit-norm so
        cosine is unarguable there, Tessera spans 1.57x so an absolute
        loss measures something real -- and S2 spans **6.6x**, the widest
        of any source in that table. It is the strongest case for an
        absolute metric, not the weakest.

        It also has to agree with the objective: the loss is MSE, so
        selecting `model_best.pt` on a direction metric would select on
        something the run is not optimising. R2 here is
        `1 - sum_g MSE_g / sum_g Var_g` over the three native grids,
        weighted per group exactly as the loss weights them. On
        standardized inputs the denominator is ~1 per band, so R2 reads as
        one minus the loss -- the interpretable form of the same number.

        The angle stays as a diagnostic, because band RATIOS are what
        spectral indices read and a model can degrade them while its MSE
        looks fine. It is computed on standardized ANOMALY vectors,
        not reflectance spectra, so it is not the SAM a remote-sensing
        reader expects (random anomalies sit at pi/2; random all-positive
        reflectance sits far below); true-reflectance SAM needs
        denormalizing through `last_stats()` first.
        """
        from georvq.sources.report import (FIGURES, Metric, ReportSpec,
                                           band_rgb_preview)
        r2 = Metric("r2", "R² (variance explained)", _s2_r2,
                    higher_better=True, bounded=True)
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
            quality=r2, extra=(rmse, sam), baselines=(),
            figures=tuple(f for f in FIGURES if f != "error_hist"),
            preview=band_rgb_preview(2, 1, 0, select=_s2_10m))


def _s2_10m(t):
    return t["10m"] if isinstance(t, dict) else t


def _masked_mse(out, x, m):
    """Mean squared error over the pixels in m (B, 1, h, w), all bands;
    zero when the batch holds no valid pixel."""
    m = m.expand_as(x).to(x.dtype)
    return ((out - x) ** 2 * m).sum() / m.sum().clamp_min(1)


def _s2_r2(out, x, valid):
    """Variance explained across all three native grids, on valid pixels.

    Per-group MSE over per-group variance, summed with equal group weight
    so the metric weights the grids the way `recon_stats` weights them.
    Masked as the loss is: nodata pixels are zero in normalized space and
    carry no target.
    """
    num, den = 0.0, 0.0
    for name in GROUPS:
        t = x[name]
        m = group_valid(valid, name)
        num = num + _masked_mse(out[name], t, m)
        mf = m.expand_as(t).to(t.dtype)
        n = mf.sum().clamp_min(1)
        mean = (t * mf).sum() / n
        den = den + ((t - mean) ** 2 * mf).sum() / n
    return 1.0 - num / den.clamp_min(1e-8)


def _s2_sam(out, x, valid):
    from georvq.metrics import spectral_angle
    return spectral_angle(_s2_10m(out), _s2_10m(x), valid)


def _s2_rmse(out, x, valid):
    from georvq.metrics import masked_rmse
    return masked_rmse(_s2_10m(out), _s2_10m(x), valid)


_PACKED_BITS: dict[str, float] = {}


def resolve_artifact(path):
    """A config's artifact path as a file that exists. Relative paths are
    written relative to the repository root (`artifacts/...`), so the root
    is tried before the working directory."""
    from pathlib import Path
    p = Path(path)
    if p.is_absolute():
        cands = [p]
    else:
        cands = [Path(__file__).resolve().parents[2] / p, Path.cwd() / p]
    for c in cands:
        if c.is_file():
            return str(c)
    raise FileNotFoundError(
        f"artifact {path} not found; looked in "
        f"{', '.join(str(c) for c in cands)}")


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
