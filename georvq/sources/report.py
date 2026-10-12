"""What a run reports: metrics, baselines, figures — declared per source.

Invariant 6 (`docs/OVERVIEW.md`) says sources own all data-specific
behavior. Reporting was the last place that wasn't true. `validate()` read
`recon_stats(...)[1]["cos"]` by name, scored a unit-norm block-mean
baseline, and drew a 3-component PCA-RGB contact sheet — three statements
about a many-band unit-norm embedding field, applied to every source that
came after it. The costs were concrete: a single-band elevation source had
to publish R² under the key `cos` to keep the shared logging path working,
and PCA-RGB cannot render one band at all.

So a source declares its reporting surface here, and the shared stack reads
the declaration instead of a hardcoded key:

- `quality` — the headline metric. It names the wandb scalars, labels the
  R-D y axis, and is what `model_best.pt` is selected on.
- `extra` — further diagnostics worth logging every validation.
- `baselines` — reference points that are meaningful for THIS field.
- `figures` — which plots to draw.
- `preview` — how to turn a plane into something you can look at.

Two gates keep a wrong declaration from reaching wandb. Every metric and
preview states the `min_bands` it is defined for, so asking for cosine on
one band (degenerate: it collapses to sign agreement) or PCA-RGB on one
band (undrawable) is refused. And `log.report.*` may only NARROW what the
source declares — naming something undeclared is a config error. Both fire
during config resolution, not at the first validation step an hour in.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from georvq.metrics import (centered_cos, masked_r2, masked_recon_stats,
                            masked_rmse, spectral_angle)

# ---------------------------------------------------------------- metrics


@dataclass(frozen=True)
class Metric:
    """A scalar quality/diagnostic measure over a validation batch.

    `fn(out, x, valid) -> scalar tensor`, the same signature (and the same
    model space) as `Source.recon_stats`, so a multi-group source supplies
    dict-aware functions without any special casing upstream.
    """
    key: str
    label: str                      # axis / print label, with units
    fn: Callable
    higher_better: bool = True
    # A bounded metric tops out at 1.0, so `1 - m` on a log axis separates
    # the 0.99x's — that is what `rd_figure(dist_log=True)` plots. An
    # unbounded one (RMSE, radians) has no such transform and skips it.
    bounded: bool = True
    # Bands the metric is DEFINED for. Cosine over a 1-band field is
    # sign agreement, not similarity; refusing it here is what stops a
    # scalar source from silently reporting a meaningless 0.99.
    min_bands: int = 1
    # The space the metric is defined in: "model" scores (out, x) as the
    # network sees them; "physical" scores prep.inverse of both. The
    # harness routes on this declaration -- nothing configures an
    # inversion point, and a physical metric under a non-invertible
    # prep is refused at report resolution.
    space: str = "model"

    def __call__(self, out, x, valid):
        return self.fn(out, x, valid)


COS = Metric(
    "cos", "cosine", lambda o, x, v: masked_recon_stats(o, x, v)[1],
    min_bands=2)
CCOS = Metric(
    "ccos", "centred cosine", centered_cos, min_bands=2)
R2 = Metric(
    "r2", "R² (variance explained)", masked_r2, min_bands=1)
RMSE = Metric(
    "rmse", "RMSE (model units)", masked_rmse,
    higher_better=False, bounded=False, min_bands=1)
SAM = Metric(
    "sam", "spectral angle (rad)", spectral_angle,
    higher_better=False, bounded=False, min_bands=2)

METRICS = {m.key: m for m in (COS, CCOS, R2, RMSE, SAM)}


# -------------------------------------------------------------- baselines


@dataclass(frozen=True)
class Baseline:
    """A rate-free reference the R-D plot draws as a horizontal line.

    `fn(x, valid) -> reconstruction`, scored with the run's quality metric.
    A baseline is declared per source because what counts as trivial is a
    property of the field: replacing a window by its own mean vector is a
    meaningful floor for an embedding field and nonsense for a source that
    already transmits that mean as a side channel.
    """
    key: str
    label: str
    fn: Callable
    min_bands: int = 1


def _blockmean_unit(x, valid):
    """Each window replaced by its L2-normalized valid-pixel mean vector.

    The normalization is what makes this the right floor for a unit-norm
    embedding field: the comparison is against the best CONSTANT direction,
    not against a shrunken one that cosine would flatter.
    """
    vm = valid.unsqueeze(1).float()
    mean = (x * vm).sum((2, 3)) / vm.sum((2, 3)).clamp_min(1)
    mean = F.normalize(mean, dim=1)
    return mean[:, :, None, None].expand_as(x)


BLOCKMEAN = Baseline("blockmean", "block-mean", _blockmean_unit, min_bands=2)

BASELINES = {b.key: b for b in (BLOCKMEAN,)}


# --------------------------------------------------------------- previews


@dataclass(frozen=True)
class Preview:
    """How to render a plane as an image for the reconstruction contact
    sheet. `basis(x, valid)` is fitted once on the ORIGINALS and reused for
    every row, so rows stay comparable; `render(t, basis) -> (B, H, W, 3)`.
    """
    key: str
    basis: Callable
    render: Callable
    min_bands: int = 1


def _pca_basis(x, valid):
    C = x.shape[1]
    flat = x.permute(0, 2, 3, 1).reshape(-1, C)[valid.reshape(-1)]
    mu = flat.mean(0)
    _, _, v = torch.linalg.svd(flat - mu, full_matrices=False)
    pcs = v[:3]
    proj = (flat - mu) @ pcs.T
    return mu, pcs, torch.quantile(proj, 0.02, 0), torch.quantile(proj, 0.98, 0)


def _pca_render(t, basis):
    mu, pcs, lo, hi = basis
    rgb = ((t.permute(0, 2, 3, 1) - mu) @ pcs.T - lo) / (hi - lo)
    return rgb.clamp(0, 1).cpu().numpy()


def _gray_basis(x, valid):
    """Robust display range from the valid pixels of the originals."""
    vals = x[valid.unsqueeze(1).expand_as(x)]
    return (torch.quantile(vals, 0.02), torch.quantile(vals, 0.98))


def _gray_render(t, basis):
    lo, hi = basis
    g = ((t[:, 0] - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
    return g.unsqueeze(-1).expand(*g.shape, 3).cpu().numpy()


PCA_RGB = Preview("pca_rgb", _pca_basis, _pca_render, min_bands=3)
GRAY = Preview("gray", _gray_basis, _gray_render, min_bands=1)

PREVIEWS = {p.key: p for p in (PCA_RGB, GRAY)}


def band_rgb_preview(r, g, b, select=None) -> Preview:
    """A true-colour `Preview` over three band indices.

    `select` picks the plane to render out of whatever the model emits,
    so a multi-group source can point it at one native grid (S2 renders
    its 10m group). It defaults to the identity for single-plane sources.
    """
    idx = [r, g, b]
    pick = select or (lambda t: t)

    def basis(x, valid):
        sel = pick(x)[:, idx]
        vals = sel[valid.unsqueeze(1).expand_as(sel)]
        return (torch.quantile(vals, 0.02), torch.quantile(vals, 0.98))

    def render(t, bs):
        lo, hi = bs
        rgb = ((pick(t)[:, idx] - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
        return rgb.permute(0, 2, 3, 1).cpu().numpy()

    # min_bands is about the SELECTED plane, which the shared band count
    # cannot describe, so a selecting preview opts out of that check.
    return Preview("band_rgb", basis, render,
                   min_bands=1 if select else max(idx) + 1)


# ------------------------------------------------------------------ specs

# Figures the trainer knows how to draw. `cosdist` additionally requires a
# conditioned model (it plots the per-tier prior-to-level cascade), and
# `rd_log` an unbounded-free metric; both are filtered where inapplicable.
FIGURES = ("rd", "rd_log", "codebook", "recon", "reset", "cosdist",
           "error_hist", "rd_err")


@dataclass(frozen=True)
class ReportSpec:
    """The full reporting surface of one source."""
    quality: Metric
    extra: tuple[Metric, ...] = ()
    baselines: tuple[Baseline, ...] = ()
    figures: tuple[str, ...] = FIGURES
    preview: Optional[Preview] = None
    #: Physical unit of the source's DEQUANTIZED values (e.g. "metres").
    #: Only the error histogram uses it, to label the panel it draws in
    #: source units via the normalizer; "source units" when unstated.
    units: str = "source units"

    @property
    def metrics(self) -> tuple[Metric, ...]:
        """Quality first, then the extras — the order they are logged in."""
        return (self.quality,) + tuple(self.extra)

    def metric(self, key) -> Metric:
        for m in self.metrics:
            if m.key == key:
                return m
        raise KeyError(key)

    def has_figure(self, name) -> bool:
        return name in self.figures


def embedding_field_spec(preview=PCA_RGB) -> ReportSpec:
    """The reporting surface for a dense per-pixel vector field.

    Cosine is the headline because the field's information is in the
    DIRECTION of each pixel vector; `ccos` accompanies it because any model
    with a transmitted mean inflates plain cosine (`lab/findings/transmitted-mean-inflates-cosine.md`), and
    a coarse-only decode reading 0.9154 while contributing no spatial
    structure is exactly the failure it catches.
    """
    # no error histogram: without a per-tile normalizer its second panel
    # is a duplicate of the first, so it costs a decode for nothing
    return ReportSpec(quality=COS, extra=(CCOS,), baselines=(BLOCKMEAN,),
                      preview=preview,
                      figures=tuple(f for f in FIGURES if f != "error_hist"))


def scalar_field_spec(quality=R2, preview=GRAY,
                      units="source units") -> ReportSpec:
    """The reporting surface for a single-plane scalar field (elevation,
    backscatter): variance explained plus RMSE, no direction-based metric,
    no unit-norm block-mean floor, and a grayscale preview.

    Includes the signed-error histogram, which such a source can render in
    its own physical unit (`units`) through its normalizer -- the form the
    result is actually judged in.
    """
    return ReportSpec(quality=quality, extra=(RMSE,), baselines=(),
                      preview=preview, units=units)


def replace_extra(spec, add=(), drop=()) -> ReportSpec:
    """A copy of `spec` with extra diagnostics added and/or removed by key.
    Lets a source start from a ready-made shape and adjust one term rather
    than restate the whole surface."""
    keep = tuple(m for m in spec.extra if m.key not in set(drop))
    have = {m.key for m in keep}
    return replace(spec, extra=keep + tuple(m for m in add
                                            if m.key not in have))


# ------------------------------------------------------------- resolution


class ReportConfigError(ValueError):
    """A `log.report.*` selection the source does not support."""


def _narrow(declared, wanted, kind):
    """Keep the declared items named by `wanted`, in the declared order."""
    have = {d.key if hasattr(d, "key") else d: d for d in declared}
    missing = [w for w in wanted if w not in have]
    if missing:
        raise ReportConfigError(
            f"log.report.{kind}: {missing} not declared by this source "
            f"(declared: {sorted(have)}). Reporting is narrowed from what "
            f"the source offers, never widened — add it to the source's "
            f"ReportSpec if it is meaningful for this field.")
    return tuple(have[w] for w in wanted)


def _check_bands(items, bands, kind):
    bad = [(i.key, i.min_bands) for i in items if bands < i.min_bands]
    if bad:
        detail = ", ".join(f"{k} needs >= {n} bands" for k, n in bad)
        raise ReportConfigError(
            f"{kind} undefined for a {bands}-band field: {detail}. "
            f"(Cosine over one band is sign agreement, not similarity; a "
            f"3-component PCA-RGB preview of one band cannot be drawn.)")


def resolve_report(source, cfg) -> ReportSpec:
    """The source's declaration, narrowed by `log.report.*` and checked
    against the model.

    Called once at the top of training so a bad selection fails there
    rather than at the first validation an hour in.
    """
    spec = source.report(cfg.model)
    rc = cfg.log.report
    bands = cfg.model.bands

    if rc.metrics is not None:
        # the FIRST listed metric becomes the headline: it names the
        # scalars, labels the R-D axis and selects model_best.pt
        chosen = _narrow(spec.metrics, rc.metrics, "metrics")
        spec = replace(spec, quality=chosen[0], extra=chosen[1:])
    if rc.baselines is not None:
        spec = replace(spec,
                       baselines=_narrow(spec.baselines, rc.baselines,
                                         "baselines"))
    if rc.figures is not None:
        unknown = [f for f in rc.figures if f not in FIGURES]
        if unknown:
            raise ReportConfigError(
                f"log.report.figures: unknown {unknown}; have {list(FIGURES)}")
        spec = replace(spec, figures=tuple(f for f in rc.figures
                                           if f in spec.figures))
    if rc.preview is not None:
        if rc.preview == "none":
            spec = replace(spec, preview=None)
        elif spec.preview is None or rc.preview != spec.preview.key:
            raise ReportConfigError(
                f"log.report.preview: {rc.preview!r} is not this source's "
                f"preview ({spec.preview.key if spec.preview else None!r}); "
                f"'none' disables it.")

    _check_bands(spec.metrics, bands, "metrics")
    _check_bands(spec.baselines, bands, "baselines")
    if spec.preview is not None and bands < spec.preview.min_bands:
        _check_bands([spec.preview], bands, "preview")

    # Drop figures whose inputs this run cannot produce, rather than
    # failing: they are a property of the model, not of the request.
    drop = set()
    if not spec.quality.bounded:
        drop.add("rd_log")          # no 1 - m log axis for an open metric
    if spec.preview is None:
        drop.add("recon")
    if drop:
        spec = replace(spec, figures=tuple(f for f in spec.figures
                                           if f not in drop))
    return spec
