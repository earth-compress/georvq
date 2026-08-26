"""Training-time figures: R-D scatter with Pareto front, codebook health,
per-tier cos-dist by level, reconstruction panels."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from georvq.model.codec import tuple_bits
from georvq.sources.report import PCA_RGB
from georvq.train.validate import canonical_tuples, tname


def rd_figure(q_by_tuple, vs, specs, tuples, step, spec, ref_fronts=(),
              px=256, dist_log=False, side_bits=0.0):
    """Rate-distortion scatter for the run's own quality metric.

    `spec` is the resolved `ReportSpec`: it supplies the axis label, which
    direction is better, and the baselines to draw. dist_log=True plots
    `1 - m` on a log scale (separates the 0.99x's) and is only offered for
    a bounded metric -- `resolve_report` drops it otherwise, since there is
    no such transform for an RMSE or an angle in radians.
    """
    q = spec.quality
    bits = np.array([tuple_bits(specs, t, side_bits) for t in tuples]) / px
    vals = np.array([q_by_tuple[tname(t)] for t in tuples])
    L0s = np.array([t[0] for t in tuples])
    Y = (lambda c: 1.0 - np.asarray(c)) if dist_log else \
        (lambda c: np.asarray(c))
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    sc = ax.scatter(bits, Y(vals), c=L0s, cmap="viridis", s=22, zorder=3)
    for label, front in ref_fronts:
        fb, fc = zip(*front)
        ax.plot(fb, Y(fc), "--", lw=1.2, alpha=0.8, label=label)
    fig.colorbar(sc, label="L0 (coarse depth)")
    # Pareto front -- "better" is metric-dependent: an angle or an RMSE
    # improves DOWNWARD, so the frontier tracks the running min there.
    order = np.argsort(bits)
    fb, fc = [], []
    best = -np.inf if q.higher_better else np.inf
    for i in order:
        if (vals[i] > best) if q.higher_better else (vals[i] < best):
            fb.append(bits[i]); fc.append(vals[i]); best = vals[i]
    ax.plot(fb, Y(fc), "k-", lw=1, alpha=0.6, label="Pareto front")
    ceil = vs[f"{q.key}_ceiling"]
    ax.axhline(Y(ceil), color="r", ls="--", lw=1,
               label=f"ceiling ({ceil:.4f})")
    for b in spec.baselines:
        bv = vs.get(f"{q.key}_{b.key}")
        if bv is not None:
            ax.axhline(Y(bv), color="grey", ls=":", lw=1,
                       label=f"{b.label} ({bv:.4f})")
    if dist_log:
        ax.set(xscale="log", yscale="log", xlabel="bits / pixel",
               ylabel=f"1 - {q.label} (log)", ylim=(1e-3, 1e-1),
               title=f"pyramid R-D (distortion) @ step {step}")
    else:
        ax.set(xscale="log", xlabel="bits / pixel", ylabel=q.label,
               title=f"pyramid R-D @ step {step}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower left" if dist_log else "lower right")
    return fig


def codebook_figure(vs, specs, step):
    """One panel per tier: usage, perplexity and index entropy per RVQ
    level, each normalized to [0, 1] so tiers with different K are
    comparable side by side.

    Read this to spot a level that has COLLAPSED, not to judge quality:
    all three bars can sit near 1.0 on a codebook that explains almost
    none of the variance (lab/findings/perplexity-is-gameable.md).
    """
    fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 3.3),
                             constrained_layout=True, squeeze=False)
    for i, (ax, sp) in enumerate(zip(axes[0], specs)):
        Ls = np.arange(1, sp.depth + 1)
        x = np.arange(len(Ls))
        ax.bar(x - 0.27, [vs[f"pyr{i}_usage_L{L}"] for L in Ls], 0.27,
               label="usage")
        ax.bar(x, [vs[f"pyr{i}_perplexity_L{L}"] / sp.codes for L in Ls],
               0.27, label="perplexity / K")
        ax.bar(x + 0.27, [vs[f"pyr{i}_entropy_L{L}"] / np.log2(sp.codes)
                          for L in Ls], 0.27, label="entropy / nominal")
        ax.set(xlabel="RVQ level", ylim=(0, 1),
               title=f"tier {i} (grid {sp.grid}, K={sp.codes}) @ {step}")
        ax.set_xticks(x); ax.set_xticklabels(Ls)
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)
    return fig


def reset_figure(rst, specs, step):
    """One figure per reset event: what the reset actually did, per level.

    Three panels per tier -- codes revived as a fraction of K, perplexity
    before vs after, and residual before vs after -- all measured on the
    SAME batch either side of the reset, so the bars isolate the reset
    rather than a step of training.

    This exists because the per-level detail is 7 numbers x 8 levels x 3
    tiers; as individual wandb scalars that is 168 panels nobody can read.
    """
    n = len(specs)
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 8.4),
                             constrained_layout=True, squeeze=False)
    for i, sp in enumerate(specs):
        Ls = np.arange(1, sp.depth + 1)
        x = np.arange(len(Ls))

        def g(metric):
            return np.array([rst.get(f"reset/pyr{i}_L{L}_{metric}", np.nan)
                             for L in Ls], dtype=float)

        ax = axes[0][i]
        ax.bar(x, 100 * g("frac_reset"), 0.6, color="crimson")
        ax.set(title=f"tier {i} (K={sp.codes}) @ step {step}",
               ylabel="% of K revived")
        ax.set_xticks(x); ax.set_xticklabels(Ls); ax.grid(alpha=0.3, axis="y")

        ax = axes[1][i]
        pb, pa = g("perp_before") / sp.codes, g("perp_after") / sp.codes
        ax.bar(x - 0.2, pb, 0.4, label="before", color="0.6")
        ax.bar(x + 0.2, pa, 0.4, label="after", color="steelblue")
        ax.set(ylabel="perplexity / K", ylim=(0, 1)); ax.legend(fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(Ls); ax.grid(alpha=0.3, axis="y")

        ax = axes[2][i]
        rb, ra = g("resid_before"), g("resid_after")
        ax.bar(x - 0.2, rb, 0.4, label="before", color="0.6")
        ax.bar(x + 0.2, ra, 0.4, label="after", color="seagreen")
        ax.set(xlabel="RVQ level", ylabel="residual (mean sq err)")
        ax.legend(fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(Ls); ax.grid(alpha=0.3, axis="y")
    return fig



def cosdist_figure(cosdist, baseline, specs, step):
    """Per-tier cos-dist vs cumulative RVQ level (solid: this model; dotted:
    an optional {tier_name: [dists]} baseline)."""
    names = ["coarse", "mid", "fine"][:len(specs)] \
        or [f"tier{i}" for i in range(len(specs))]
    fig, axes = plt.subplots(1, len(specs), figsize=(3.7 * len(specs), 3.2),
                             constrained_layout=True)
    if len(specs) == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        cd = cosdist[t]                       # L0..depth (L0 = prior)
        ax.plot(range(len(cd)), cd, "o-", color="crimson", lw=1.6, ms=4,
                label="model")
        if baseline is not None and names[t] in baseline:
            bl = baseline[names[t]]           # L1..depth (no prior)
            ax.plot(range(1, len(bl) + 1), bl, ":", color="0.4", lw=1.6,
                    label="baseline")
        ax.set(yscale="log", xlabel="RVQ level (0=prior)",
               title=f"{names[t]} cos-dist @ {step}")
        ax.set_ylabel("1 - cos") if t == 0 else None
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    return fig


# Back-compat aliases: the PCA-RGB renderer moved into the report layer so
# that a source can declare a different one (grayscale relief, true colour)
# instead of every plane being forced through three principal components.
pca_rgb_basis = PCA_RGB.basis
_pca_rgb = PCA_RGB.render


@torch.no_grad()
def recon_figure(model, source, val_s, step, n=8, mins=None, tuples=None,
                 val_loc=None, preview=PCA_RGB):
    """Original vs reconstructions at named depth tuples, PCA-RGB rendered."""
    # eval() is load-bearing, not hygiene: VectorQuantizer.forward runs its
    # lazy `_data_init` under `self.training`, so drawing this figure in
    # train mode INITIALISES the codebooks -- from 8 val windows, by random
    # sampling rather than the configured init_mode, and at whatever step
    # the first figure lands on. It also lets an EMA update and a step
    # increment run on validation data. (stage1_recon_figure below has
    # always done this correctly; this one never did.)
    was_training = model.training
    model.eval()
    try:
        return _recon_figure(model, source, val_s, step, n, mins, tuples,
                             val_loc, preview)
    finally:
        if was_training:
            model.train()


def _recon_figure(model, source, val_s, step, n, mins, tuples, val_loc,
                  preview):
    """Reconstruction contact sheet: the original over one row per depth
    tuple, projected to RGB through a shared PCA basis.

    The basis is fitted on the ORIGINALS and reused for every row, so
    rows are comparable to each other and to the source; a per-row
    basis would hide exactly the errors this figure exists to show.
    Call `recon_figure`, which owns the eval-mode guard.
    """
    val_x, val_valid = source.prep(val_s[:n], val_s.device)
    loc = val_loc[:n] if val_loc is not None else None
    if tuples:
        pts = {tname(t): t for t in tuples}
    else:
        pts = canonical_tuples(model.specs, mins)
    _, cums = model.level_cumsums(val_x, loc=loc)
    rows = {"orig": val_x}
    for name, t in pts.items():
        rows[f"{name}\n({tname(t)})"] = model.eval_tuple(cums, t, val_x)
    basis = preview.basis(val_x, val_valid)
    fig, axes = plt.subplots(len(rows), n, figsize=(1.3 * n, 1.45 * len(rows)),
                             constrained_layout=True)
    for r, (name, t) in enumerate(rows.items()):
        rgb = preview.render(t, basis)
        for c in range(n):
            axes[r, c].imshow(rgb[c]); axes[r, c].axis("off")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([]); axes[r, 0].set_ylabel(name, fontsize=7)
    fig.suptitle(f"step {step}", fontsize=9)
    return fig


@torch.no_grad()
def stage1_recon_figure(model, source, val_s, sigma_max, step, n=6,
                        preview=PCA_RGB):
    """Original vs sigma-sweep reconstructions (stage-1 continuous mode)."""
    was_training = model.training
    model.eval()
    x, valid = source.prep(val_s[:n], val_s.device)
    var = x.reshape(x.shape[0], x.shape[1], -1).var(-1).mean(-1)
    sel = var.topk(min(n, x.shape[0])).indices
    x = x[sel]; valid = valid[sel]
    basis = preview.basis(x, valid)
    n_tiers = len(model.specs)
    rows = [("orig", preview.render(x, basis))]
    for fr in (0.0, 0.25, 0.5, 1.0):
        sig = torch.full((x.shape[0], n_tiers), sigma_max * fr,
                         device=x.device)
        rows.append((f"σ={fr:.2f}×max",
                     preview.render(model.forward_stage1(x, sig), basis)))
    fig, axes = plt.subplots(len(rows), x.shape[0],
                             figsize=(1.4 * x.shape[0], 1.5 * len(rows)),
                             constrained_layout=True)
    for r, (name, im) in enumerate(rows):
        for c in range(x.shape[0]):
            axes[r, c].imshow(im[c], interpolation="nearest")
            axes[r, c].axis("off")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle(f"Stage 1 recon @ step {step}", fontsize=9)
    if was_training:
        model.train()
    return fig


@torch.no_grad()
def error_hist_figure(model, source, val_s, step, tuples=None, mins=None,
                      val_loc=None, max_tokens=1024, chunk=128,
                      unit="source units"):
    """Signed reconstruction-error histograms, one colored line per depth
    tuple (default: the canonical R-D points): model (normalized) space
    plus, when the source has a per-tile normalizer, the denormalized
    source units (e.g. DEM metres)."""
    was_training = model.training
    model.eval()
    pts = ({tname(t): t for t in tuples} if tuples
           else canonical_tuples(model.specs, mins))
    norm = source.normalizer(source.cfg)
    rel = {k: [] for k in pts}
    absu = {k: [] for k in pts} if norm is not None else None
    for i in range(0, min(max_tokens, val_s.shape[0]), chunk):
        x, v = source.prep(val_s[i:i + chunk], val_s.device)
        loc = val_loc[i:i + chunk] if val_loc is not None else None
        _, cums = model.level_cumsums(x, loc=loc)
        stats = source.last_stats() if norm is not None else None
        for k, t in pts.items():
            out = model.eval_tuple(cums, t, x).float()
            m = v.unsqueeze(1).expand_as(out)
            rel[k].append((out - x)[m].cpu())
            if norm is not None:
                absu[k].append((norm.denormalize(out, stats)
                                - norm.denormalize(x, stats))[m].cpu())
    panels = [("model space (per-tile normalized)", rel)]
    if absu is not None:
        panels.append((unit, absu))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pts)))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6),
                             constrained_layout=True, squeeze=False)
    for ax, (name, errs) in zip(axes[0], panels):
        es = {k: torch.cat(v).numpy() for k, v in errs.items()}
        lim = max(np.quantile(np.abs(e), 0.999) for e in es.values())
        edges = np.linspace(-lim, lim, 202)
        for c, (k, e) in zip(colors, es.items()):
            h, _ = np.histogram(np.clip(e, -lim, lim), bins=edges)
            ax.stairs(h, edges, color=c, lw=1.2,
                      label=f"{k}  rmse={np.sqrt((e ** 2).mean()):.3g}  "
                            f"med={np.median(e):.2g}")
        ax.set_yscale("log")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("error (out - x)", fontsize=8)
        ax.legend(fontsize=6.5)
    fig.suptitle(f"recon error by depth tuple, step {step}", fontsize=9)
    if was_training:
        model.train()
    return fig
