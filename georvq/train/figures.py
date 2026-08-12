"""Training-time figures: R-D scatter with Pareto front, codebook health,
per-tier cos-dist by level, reconstruction panels."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from georvq.model.codec import tuple_bits
from georvq.train.validate import canonical_tuples, tname


def rd_figure(cos_by_tuple, vs, specs, tuples, step, ref_fronts=(), px=256,
              dist_log=False):
    """dist_log=True: y = 1 - cos on a log scale (separates the 0.99x's)."""
    bits = np.array([tuple_bits(specs, t) for t in tuples]) / px
    cos = np.array([cos_by_tuple[tname(t)] for t in tuples])
    L0s = np.array([t[0] for t in tuples])
    Y = (lambda c: 1.0 - np.asarray(c)) if dist_log else \
        (lambda c: np.asarray(c))
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    sc = ax.scatter(bits, Y(cos), c=L0s, cmap="viridis", s=22, zorder=3)
    for label, front in ref_fronts:
        fb, fc = zip(*front)
        ax.plot(fb, Y(fc), "--", lw=1.2, alpha=0.8, label=label)
    fig.colorbar(sc, label="L0 (coarse depth)")
    # Pareto front
    order = np.argsort(bits)
    fb, fc, best = [], [], -1
    for i in order:
        if cos[i] > best:
            fb.append(bits[i]); fc.append(cos[i]); best = cos[i]
    ax.plot(fb, Y(fc), "k-", lw=1, alpha=0.6, label="Pareto front")
    ax.axhline(Y(vs["cos_ceiling"]), color="r", ls="--", lw=1,
               label=f"ceiling ({vs['cos_ceiling']:.4f})")
    ax.axhline(Y(vs["cos_blockmean"]), color="grey", ls=":", lw=1,
               label=f"block-mean ({vs['cos_blockmean']:.4f})")
    if dist_log:
        ax.set(xscale="log", yscale="log", xlabel="bits / pixel",
               ylabel="1 - cosine (log)", ylim=(1e-3, 1e-1),
               title=f"pyramid R-D (distortion) @ step {step}")
    else:
        ax.set(xscale="log", xlabel="bits / pixel", ylabel="cosine",
               title=f"pyramid R-D @ step {step}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower left" if dist_log else "lower right")
    return fig


def codebook_figure(vs, specs, step):
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


def pca_rgb_basis(x, valid):
    """Fit a 3-PC RGB rendering basis on valid pixels of a batch."""
    C = x.shape[1]
    flat = x.permute(0, 2, 3, 1).reshape(-1, C)[valid.reshape(-1)]
    mu = flat.mean(0)
    _, _, v = torch.linalg.svd(flat - mu, full_matrices=False)
    pcs = v[:3]
    proj = (flat - mu) @ pcs.T
    lo = torch.quantile(proj, 0.02, 0)
    hi = torch.quantile(proj, 0.98, 0)
    return mu, pcs, lo, hi


def _pca_rgb(t, basis):
    mu, pcs, lo, hi = basis
    rgb = ((t.permute(0, 2, 3, 1) - mu) @ pcs.T - lo) / (hi - lo)
    return rgb.clamp(0, 1).cpu().numpy()


@torch.no_grad()
def recon_figure(model, source, val_s, step, n=8, mins=None, tuples=None,
                 val_loc=None):
    """Original vs reconstructions at named depth tuples, PCA-RGB rendered."""
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
    basis = pca_rgb_basis(val_x, val_valid)
    fig, axes = plt.subplots(len(rows), n, figsize=(1.3 * n, 1.45 * len(rows)),
                             constrained_layout=True)
    for r, (name, t) in enumerate(rows.items()):
        rgb = _pca_rgb(t, basis)
        for c in range(n):
            axes[r, c].imshow(rgb[c]); axes[r, c].axis("off")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([]); axes[r, 0].set_ylabel(name, fontsize=7)
    fig.suptitle(f"step {step}", fontsize=9)
    return fig


@torch.no_grad()
def stage1_recon_figure(model, source, val_s, sigma_max, step, n=6):
    """Original vs sigma-sweep reconstructions (stage-1 continuous mode)."""
    was_training = model.training
    model.eval()
    x, valid = source.prep(val_s[:n], val_s.device)
    var = x.reshape(x.shape[0], x.shape[1], -1).var(-1).mean(-1)
    sel = var.topk(min(n, x.shape[0])).indices
    x = x[sel]; valid = valid[sel]
    basis = pca_rgb_basis(x, valid)
    n_tiers = len(model.specs)
    rows = [("orig", _pca_rgb(x, basis))]
    for fr in (0.0, 0.25, 0.5, 1.0):
        sig = torch.full((x.shape[0], n_tiers), sigma_max * fr,
                         device=x.device)
        rows.append((f"σ={fr:.2f}×max",
                     _pca_rgb(model.forward_stage1(x, sig), basis)))
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
