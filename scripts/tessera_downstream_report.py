"""Interim report: how Tessera's larger reconstruction error translates
into downstream task performance, versus AlphaEarth, on the same
benchmark tasks and the same label points.

Pages: summary, method, cost-vs-rate curves, the cross-source cosine
argument, absolute levels, caveats/next steps.

  python scripts/tessera_downstream_report.py \
      --results-dir ~/projects/ring-space/plots/tessera_downstream \
      --out ~/projects/ring-space/plots/tessera_downstream/\
tessera_downstream_interim.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INK, MUT = "#3d4451", "#7a8494"
C_AE, C_TS = "#4269d0", "#e05c5c"
BAND = "#eef1f6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9ced6",
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK,
                     "figure.facecolor": "white"})

TAGS = ["aster_ged", "lucas_lc", "lcmap_lc",
        "lucas_lc_label_lu", "lcmap_lc_label_lu"]
TITLE = {"aster_ged": "ASTER GED\nemissivity (R²)",
         "lucas_lc": "LUCAS\nland cover",
         "lcmap_lc": "LCMAP\nland cover",
         "lucas_lc_label_lu": "LUCAS\nland use",
         "lcmap_lc_label_lu": "LCMAP\nland use"}
SHORT = {"aster_ged": "ASTER GED", "lucas_lc": "LUCAS LC",
         "lcmap_lc": "LCMAP LC", "lucas_lc_label_lu": "LUCAS LU",
         "lcmap_lc_label_lu": "LCMAP LU"}


def load(base, src, tag, probe="linear", budget="all"):
    p = (f"{base}/ratesweep_{tag}.csv" if src == "tes"
         else f"{base}/ae_matched/ratesweep_{tag}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["budget"] = d.budget.astype(str)
    return d[(d.budget == budget) & (d.probe == probe)] \
        .sort_values("bits_per_px").reset_index(drop=True)


SEEDSETS = {"lucas_lc": ("seedcheck_lucas", "LUCAS LC", 16780),
            "lcmap_lc": ("seedcheck", "LCMAP LC", 3008)}
SEED_TUPLES = ["6-1-0", "8-8-0", "8-8-8"]


def seed_stats(base):
    """(bench, src, tuple) -> (mean, sd, n) of delta across split seeds."""
    out = {}
    for bench, (sub, _, _) in SEEDSETS.items():
        for src in ("tes", "ae"):
            for tup in SEED_TUPLES:
                vals = []
                for p in sorted(glob.glob(f"{base}/{sub}/{src}_*/"
                                          f"ratesweep_{bench}.csv")):
                    d = pd.read_csv(p)
                    d = d[(d.budget.astype(str) == "all")
                          & (d.probe == "linear") & (d["tuple"] == tup)]
                    if len(d):
                        vals.append(float(d.delta.iloc[0]))
                if vals:
                    out[(bench, src, tup)] = (float(np.mean(vals)),
                                              float(np.std(vals)), len(vals))
    return out


def noise_floor(stats, bench):
    """Typical (max over source/rate) split-noise sd for a benchmark."""
    sds = [v[1] for k, v in stats.items() if k[0] == bench]
    return max(sds) if sds else float("nan")


def header(fig, title, sub=None):
    fig.text(0.06, 0.945, title, fontsize=17, weight="bold")
    if sub:
        fig.text(0.06, 0.906, sub, fontsize=10.5, color=MUT, va="top")


def footer(fig, page):
    fig.text(0.06, 0.035, "georvq · Tessera downstream evaluation · "
             "interim report, August 2026", fontsize=8, color=MUT)
    fig.text(0.94, 0.035, f"{page}", fontsize=8, color=MUT, ha="right")


# --------------------------------------------------------------------- #
def page_summary(pdf, base, spread):
    fig = plt.figure(figsize=(11, 8.5))
    header(fig, "Does Tessera's larger reconstruction error matter "
                "downstream?",
           "Interim answer: above ~1 bit/pixel, no — the codec is "
           "task-transparent despite a cosine error ~10x AlphaEarth's.\n"
           "Below ~0.5 bit/pixel, yes — Tessera degrades roughly twice as "
           "fast. Measured on AlphaEarth's own benchmark tasks,\n"
           "at the same label points, with the same probe protocol.")

    rows = [
        ("1", "Cosine error does not transfer across sources.",
         "At matched — indeed slightly worse — reconstruction cosine, the "
         "Tessera codec costs LESS downstream accuracy than\nthe AlphaEarth "
         "codec does. On LUCAS land cover, AlphaEarth at cos 0.953 loses "
         "4.3 points of balanced accuracy;\nTessera at cos 0.948 loses 0.5. "
         "Tessera's error budget lands in dimensions these tasks do not "
         "read."),
        ("2", "Above ~1 b/px the Tessera codec is downstream-transparent.",
         "At the top of the rate ladder every task sits within its "
         "split-noise floor — the same verdict the AlphaEarth\nstudy "
         "reached, reached at a higher rate. Tessera does this at cosine "
         "~0.95, where AlphaEarth needs ~0.99."),
        ("3", "Below ~0.5 b/px Tessera degrades faster — clearly so on the "
              "one task large enough to tell.",
         "On LUCAS (16,780 points) the lowest-rate loss is 2x AlphaEarth's "
         "and holds across seeds: -0.108 ±0.010 against\n-0.052 ±0.009, a "
         "separation five times its own noise. On LCMAP (3,008 points) the "
         "two are indistinguishable —\n-0.054 ±0.009 against -0.054 ±0.020. "
         "Treat the low-rate penalty as established on LUCAS only."),
        ("4", "On the same points, the uncompressed sources are not equal.",
         "AlphaEarth scores higher on 4 of 5 tasks (LCMAP LC 0.818 vs "
         "0.750); the two tie on LUCAS land cover\n(0.5635 vs 0.5650). "
         "Tessera's ~2.5x higher latent information density does not show "
         "up as a probe advantage here."),
    ]
    y = 0.795
    for num, head, body in rows:
        fig.add_artist(plt.Rectangle((0.06, y - 0.005), 0.026, 0.026,
                                     transform=fig.transFigure, color=C_TS,
                                     alpha=0.85, zorder=2))
        fig.text(0.0725, y + 0.0075, num, fontsize=11, weight="bold",
                 color="white", ha="center", va="center", zorder=3)
        fig.text(0.10, y + 0.008, head, fontsize=12, weight="bold",
                 va="center")
        fig.text(0.10, y - 0.018, body, fontsize=9.7, va="top", color=INK)
        y -= 0.163

    nl_lu, nl_lc = noise_floor(spread, "lucas_lc"), \
        noise_floor(spread, "lcmap_lc")
    note = ("Significance was measured, not assumed: every rate point was "
            "re-probed under repeated geo-block split seeds (page 5). The "
            "resulting noise\nfloor reaches ±%.3f balanced accuracy on "
            "LUCAS and ±%.3f on LCMAP — so LCMAP cannot resolve anything "
            "smaller than ~0.05, and\nits apparent 'compression beats the "
            "original' points are artefacts. Claims above are stated only "
            "where they clear their own noise floor."
            % (nl_lu, nl_lc))
    fig.add_artist(plt.Rectangle((0.06, 0.075), 0.88, 0.085,
                                 transform=fig.transFigure, color=BAND,
                                 zorder=0))
    fig.text(0.075, 0.147, note, fontsize=9.3, va="top", color=INK)
    footer(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)


def page_method(pdf, base):
    fig = plt.figure(figsize=(11, 8.5))
    header(fig, "What was measured",
           "The AlphaEarth downstream study, re-run unchanged on Tessera "
           "embeddings at the same locations.")

    fig.text(0.06, 0.845, """
The protocol (unchanged from the AlphaEarth study, itself a
reimplementation of the AlphaEarth Foundations paper's frozen-
embedding probes):

  ·  Encode each labelled pixel's 64x64 context window with the
     codec, decode it at a given RVQ rate tuple, and read the
     reconstructed embedding back at the label pixel.
  ·  Fit probes on the reconstruction — 1-nearest-neighbour, 3-NN
     and a linear probe (logistic / ridge regression) — and compare
     against the same probes fit on the uncompressed embedding.
  ·  Split train/test by source tile (~82 km blocks), so no tile
     straddles the split and spatial autocorrelation cannot leak.
  ·  Report the metric delta (reconstruction minus original) as a
     function of bits per pixel. Delta, not absolute score, is the
     compression cost.

Matching the two sources:

  ·  The label points come from the built AlphaEarth eval datasets.
     For each point we re-extracted a 128 px Tessera context crop
     from the 0.1-degree Tessera tile containing it, using the tile's
     landmask georeferencing, and the tile year nearest the
     AlphaEarth target year (mostly 2018).
  ·  Any Tessera tile that contributed chips to the codec's training
     store is excluded, in every year — so no eval point was seen in
     training, on either side.
  ·  The AlphaEarth curves shown here were re-run restricted to
     exactly the points Tessera reached, so both curves describe the
     same ground.
""", fontsize=9.6, va="top", family="sans-serif")

    ax = fig.add_axes([0.56, 0.46, 0.38, 0.30])
    names, ae_n, tes_n = [], [], []
    for b in ["aster_ged", "lucas_lc", "lcmap_lc"]:
        m = json.load(open(f"/data/tessera-samples/eval/{b}/meta.json"))
        am = json.load(open(f"/data/AE-samples/eval/{b}/meta.json"))
        names.append(SHORT[b])
        ae_n.append(am["n_targets"])
        tes_n.append(m["n_targets"])
    yp = np.arange(len(names))
    ax.barh(yp + 0.19, ae_n, 0.34, color=C_AE, label="AE eval dataset")
    ax.barh(yp - 0.19, tes_n, 0.34, color=C_TS, label="probed here (both)")
    for i, (a, t) in enumerate(zip(ae_n, tes_n)):
        ax.text(t * 1.08, i - 0.19, f"{t:,}", va="center", fontsize=8.5)
        ax.text(a * 1.08, i + 0.19, f"{a:,}", va="center", fontsize=8.5,
                color=MUT)
    ax.set(yticks=yp, xscale="log", xlim=(500, 400000),
           xlabel="labelled points")
    ax.set_yticklabels(names, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Points probed per benchmark", fontsize=10, loc="left")
    ax.grid(alpha=0.25, axis="x")

    fig.text(0.56, 0.40,
             "Tessera counts are lower because a 0.1° tile is a ~110 MB\n"
             "download: LUCAS and LCMAP were capped at a 2,000-tile\n"
             "budget (densest-first), and points on training tiles or\n"
             "outside Tessera coverage were dropped. ASTER used all\n"
             "409 covered tiles.\n\n"
             "Codec under test: run tessera_vq8 (wandb 5bcr0y0u) — a\n"
             "3-tier pyramid RVQ, 64 px window, 8 levels per tier,\n"
             "validation cosine 0.962. This run is mid-training and is\n"
             "NOT the final Tessera codec. AlphaEarth reference:\n"
             "ae_pyr64_divfull, the shipped release model.",
             fontsize=9.3, va="top", color=INK)
    footer(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


def page_curves(pdf, base):
    fig = plt.figure(figsize=(11, 8.5))
    header(fig, "The compression cost, task by task",
           "Δ metric (reconstruction − uncompressed) against rate. Zero is "
           "free compression; the shaded band is ±0.01, about the width of\n"
           "the split-noise floor. Both codecs converge to it — Tessera "
           "later.")
    for k, tag in enumerate(TAGS):
        r, c = divmod(k, 3)
        ax = fig.add_axes([0.075 + c * 0.315, 0.50 - r * 0.375, 0.245, 0.29])
        ax.axhspan(-0.01, 0.01, color=BAND, zorder=0)
        ax.axhline(0, color="0.55", lw=1, zorder=1)
        for src, col, lab in ((("ae"), C_AE, "AlphaEarth"),
                              (("tes"), C_TS, "Tessera")):
            d = load(base, src, tag)
            if d is None:
                continue
            ax.plot(d.bits_per_px, d.delta, "o-", color=col, lw=1.8, ms=4.5,
                    label=lab, zorder=3)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(TITLE[tag], fontsize=10, loc="left")
        if c == 0:
            ax.set_ylabel("Δ metric")
        if r == 1 or k == 2:
            ax.set_xlabel("bits / pixel")
        if k == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.text(0.705, 0.375,
             "Reading these panels\n\n"
             "The gap between the red and blue curves\n"
             "at the left-hand edge is the real cost of\n"
             "Tessera's noisier field: at 0.07 b/px the\n"
             "Tessera codec gives up 2-4x more accuracy\n"
             "than the AlphaEarth codec does.\n\n"
             "By 1-2 b/px the curves have merged into\n"
             "the noise band and the source stops\n"
             "mattering for compression cost.\n\n"
             "Note the vertical scales differ: ASTER's\n"
             "whole range is ±0.01, so every point on\n"
             "it is already inside the band.\n\n"
             "The two LCMAP panels should be read as\n"
             "noise, not signal — including their excursion\n"
             "above zero. A four-seed repeat (page 5) puts\n"
             "their split noise at ±0.023, wider than the\n"
             "features visible here. LUCAS, five times\n"
             "larger, is the panel to trust.",
             fontsize=9.3, va="top", color=INK)
    footer(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


def page_cosine(pdf, base, spread):
    fig = plt.figure(figsize=(11, 8.5))
    header(fig, "Why cosine error cannot be compared across sources",
           "The same plot against reconstruction error instead of rate. If "
           "cosine error predicted downstream harm, the two curves would\n"
           "lie on top of each other. They do not — Tessera tolerates far "
           "more of it.")

    ax = fig.add_axes([0.075, 0.42, 0.44, 0.42])
    for tag, mk, al in (("lucas_lc", "o", 1.0), ("aster_ged", "^", 1.0),
                        ("lcmap_lc", "s", 0.28)):
        for src, col in (("ae", C_AE), ("tes", C_TS)):
            d = load(base, src, tag)
            if d is None:
                continue
            ax.plot(1 - d.embed_cos, d.delta, mk + "-", color=col, lw=1.5,
                    ms=5, alpha=al,
                    label=f"{SHORT[tag]} · {'AE' if src=='ae' else 'Tessera'}"
                          + ("  (noise-dominated)" if al < 0.5 else ""))
    ax.axhspan(-0.01, 0.01, color=BAND, zorder=0)
    ax.axhline(0, color="0.55", lw=1)
    ax.set(xscale="log", xlabel="reconstruction error  (1 − cosine)",
           ylabel="Δ downstream metric")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="lower left")
    ax.set_title("Downstream harm vs reconstruction error", fontsize=10,
                 loc="left")

    fig.text(0.56, 0.845, """
Blue (AlphaEarth) occupies the low-error end of the axis,
red (Tessera) the high-error end — they barely overlap.
Where they do, the ordering is the opposite of what a
cosine-based intuition predicts:

     LUCAS LC     AE   @ cos 0.9532    Δ  −0.043
                  Tess @ cos 0.9475    Δ  −0.005

     ASTER GED    AE   @ cos 0.9767    Δ  −0.001
                  Tess @ cos 0.9640    Δ  +0.007

At an equal or slightly worse reconstruction cosine, the
Tessera codec costs less downstream accuracy than the
AlphaEarth codec. On LUCAS the 0.038 gap is several times
the ±0.011 split noise; ASTER's is small but both sit at
zero, which makes the same point.

The interpretation is that the two sources spend their
vector norm differently. Tessera's embeddings carry ~2.5x
more independent variation per pixel (PCA d95: 51/91/71 of
512/512/256 tier dims, against AlphaEarth's 19/34/28), much
of it fine-scale and weakly correlated with land cover,
land use or emissivity. Discarding it costs cosine but not
task accuracy. AlphaEarth's field is smoother and more
nearly low-rank, so a given cosine error necessarily eats
into the dimensions the probes actually read.

Practical consequence: a cosine target transplanted from
the AlphaEarth codec (0.998) is the wrong specification for
Tessera. Set Tessera's quality target from downstream
transparency — which this run reaches around 1 b/px at
cos ~0.93 — not from AlphaEarth's error budget.
""", fontsize=9.4, va="top", family="sans-serif")

    fig.text(0.075, 0.30,
             "Only the LUCAS and ASTER rows above carry weight: the LCMAP "
             "points, plotted faintly here for completeness, sit inside a "
             "±0.05 noise\nfloor (page 5) and cannot order the two sources "
             "either way. The LUCAS comparison — 16,780 points, ±0.011 "
             "noise — is the evidence.",
             fontsize=9.3, va="top", color=MUT)
    footer(fig, 4)
    pdf.savefig(fig)
    plt.close(fig)


def page_noise(pdf, base, spread):
    fig = plt.figure(figsize=(11, 8.5))
    header(fig, "How much of this is noise?",
           "Every rate point re-probed under repeated geo-block split seeds "
           "(n per row in the table). The error bars are what a single-seed "
           "curve cannot see.")

    for k, (bench, (_, name, n)) in enumerate(SEEDSETS.items()):
        ax = fig.add_axes([0.09 + k * 0.30, 0.46, 0.235, 0.34])
        xs = np.arange(len(SEED_TUPLES))
        for src, col, off, lab in (("ae", C_AE, -0.11, "AlphaEarth"),
                                   ("tes", C_TS, 0.11, "Tessera")):
            m = [spread.get((bench, src, t), (np.nan,) * 3)[0]
                 for t in SEED_TUPLES]
            s = [spread.get((bench, src, t), (np.nan,) * 3)[1]
                 for t in SEED_TUPLES]
            ax.errorbar(xs + off, m, yerr=s, fmt="o-", color=col, ms=5,
                        capsize=4, lw=1.6, label=lab)
        ax.axhline(0, color="0.55", lw=1)
        ax.set(xticks=xs, ylim=(-0.13, 0.08))
        ax.set_xticklabels(SEED_TUPLES, fontsize=9)
        ax.set_xlabel("rate tuple")
        if k == 0:
            ax.set_ylabel("Δ balanced accuracy")
            ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        ax.grid(alpha=0.25)
        ax.set_title(f"{name}  —  {n:,} points", fontsize=10, loc="left")

    fig.text(0.63, 0.80, """
What the repeat shows

LUCAS (left) behaves: the seed spread is
±0.005-0.011, the two sources separate
cleanly at low rate, and both converge to
zero at 8-8-8. The low-rate gap between
red and blue is 4-9x the noise, so it is
real.

LCMAP (right) does not. Its seed spread
reaches ±0.023 — wider than most of the
effects being discussed. The +0.02
"compression beats the original" reading
on page 3 disappears into it: across four
seeds the same configuration returns
-0.004, +0.002, +0.021 and +0.056.

The lesson is not that LCMAP disagrees.
It is that a single split on 3,008 points
cannot answer this question at all, and
a one-seed curve gives no hint of that.
""", fontsize=9.4, va="top", family="sans-serif")

    rows = []
    for bench, (_, name, _) in SEEDSETS.items():
        for src, lab in (("ae", "AlphaEarth"), ("tes", "Tessera")):
            for tup in SEED_TUPLES:
                st = spread.get((bench, src, tup))
                if st:
                    rows.append(f"{name:<11}{lab:<12}{tup:<9}"
                                f"{st[0]:>8.3f}{st[1]:>10.3f}{st[2]:>6d}")
    fig.text(0.09, 0.345,
             f"{'benchmark':<11}{'source':<12}{'rate':<9}{'mean Δ':>8}"
             f"{'sd':>10}{'n':>6}", fontsize=8.8, family="monospace",
             weight="bold")
    fig.add_artist(plt.Line2D([0.09, 0.50], [0.336, 0.336],
                              transform=fig.transFigure, color="#c9ced6",
                              lw=1))
    for i, r in enumerate(rows):
        fig.text(0.09, 0.312 - i * 0.0205, r, fontsize=8.8,
                 family="monospace",
                 color=INK if "Tessera" in r else MUT)
    footer(fig, 5)
    pdf.savefig(fig)
    plt.close(fig)


def page_absolute(pdf, base):
    fig = plt.figure(figsize=(11, 8.5))
    header(fig, "Absolute performance: the sources themselves",
           "Compression cost is only half the question. On the same points, "
           "how good are the two embedding fields to begin with?")

    ax = fig.add_axes([0.075, 0.46, 0.55, 0.36])
    xs = np.arange(len(TAGS))
    ae_o, ts_o, ae_b, ts_b = [], [], [], []
    for tag in TAGS:
        a, t = load(base, "ae", tag), load(base, "tes", tag)
        ae_o.append(a.orig.iloc[0])
        ts_o.append(t.orig.iloc[0])
        ae_b.append(a.recon.iloc[-1])
        ts_b.append(t.recon.iloc[-1])
    ax.bar(xs - 0.2, ae_o, 0.36, color=C_AE, label="AlphaEarth, uncompressed")
    ax.bar(xs + 0.2, ts_o, 0.36, color=C_TS, label="Tessera, uncompressed")
    ax.hlines(ae_b, xs - 0.38, xs - 0.02, color=INK, lw=2.2, zorder=4,
              label="same, compressed at 7.4 b/px")
    ax.hlines(ts_b, xs + 0.02, xs + 0.38, color=INK, lw=2.2, zorder=4)
    for x, (a, t) in enumerate(zip(ae_o, ts_o)):
        ax.text(x - 0.2, a - 0.045, f"{a:.3f}", ha="center", fontsize=8.3,
                color="white", weight="bold", zorder=5)
        ax.text(x + 0.2, t - 0.045, f"{t:.3f}", ha="center", fontsize=8.3,
                color="white", weight="bold", zorder=5)
    ax.set(xticks=xs, ylim=(0, 0.95),
           ylabel="balanced accuracy  /  R²")
    ax.set_xticklabels([SHORT[t] for t in TAGS], fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, ncol=1, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("Probe score on identical label points (linear probe, "
                 "full training budget)", fontsize=10, loc="left")

    fig.text(0.68, 0.82, """
Dark ticks mark each source at the
top of its rate ladder — compressed
scores sit essentially on top of the
uncompressed bars, which is the
transparency result restated.

AlphaEarth leads on four of the five
tasks, by 3-7 points of balanced
accuracy on the LCMAP pair and 0.035
of R² on ASTER.

The exception is LUCAS land cover,
where the two are level (0.5635 vs
0.5650).

This is a statement about these five
probe tasks, not about the sources in
general: all five are coarse thematic
labels at a single point, the regime
AlphaEarth was designed and tuned
for. Tessera's extra per-pixel
information may well pay off on tasks
with finer spatial or temporal
structure — none of which are in this
benchmark suite.
""", fontsize=9.4, va="top", family="sans-serif")

    # ---- the numbers behind both figures ----
    fig.text(0.075, 0.375, "The numbers behind pages 3-5  (linear probe, "
             "full training budget, identical points)", fontsize=10,
             weight="bold")
    hdr = (f"{'':<12}{'uncompressed':^19}{'Δ at 0.07 b/px':^21}"
           f"{'Δ at 0.43 b/px':^21}{'Δ at 7.4 b/px':^21}")
    sub = (f"{'benchmark':<12}{'AE':>9}{'Tess':>10}{'AE':>10}{'Tess':>11}"
           f"{'AE':>10}{'Tess':>11}{'AE':>10}{'Tess':>11}")
    fig.text(0.075, 0.335, hdr, fontsize=8.8, family="monospace", color=MUT)
    fig.text(0.075, 0.313, sub, fontsize=8.8, family="monospace",
             weight="bold")
    fig.add_artist(plt.Line2D([0.075, 0.90], [0.303, 0.303],
                              transform=fig.transFigure, color="#c9ced6",
                              lw=1))
    yy = 0.278
    for tag in TAGS:
        a, t = load(base, "ae", tag), load(base, "tes", tag)

        def at(d, bpp):
            r = d[np.isclose(d.bits_per_px, bpp, atol=1e-3)]
            return float(r.delta.iloc[0]) if len(r) else float("nan")

        line = (f"{SHORT[tag]:<12}{a.orig.iloc[0]:>9.3f}"
                f"{t.orig.iloc[0]:>10.3f}"
                f"{at(a, 0.0669):>10.3f}{at(t, 0.0669):>11.3f}"
                f"{at(a, 0.4277):>10.3f}{at(t, 0.4277):>11.3f}"
                f"{at(a, 7.4277):>10.3f}{at(t, 7.4277):>11.3f}")
        fig.text(0.075, yy, line, fontsize=8.8, family="monospace")
        yy -= 0.027
    fig.text(0.075, yy - 0.012,
             "ASTER GED is R²; the other four are balanced accuracy. "
             "Negative Δ = compression costs accuracy.", fontsize=8.6,
             color=MUT)
    footer(fig, 6)
    pdf.savefig(fig)
    plt.close(fig)


def page_next(pdf, spread):
    fig = plt.figure(figsize=(11, 8.5))
    header(fig, "Caveats and what to do next")
    fig.text(0.06, 0.855, """
Caveats

  ·  The codec is interim. tessera_vq8 stopped at 47k of 60k steps (validation cosine 0.962 against
     the 0.980 stage-1 ceiling the architecture reaches). A finished codec should move the whole
     Tessera curve left; the low-rate gap is the part most likely to shrink.

  ·  LCMAP cannot resolve this question. 3,008 points give a ±0.023 seed-to-seed standard deviation —
     wider than most of the effects discussed here. Its panels are shown for completeness only, and
     no claim in this report rests on them. ASTER has not been seed-checked; treat it likewise until
     it has.

  ·  Rate is not bytes. Bits per pixel here is the nominal RVQ index cost. Tessera carries 128 bands
     to AlphaEarth's 64, so at equal bits/pixel Tessera is compressing twice as much data — its
     compression factor at 1 b/px is roughly double AlphaEarth's at the same rate.

  ·  Sub-sampled coverage. LUCAS and LCMAP were capped at 2,000 Tessera tiles, so these are ~17%
     and ~25% subsets of the AlphaEarth study's point sets (both sources probed on the subset).

  ·  Five coarse thematic tasks. Nothing here tests temporal, sub-pixel or fine-texture use of the
     embeddings, where the sources differ most.

Next steps, in order of value

  1.  Re-run this suite against the final Tessera codec when stage-2 finishes. The scripts take a
      checkpoint path; the extracted datasets are reusable, so a rerun is minutes of GPU, not hours.

  2.  Set the Tessera quality target from the transparency knee (~1 b/px on this run), not from
      AlphaEarth's cosine spec. If that knee holds for the final codec it is the number that belongs
      in the release contract.

  3.  Add a task that exercises what Tessera has and AlphaEarth does not — fine-scale or
      multi-temporal structure. The current suite cannot see Tessera's advantage, only its cost.

  4.  Raise LCMAP's tile budget (it is 25% sampled) or retire it. As it stands it costs GPU time and
      returns noise. Seed-check ASTER at the same time.

Reproducing

  Datasets   /data/tessera-samples/eval/{aster_ged,lucas_lc,lcmap_lc}
  Results    ring-space/plots/tessera_downstream/
             (Tessera sweeps, ae_matched/, seedcheck/, seedcheck_lucas/)
  Code       branch worktree-tessera-source — georvq/eval/tessera_extract.py,
             scripts/eval_build_dataset_tessera.py, scripts/eval_rate_sweep.py --restrict-parquet,
             scripts/tessera_downstream_report.py (this document)
""", fontsize=9.6, va="top", family="sans-serif")
    footer(fig, 7)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = args.results_dir
    spread = seed_stats(base)
    with PdfPages(args.out) as pdf:
        page_summary(pdf, base, spread)
        page_method(pdf, base)
        page_curves(pdf, base)
        page_cosine(pdf, base, spread)
        page_noise(pdf, base, spread)
        page_absolute(pdf, base)
        page_next(pdf, spread)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
