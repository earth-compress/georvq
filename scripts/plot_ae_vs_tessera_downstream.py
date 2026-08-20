"""AE vs Tessera downstream-compression comparison figure.

Overlays, per benchmark, the downstream metric (and its delta to the
uncompressed embedding) as a function of bits/pixel, for the AE codecs
(prior study) and the Tessera codec — probes run over the same label
points (the Tessera eval datasets are built AT the AE eval datasets'
points; coverage differs only where Tessera tiles were unavailable or
were seen in training).

Absolute panels compare sources (each probe measured against its own
source's original embedding); the delta panels are the compression cost
and are the apples-to-apples codec comparison.

Usage:
  python scripts/plot_ae_vs_tessera_downstream.py \
      --ae-dir  ~/projects/ring-space/plots/ae_downstream \
      --tes-dir ~/projects/ring-space/plots/tessera_downstream \
      --out-dir ~/projects/ring-space/plots/tessera_downstream
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH_METRIC = {"aster_ged": "R$^2$ (emissivity)",
                "lucas_lc": "balanced acc (LUCAS LC)",
                "lcmap_lc": "balanced acc (LCMAP LC)",
                "lucas_lc_label_lu": "balanced acc (LUCAS LU)",
                "lcmap_lc_label_lu": "balanced acc (LCMAP LU)"}


def load(dirpath, tag):
    p = os.path.join(dirpath, f"ratesweep_{tag}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["budget"] = d.budget.astype(str)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-dir", required=True)
    ap.add_argument("--tes-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget", default="all")
    ap.add_argument("--probe", default="linear")
    args = ap.parse_args()

    tags = [t for t in BENCH_METRIC if load(args.tes_dir, t) is not None]
    ncol = len(tags)
    fig, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol, 8), squeeze=False)
    for k, tag in enumerate(tags):
        ax_abs, ax_d = axes[0][k], axes[1][k]
        for dirpath, color_cycle, style in (
                (args.ae_dir, ["#1f77b4", "#17becf"], "-"),
                (args.tes_dir, ["#d62728"], "-")):
            d = load(dirpath, tag)
            if d is None:
                continue
            d = d[(d.budget == args.budget) & (d.probe == args.probe)]
            for ci, mo in enumerate(dict.fromkeys(d.model)):
                s = d[d.model == mo].sort_values("bits_per_px")
                col = color_cycle[ci % len(color_cycle)]
                ax_abs.plot(s.bits_per_px, s.recon, style, marker="o",
                            ms=4, color=col, label=mo)
                ax_abs.axhline(s.orig.iloc[0], color=col, ls=":", lw=1,
                               alpha=0.6)
                ax_d.plot(s.bits_per_px, s.delta, style, marker="o",
                          ms=4, color=col, label=mo)
        ax_d.axhline(0, color="0.4", lw=1)
        for a in (ax_abs, ax_d):
            a.set_xscale("log")
            a.grid(alpha=0.3)
        ax_abs.set_title(BENCH_METRIC[tag], fontsize=10)
        ax_abs.set_ylabel("downstream metric" if k == 0 else "")
        ax_d.set_ylabel("Δ metric (recon − orig)" if k == 0 else "")
        ax_d.set_xlabel("bits / pixel")
        if k == 0:
            ax_abs.legend(fontsize=7)
            ax_d.legend(fontsize=7)
    fig.suptitle(f"Downstream performance under compression — AE vs Tessera "
                 f"(probe={args.probe}, budget={args.budget}; dotted = "
                 f"uncompressed baseline)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(args.out_dir,
                       f"ae_vs_tessera_downstream_{args.probe}_{args.budget}.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
