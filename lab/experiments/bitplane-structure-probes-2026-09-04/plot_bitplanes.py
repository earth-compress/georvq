"""Figures for the T500 bit-plane structure probes (values transcribed
from the probe runs of 2026-09-04; val split, 16-bit store codes)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/data/ERA5-samples/analysis_2026-09-04/plots"
os.makedirs(OUT, exist_ok=True)

BITS = np.arange(1, 17)
# per-plane conditional entropy H_k (bits), MSB -> LSB
H = {
    "marginal": [0.90, 0.95, 0.94, 1.00, 1.00, 1.00, 1.00, 1.00,
                 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "PCA-8 of column": [0.31, 0.61, 0.88, 0.98, 0.99, 1.00, 1.00, 1.00,
                        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "column (36 lvls)": [0.15, 0.32, 0.56, 0.93, 0.99, 1.00, 1.00, 1.00,
                         1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "left pixel": [0.04, 0.09, 0.20, 0.65, 0.97, 1.00, 1.00, 1.00,
                   1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "3x3x3 cube (26)": [0.05, 0.08, 0.15, 0.33, 0.82, 0.99, 1.00, 1.00,
                        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}
STRUCT = {k: 16 - sum(v) for k, v in H.items()}
MED_LSB = 0.000355                       # K per LSB, median cube

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                              constrained_layout=True,
                              gridspec_kw={"width_ratios": [2, 1]})
colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(H)))
for c, (name, h) in zip(colors, H.items()):
    ax.plot(BITS, 1 - np.array(h), "o-", ms=4, lw=1.4, color=c,
            label=f"{name}  ({STRUCT[name]:.2f} bits)")
ax.fill_between(BITS, 0, 1 - np.array(H["3x3x3 cube (26)"]),
                alpha=0.08, color=colors[-1])
ax.set_xlabel("bit plane (MSB → LSB)")
ax.set_ylabel("predictability  1 − H(bit | prior, MSBs)")
ax.set_xticks(BITS[::2])
sec = ax.secondary_xaxis("top")
sec.set_xticks(BITS[::2])
sec.set_xticklabels([f"{MED_LSB * 2 ** (16 - b):.2g}" for b in BITS[::2]])
sec.set_xlabel("bit step size (K, median cube)")
ax.axvspan(5.5, 16.5, color="gray", alpha=0.06)
ax.text(10.5, 0.5, "white to all priors\n(≈ reanalysis noise)",
        ha="center", fontsize=9, color="gray")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="center right")
ax.set_title("T500: bit-plane structure by prior (val, 16-bit store code)")

names = list(STRUCT)
ax2.barh(range(len(names)), [STRUCT[n] for n in names],
         color=colors, height=0.6)
ax2.set_yticks(range(len(names)), names, fontsize=8)
ax2.axvline(4.63, color="k", lw=1, ls="--")
ax2.text(4.63, len(names) - 0.4, " codec @0.23 K\n (Gaussian-equiv.)",
         fontsize=7.5, va="top")
ax2.set_xlabel("certified structure (bits of 16)")
ax2.grid(axis="x", alpha=0.25)
fig.savefig(f"{OUT}/bitplane_priors_t500.png", dpi=160)
print(f"wrote {OUT}/bitplane_priors_t500.png")
