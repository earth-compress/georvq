"""T500 bit-plane predictability: analytical priors (2026-09-04 runs)
plus the codec priors from bitplane_codec_results.npz -- the 37-channel
column model (scstd 60k) and the single-level mono model (20k), each at
three saturation rungs. Rate labels are index+stats bits over the
number of pixels the payload serves (37 levels vs 1)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/data/ERA5-samples/analysis_2026-09-04/plots"
os.makedirs(OUT, exist_ok=True)
BITS = np.arange(1, 17)

ANALYTIC = {
    "marginal": [0.90, 0.95, 0.94, 1.00, 1.00, 1.00, 1.00, 1.00,
                 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "column (36 lvls)": [0.15, 0.32, 0.56, 0.93, 0.99, 1.00, 1.00, 1.00,
                         1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "left pixel": [0.04, 0.09, 0.20, 0.65, 0.97, 1.00, 1.00, 1.00,
                   1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "3x3x3 cube (26)": [0.05, 0.08, 0.15, 0.33, 0.82, 0.99, 1.00, 1.00,
                        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}
R = np.load("/data/ERA5-samples/analysis_2026-09-04/"
            "bitplane_codec_results.npz")
IDX = {"20-0-0": 180, "20-20-0": 3380, "20-20-20": 59700}
CODEC = {}
for fam, side, px, tag in (("col", 444, 4096 * 37, "column 37ch"),
                           ("mono", 12, 4096, "mono 1ch")):
    for rung, ib in IDX.items():
        k = f"{fam}-{rung}"
        bpp = (ib + side) / px
        rmse = float(R[k + "_rmse"][0])
        CODEC[f"{tag} {rung} ({bpp:.2f} b/px, {rmse:.2f} K)"] = (
            np.array(R[k]), fam)

MED_LSB = 0.000355
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8),
                              constrained_layout=True,
                              gridspec_kw={"width_ratios": [2, 1.2]})
greys = plt.cm.Greys(np.linspace(0.35, 0.75, len(ANALYTIC)))
for c, (name, h) in zip(greys, ANALYTIC.items()):
    ax.plot(BITS, 1 - np.array(h), "o-", ms=3, lw=1.0, color=c,
            label=f"{name}  ({16 - sum(h):.2f} b)")
fam_cm = {"col": plt.cm.Blues, "mono": plt.cm.Oranges}
fam_i = {"col": 0, "mono": 0}
for name, (h, fam) in CODEC.items():
    fam_i[fam] += 1
    c = fam_cm[fam](0.35 + 0.22 * fam_i[fam])
    ax.plot(BITS, 1 - h, "s-", ms=4, lw=1.6, color=c,
            label=f"{name}  ({16 - h.sum():.2f} b)")
ax.set_xlabel("bit plane (MSB -> LSB)")
ax.set_ylabel("predictability  1 - H(bit | prior, MSBs)")
ax.set_xticks(BITS[::2])
sec = ax.secondary_xaxis("top")
sec.set_xticks(BITS[::2])
sec.set_xticklabels([f"{MED_LSB * 2 ** (16 - b):.2g}" for b in BITS[::2]])
sec.set_xlabel("bit step size (K, median cube)")
ax.grid(alpha=0.25)
ax.legend(fontsize=7, loc="upper right")
ax.set_title("T500 bit-plane structure: codec priors vs analytical "
             "(val, 16-bit store code)")

names = (list(ANALYTIC)
         + [n for n in CODEC])
vals = ([16 - sum(v) for v in ANALYTIC.values()]
        + [16 - h.sum() for h, _ in CODEC.values()])
cols = (list(greys)
        + [fam_cm[f](0.35 + 0.22 * (i % 3 + 1))
           for i, (_, f) in enumerate(CODEC.values())])
ax2.barh(range(len(names)), vals, color=cols, height=0.6)
ax2.set_yticks(range(len(names)), names, fontsize=7)
ax2.invert_yaxis()
ax2.set_xlabel("certified structure (bits of 16)")
ax2.grid(alpha=0.25, axis="x")
fig.savefig(f"{OUT}/bitplane_codec_mono_vs_col.png", dpi=130)
print("wrote", f"{OUT}/bitplane_codec_mono_vs_col.png")
