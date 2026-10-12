"""Figure for the low-pass Fourier bit-plane sweep (T500, val)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/data/ERA5-samples/analysis_2026-09-04/plots"
os.makedirs(OUT, exist_ok=True)
BITS = np.arange(1, 17)
MED_LSB = 0.000355

H8 = {   # first 8 planes; rest are 1.00
    1: [0.45, 0.76, 0.94, 0.98, 0.99, 1.00, 1.00, 1.00],
    2: [0.31, 0.62, 0.88, 0.98, 0.99, 1.00, 1.00, 1.00],
    4: [0.22, 0.47, 0.72, 0.96, 0.99, 1.00, 1.00, 1.00],
    8: [0.13, 0.31, 0.53, 0.91, 0.99, 1.00, 1.00, 1.00],
    16: [0.08, 0.21, 0.38, 0.84, 0.99, 1.00, 1.00, 1.00],
    32: [0.06, 0.16, 0.33, 0.80, 0.99, 1.00, 1.00, 1.00],
}
STR = {1: 0.89, 2: 1.22, 4: 1.64, 8: 2.12, 16: 2.48, 32: 2.63}
NCOEF = {1: 4, 2: 9, 4: 29, 8: 107, 16: 415, 32: 1637}
REFS = {"left pixel": 3.02, "3x3x3 cube": 3.54, "column36": 2.04}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                              constrained_layout=True)
colors = plt.cm.plasma(np.linspace(0.05, 0.8, len(H8)))
for c, (r, h) in zip(colors, H8.items()):
    hh = np.concatenate([h, np.ones(8)])
    ax.plot(BITS, 1 - hh, "o-", ms=3.5, lw=1.3, color=c,
            label=f"r ≤ {r} ({NCOEF[r]} coefs, {STR[r]:.2f} bits)")
ax.set_xlabel("bit plane (MSB → LSB)")
ax.set_ylabel("predictability  1 − H(bit | low-pass, MSBs)")
ax.set_xticks(BITS[::2])
sec = ax.secondary_xaxis("top")
sec.set_xticks(BITS[::2])
sec.set_xticklabels([f"{MED_LSB * 2 ** (16 - b):.2g}" for b in BITS[::2]])
sec.set_xlabel("bit step size (K, median cube)")
ax.grid(alpha=0.25)
ax.legend(fontsize=7.5)
ax.set_title("T500: bit-plane structure vs low-pass cutoff")

rr = np.array(list(STR))
ax2.semilogx(rr, [STR[r] for r in rr], "o-", lw=1.5, color="tab:red",
             label="Fourier low-pass")
for c, (name, v) in zip(["tab:blue", "tab:green", "tab:gray"], REFS.items()):
    ax2.axhline(v, color=c, lw=1, ls="--")
    ax2.text(1.05, v + 0.04, name, fontsize=7.5, color=c)
ax2.set_xlabel("cutoff radius r (cycles / 128 px window)")
ax2.set_ylabel("certified structure (bits of 16)")
ax2.set_xticks(rr, [str(r) for r in rr])
ax2.grid(alpha=0.25, which="both")
ax2.set_title("structure vs frequency content")
fig.savefig(f"{OUT}/bitplane_fourier_t500.png", dpi=160)
print(f"wrote {OUT}/bitplane_fourier_t500.png")
