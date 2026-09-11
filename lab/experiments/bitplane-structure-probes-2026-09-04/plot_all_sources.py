"""Combined bit-plane structure figure across all four sources.
Values transcribed from the certified probe runs of 2026-09-04."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/data/ERA5-samples/analysis_2026-09-04/plots"


def curve(ax, H, n, label, color, ls="-"):
    b = np.arange(1, len(H) + 1)
    ax.plot(b, 1 - np.array(H), "o", ls=ls, ms=3.5, lw=1.3,
            color=color, label=label)


fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

ax = axes[0, 0]   # era5 T500, 16-bit codes, with codec rungs
curve(ax, [0.90, 0.95, 0.94, 1, 1, 1, 1, 1], 16, "marginal", "#999")
curve(ax, [0.15, 0.32, 0.56, 0.93, 0.99, 1, 1, 1], 16,
      "column36 (2.0b)", "#66c2a5")
curve(ax, [0.04, 0.09, 0.20, 0.65, 0.97, 1, 1, 1], 16,
      "left px (3.0b)", "#fc8d62")
curve(ax, [0.05, 0.08, 0.15, 0.33, 0.82, 0.99, 1, 1], 16,
      "3x3x3 (3.5b)", "#8da0cb")
curve(ax, [0.10, 0.23, 0.50, 0.85, 0.99, 1, 1, 1], 16,
      "codec 20-0-0 (2.3b)", "tab:red", ls="--")
curve(ax, [0.03, 0.07, 0.19, 0.44, 0.90, 1, 1, 1], 16,
      "codec full (3.3b)", "darkred", ls="--")
ax.set_title("era5 T500 (16-bit store code)")

ax = axes[0, 1]   # AE dim curves, 8-bit
curve(ax, [0.90, 0.81, 0.95, 0.96, 0.99, 1.01, 1.01, 1.0], 8,
      "marginal", "#999")
curve(ax, [0.12, 0.12, 0.25, 0.49, 0.72, 0.97, 1.03, 1.0], 8,
      "left px (3.3b)", "#fc8d62")
curve(ax, [0.02, 0.03, 0.06, 0.18, 0.45, 0.85, 0.99, 1.01], 8,
      "3x3 ring (4.4b)", "#8da0cb")
curve(ax, [0.62, 0.49, 0.55, 0.82, 1.07, 1.17, 1.02, 1.0], 8,
      "other 63 dims + unit-norm (1.3b)", "#66c2a5")
ax.set_title("AlphaEarth dim (8-bit code; unit-norm buys ~1.3b, "
             "max-var dim)")

ax = axes[1, 0]   # tessera, 8-bit
curve(ax, [0.97, 1.07, 0.97, 1.08, 1.01, 1.0, 1.01, 1.01], 8,
      "marginal", "#999")
curve(ax, [0.16, 0.29, 0.57, 0.98, 1.0, 1.0, 1.0, 1.0], 8,
      "3x3 ring, median dim (2.0b)", "#8da0cb")
curve(ax, [0.15, 0.28, 0.78, 0.86, 0.87, 0.90, 0.88, 0.88], 8,
      "3x3 ring, max-var dim (2.4b)", "#7570b3")
curve(ax, [0.64, 0.79, 0.91, 1.04, 1.0, 1.0, 1.0, 1.0], 8,
      "other 127 dims (0.6b)", "#66c2a5")
ax.set_title("tessera dims (8-bit code; heavy tail never goes white)")

ax = axes[1, 1]   # S2, 16-bit DN
curve(ax, [0.14, 0.12, 0.98, 1.48, 0.82, 1.01, 1.11, 1.0, 1.0, 1.0], 16,
      "B03 marginal", "#999")
curve(ax, [0.0, 0.0, 0.08, 0.07, 0.33, 0.69, 2.22, 1.21, 1.12, 1.04], 16,
      "B03 left px (3.1b)", "#fc8d62")
curve(ax, [0.54, 0.45, 0.56, 0.52, 0.63, 1.16, 0.83, 0.84, 0.89, 0.87],
      16, "B10 marginal", "#bbb")
curve(ax, [0.0, 0.0, 0.0, 0.0, 0.04, 0.28, 0.59, 0.52, 0.65, 0.65], 16,
      "B10 left px (7.5b)", "#d95f02")
ax.set_title("Sentinel-2 L1C DN (nonstationary: only local priors "
             "certify)")

for ax in axes.flat:
    ax.set_xlabel("bit plane (MSB → LSB)")
    ax.set_ylabel("predictability 1−H")
    ax.set_ylim(-0.55, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
fig.suptitle("Bit-plane structure across sources: certified predictable"
             " bits per prior (2026-09-04)", fontsize=12)
fig.savefig(f"{OUT}/bitplane_all_sources.png", dpi=150)
print("wrote bitplane_all_sources.png")
