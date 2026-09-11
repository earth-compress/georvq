"""Three-source marginal-purchase comparison figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = "/data/ERA5-samples/analysis_2026-09-04"
era = np.load(f"{A}/rung_purchase.npy")          # bpv, struct, sigma, ratio
tes = np.load(f"{A}/rungs_tessera.npy")          # bpv, sigma, dh, ratio
ae = np.load(f"{A}/rungs_ae.npy")

fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
curves = [("era5-t (scstd 60k)", era[:, 0], era[:, 3], "tab:red"),
          ("tessera (release)", tes[:, 0], tes[:, 3], "tab:green"),
          ("AlphaEarth (release)", ae[:, 0], ae[:, 3], "tab:blue")]
for name, bpv, ratio, c in curves:
    good = np.isfinite(ratio) & (ratio > 0)
    ax.plot(bpv[good], ratio[good], "o-", lw=1.6, ms=4.5, color=c,
            label=name)
    bad = np.isfinite(ratio) & (ratio <= 0)
    if bad.any():
        ax.plot(bpv[bad], np.abs(ratio[bad]), "x", ms=11, color=c)
        for b, r in zip(bpv[bad], ratio[bad]):
            ax.annotate(f"era5 20-20-1\nNEGATIVE ({r:.1f})", (b, abs(r)),
                        fontsize=7.5, color=c,
                        xytext=(8, -2), textcoords="offset points")
ax.axhline(1.0, color="k", lw=1.1, ls="--")
ax.text(0.0011, 1.18, "break-even: one bit buys one bit", fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("cumulative transmitted bits / value (incl. side channels)")
ax.set_ylabel("marginal purchase  (σ-halvings per bit ×16-bit equiv.)")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=9)
ax.set_title("How many bits each bit buys — release codecs, "
             "fill-order rungs")
fig.savefig(f"{A}/plots/rung_purchase_all_sources.png", dpi=160)
print("wrote rung_purchase_all_sources.png")
