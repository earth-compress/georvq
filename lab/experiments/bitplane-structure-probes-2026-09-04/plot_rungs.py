"""Figure: marginal purchase along the fill-order pareto (T500)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = np.load("/data/ERA5-samples/analysis_2026-09-04/rung_purchase.npy")
names = ["stats", "15-0-0", "17-0-0", "20-0-0", "20-2-0", "20-5-0",
         "20-10-0", "20-15-0", "20-20-0", "20-20-1", "20-20-2",
         "20-20-3", "20-20-5", "20-20-8", "20-20-12", "20-20-16",
         "20-20-20"]
bpv, struct, sig, ratio = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
seg = ([0] * 1 + [1] * 3 + [2] * 5 + [3] * 8)     # stats, t0, t1, t2
segc = ["#666666", "#1b9e77", "#d95f02", "#7570b3"]
lab = ["stats channel", "tier 0 (coarse)", "tier 1 (mid)", "tier 2 (fine)"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                              constrained_layout=True)
for s in range(4):
    m = [i for i in range(len(bpv)) if seg[i] == s]
    mm = ([m[0] - 1] + m) if m[0] > 0 else m
    ax.plot(bpv[mm], struct[mm], "o-", color=segc[s], lw=1.6, ms=4,
            label=lab[s])
ax.set_xscale("log")
ax.set_xlabel("transmitted bits / value (cumulative, incl. stats)")
ax.set_ylabel("T500 structure delivered (bits of 16)")
ax.grid(alpha=0.25, which="both")
for i in (1, 8, 9, 16):
    ax.annotate(names[i], (bpv[i], struct[i]), fontsize=7,
                xytext=(4, -9), textcoords="offset points")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("structure vs rate along the fill path")

good = ratio > 0
ax2.plot(bpv[good], ratio[good], "o-", lw=1.4, ms=4, color="tab:blue")
bad = ~good
ax2.plot(bpv[bad], np.abs(ratio[bad]), "x", ms=9, color="tab:red")
for i in np.where(bad)[0]:
    ax2.annotate(f"{names[i]}  (NEGATIVE: {ratio[i]:.1f})",
                 (bpv[i], abs(ratio[i])), fontsize=7.5, color="tab:red",
                 xytext=(6, 4), textcoords="offset points")
ax2.axhline(1.0, color="k", lw=1, ls="--")
ax2.text(0.0035, 1.15, "break-even: 1 bit buys 1 bit", fontsize=8)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("transmitted bits / value (cumulative)")
ax2.set_ylabel("marginal purchase (Δ structure / Δ bits)")
ax2.grid(alpha=0.25, which="both")
ax2.set_title("how many bits each bit buys, rung by rung")
fig.savefig("/data/ERA5-samples/analysis_2026-09-04/plots/"
            "rung_purchase_t500.png", dpi=160)
print("wrote rung_purchase_t500.png")
