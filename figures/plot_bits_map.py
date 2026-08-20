"""Mollweide map of adaptive bits/window (5 m MAE bound) over the corpus."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SP = "/tmp/claude-1001/-home-ali-projects-georvq/24bbcb31-2bfb-40cb-90ef-13a2ea7e5780/scratchpad"
BOUND = 5.0

z = np.load(f"{SP}/error_bounded_dump.npz")
mae, rb, lat, lon = z["mae"], z["rung_bits"], z["lat"], z["lon"]
ok = mae <= BOUND
cost = np.where(ok, rb, np.inf)
pick = cost.argmin(1)
pick[~ok.any(1)] = rb.shape[1] - 1
wb = rb[np.arange(len(pick)), pick]

# aggregate the 9 windows per chip to one point
_, inv = np.unique(np.stack([lat, lon], 1), axis=0, return_inverse=True)
n = np.bincount(inv)
clat = np.bincount(inv, lat) / n
clon = np.bincount(inv, lon) / n
cbits = np.bincount(inv, wb) / n

fig = plt.figure(figsize=(11, 6), facecolor="white")
ax = fig.add_subplot(111, projection="mollweide", facecolor="white")

for feat in json.load(open(f"{SP}/coastline.geojson"))["features"]:
    g = feat["geometry"]
    lines = [g["coordinates"]] if g["type"] == "LineString" \
        else g["coordinates"]
    for line in lines:
        a = np.radians(np.asarray(line))
        brk = np.where(np.abs(np.diff(a[:, 0])) > np.pi * 0.9)[0]
        for seg in np.split(a, brk + 1):
            if len(seg) > 1:
                ax.plot(seg[:, 0], seg[:, 1], color="#c8c8c8", lw=0.6,
                        zorder=1)

order = np.argsort(cbits)          # draw high-bit points on top
sc = ax.scatter(np.radians(clon[order]), np.radians(clat[order]),
                c=cbits[order], s=2.5, cmap="Blues",
                norm=matplotlib.colors.LogNorm(vmin=max(cbits.min(), 30),
                                               vmax=cbits.max()),
                linewidths=0, zorder=2)

ax.grid(True, color="#eeeeee", lw=0.4)
ax.tick_params(labelsize=7, colors="#888888")
cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.04)
cb.set_label("bits per 128 px window (log scale)", fontsize=9,
             color="#444444")
cb.ax.tick_params(labelsize=8, colors="#444444")
ax.set_title(f"COPDEM archive: adaptive rate at {BOUND:g} m MAE bound\n"
             f"(7,000-chip sample; mean of 9 windows per chip)",
             fontsize=11, color="#222222")
fig.savefig(f"{SP}/bits_map.png", dpi=170, bbox_inches="tight")
print(f"saved {SP}/bits_map.png; bits range "
      f"{cbits.min():.0f}-{cbits.max():.0f}, median {np.median(cbits):.0f}")
