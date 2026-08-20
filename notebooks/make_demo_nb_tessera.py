"""Generate notebooks/demo_tessera_codec.ipynb (run once; the notebook is
the artifact, this keeps it reproducible)."""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None,
                  "source": src.splitlines(keepends=True)})


md("""# Tessera neural codec — demo

A learned pyramidal RVQ codec for [Tessera](https://github.com/ucam-eo/geotessera)
embedding tiles (128 bands, 10 m grid). One encode serves the whole
rate–distortion front: the packet is stored at full depth and *truncating
RVQ levels at decode time is the rate control*, so a 0.19 b/px preview and
a 1.12 b/px reconstruction come from the same bytes.

| the packet carries | |
|---|---|
| per-tier code indices | 3 tiers on a 64 px window: 1, 16 and 256 cells |
| the patch mean | 128 bands x 6 bits, charged at every rate |
| an optional refinement | fine-tier residual, off by default |

**Decoding needs no GPU and no extras** — this notebook runs on CPU in
about a minute. Full contract: `docs/TESSERA_RELEASE.md`.""")

code("""import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "..")                    # georvq (run from notebooks/)
from georvq.tessera import (load, encode_tile, decode_tile, prep,
                            requantize, to_windows, from_windows)
from georvq.metrics import centered_cos

BUNDLE = os.environ.get("GEORVQ_TESSERA_BUNDLE",
                        "/data/georvq-releases/tessera-beta-v1/bundle")
DEVICE = "cpu"          # a GPU is not needed; set "cuda:0" if you have one
YEAR, LON, LAT = 2024, 40.65, 56.55         # a registry tile (Vladimir, RU)
CROP = 512                                  # px; 8x8 = 64 codec windows

R = load(BUNDLE, device=DEVICE)
print("bitstream_id :", R.bundle["bitstream_id"])
print("tiers        :", R.bundle["tiers"])
print("mean channel :", R.bundle["mean_channel"])
print("refinement   :", R.bundle["refinement"],
      "| on by default:", R.bundle["refine_by_default"])""")

md("""## 1. A tile, straight from `geotessera`

No chip store, no preprocessing of ours: `geotessera` hands back int8
codes and a per-pixel scale plane, and that is exactly what the codec
takes. (`georvq.tessera.prep` accepts either `(H, W, bands)` as returned
here or `(bands, H, W)`; both land on identical tensors.)""")

code("""import geotessera
gt = geotessera.GeoTessera()

t0 = time.time()
q = np.load(gt.registry.fetch(year=YEAR, lon=LON, lat=LAT))
s = np.load(gt.registry.fetch(year=YEAR, lon=LON, lat=LAT, is_scales=True))
print(f"fetched in {time.time()-t0:.0f}s")
print("embedding", q.shape, q.dtype, " scales", s.shape, s.dtype)

q, s = q[:CROP, :CROP], s[:CROP, :CROP]     # one crop, to keep the demo quick
x, valid = prep(q, s, device=DEVICE)        # (1, 128, CROP, CROP) model space
print("model space", tuple(x.shape), "| valid", f"{valid.float().mean():.1%}")""")

md("""## 2. What 128 bands look like

There is no natural RGB for an embedding field, so everything below is
projected through the *same* PCA basis — fitted on the original — which
keeps originals and reconstructions comparable to each other.""")

code("""def pca_basis(t):
    v = t[0].reshape(t.shape[1], -1).T.float()
    v = v - v.mean(0)
    return torch.linalg.svd(v[::7], full_matrices=False)[2][:3].T

def to_rgb(t, basis):
    v = t[0].reshape(t.shape[1], -1).T.float() @ basis
    lo, hi = torch.quantile(v, torch.tensor([0.02, 0.98]), dim=0)
    v = ((v - lo) / (hi - lo)).clamp(0, 1)
    return v.reshape(t.shape[2], t.shape[3], 3).numpy()

BASIS = pca_basis(x)
plt.figure(figsize=(5, 5)); plt.imshow(to_rgb(x, BASIS)); plt.axis("off")
plt.title(f"original — {CROP}x{CROP} px, 128 bands"); plt.show()""")

md("""## 3. Encode once

`encode_tile` cuts the tile into 64 px windows, encodes each at **full
depth**, and returns the packets plus the window grid. Whole windows
only: a tile that does not divide evenly is cropped rather than padded,
because padding invents data the encoder would then spend rate on.""")

code("""t0 = time.time()
packets, grid = encode_tile(R, q, s, batch=32)
print(f"encoded {len(packets)} windows in {time.time()-t0:.1f}s  (grid {grid})")

p = packets[0]
print("\\nper window:")
for i, idx in enumerate(p.indices):
    print(f"  tier {i}: indices {idx.shape}  {idx.dtype}")
print(f"  mean   : {p.mean.shape} {p.mean.dtype}  <- the side channel")
print(f"  refine : {p.refinements}")""")

md("""## 4. The rate ladder — one encode, every rate

`depths` truncates RVQ levels at decode time. The same `packets` object
serves every row below; nothing is re-encoded.

`ccos` is the centred cosine, and it is the honest one here: the patch
mean is transmitted, so plain cosine is inflated by a channel that is
already paid for.

The numbers below are for **this one tile**, not the release figures: `docs/TESSERA_RELEASE.md` reports the ladder over a held-out set, where full depth is ccos 0.807 rather than the ~0.93 an easy tile gives.""")

code("""LADDER = [(20, 0, 0), (20, 20, 0), (20, 20, 1), (20, 20, 3), (20, 20, 6),
          (20, 20, 20)]
LOSSLESS = 672.0        # b/px, measured by analysis/data_probe.py

recons, rows = {}, []
for d in LADDER:
    out = decode_tile(R, packets, grid, depths=list(d), batch=32)
    recons[d] = out
    cos = torch.nn.functional.cosine_similarity(
        out, x[:, :, :out.shape[2], :out.shape[3]], dim=1).mean()
    cc = centered_cos(out, x[:, :, :out.shape[2], :out.shape[3]],
                      valid[:, :out.shape[2], :out.shape[3]])
    rows.append((d, R.bpp(list(d)), float(cos), float(cc)))

print(f"{'tuple':>10} {'b/px':>8} {'vs lossless':>12} {'cos':>8} {'ccos':>8}")
for d, bpp, cos, cc in rows:
    print(f"{'-'.join(map(str,d)):>10} {bpp:8.3f} {LOSSLESS/bpp:11.0f}x "
          f"{cos:8.4f} {cc:8.4f}")""")

code("""show = [(20, 0, 0), (20, 20, 0), (20, 20, 1), (20, 20, 6)]
fig, ax = plt.subplots(1, len(show) + 1, figsize=(4 * (len(show) + 1), 4.2))
ax[0].imshow(to_rgb(x, BASIS)); ax[0].set_title("original"); ax[0].axis("off")
for a, d in zip(ax[1:], show):
    a.imshow(to_rgb(recons[d], BASIS)); a.axis("off")
    a.set_title(f"{'-'.join(map(str,d))}\\n{R.bpp(list(d)):.3f} b/px")
plt.tight_layout(); plt.show()""")

md("""## 5. The transmitted mean

Tessera vectors are not unit-norm, and the per-window mean is most of a
window's energy — cheap to send and worth far more per bit than extra
pyramid depth. So the codec subtracts it, codes what is left, and adds it
back, with both encoder and decoder conditioned on it.

It is **charged at every rate**, which matters most at the bottom of the
ladder where it is the majority of the bitstream. Quantizing it costs
almost nothing against sending it exactly.""")

code("""side = R.side_bits
print(f"mean channel: {side:.0f} bits/window = {side/64**2:.4f} b/px")
for d in [(20, 0, 0), (20, 20, 0), (20, 20, 20)]:
    tot = R.bpp(list(d))
    print(f"  at {'-'.join(map(str,d)):>9}: {side/64**2/tot:5.1%} of the bitstream")

# what the quantization of that channel costs, end to end
mu = R.model.patch_mean(x, valid)
print(f"\\nmean vector: {tuple(mu.shape)}, quantized to "
      f"{R.mean_channel.bits} bits/band against shipped per-band constants")
err = (R.mean_channel.round_trip(mu.cpu().numpy()) - mu.cpu().numpy())
print(f"round-trip error: max {np.abs(err).max():.5f}, "
      f"rms {np.sqrt((err**2).mean()):.5f}")""")

md("""## 6. The optional refinement

After all 20 levels a tier still has a residual. Coding it at ~1 bit per
dimension in a fixed random rotation buys part of the gap back to the
un-quantized ceiling, with **no extra trained codebooks** — the whole
"codebook" is a seed plus a Lloyd table fitted once.

It is **off by default here**, unlike the AlphaEarth release's coarse
sidecar, and the reason is worth knowing: *which* tier holds recoverable
slack is a property of the model, not of the architecture. On this model
the coarse tier returns nothing and the fine tier returns a lot — but the
fine tier has 256 cells, so it costs 128x more.""")

code("""full = [20, 20, 20]
pk_r, grid_r = encode_tile(R, q, s, batch=32, refine=True)
out_r = decode_tile(R, pk_r, grid_r, depths=full, batch=32)
xc = x[:, :, :out_r.shape[2], :out_r.shape[3]]
vc = valid[:, :out_r.shape[2], :out_r.shape[3]]

base_cc = float(centered_cos(recons[tuple(full)], xc, vc))
ref_cc = float(centered_cos(out_r, xc, vc))
# the model works on 64px windows, so the ceiling is measured there and
# stitched back, giving a tile-shaped result comparable to the decodes
xw, vw, gr = to_windows(x, valid, R.window)
ceil = from_windows(R.model.eval_ceiling(xw, valid=vw), gr, R.window)
ceil_cc = float(centered_cos(ceil[:, :, :xc.shape[2], :xc.shape[3]], xc, vc))

print(f"{'':>22} {'b/px':>8} {'ccos':>8}")
print(f"{'RVQ x20':>22} {R.bpp(full):8.3f} {base_cc:8.4f}")
print(f"{'+ refinement':>22} {R.bpp(full, refine=True):8.3f} {ref_cc:8.4f}")
print(f"{'un-quantized ceiling':>22} {'-':>8} {ceil_cc:8.4f}")
gap = (ref_cc - base_cc) / max(ceil_cc - base_cc, 1e-9)
print(f"\\ngap to the ceiling closed: {gap:.0%} "
      f"for +{R.bpp(full, refine=True)-R.bpp(full):.2f} b/px")""")

md("""## 7. Back to source units

`requantize` undoes the model's internal rescaling, and with the scale
plane it also returns int8 codes on Tessera's own quantization grid — so
a reconstruction can be written back in the layout it arrived in.""")

code("""orig_units = requantize(recons[(20, 20, 6)])
_, codes = requantize(recons[(20, 20, 6)], s[:orig_units.shape[2],
                                             :orig_units.shape[3]])
print("float, source units:", tuple(orig_units.shape), orig_units.dtype)
print("int8, source grid  :", tuple(codes.shape), codes.dtype)""")

md("""## Wrap-up

- **One encode, every rate.** Truncate levels at decode; nothing is
  re-encoded, and a deeper decode is a prefix read away.
- **Pick the rate from the task, not the cosine.** On the five downstream
  benchmarks we have (ASTER GED, LUCAS LC/LU, LCMAP LC/LU) this codec is
  *transparent* — indistinguishable from the uncompressed embedding —
  from about 1 b/px, i.e. around `20-20-1`. The upper ladder exists for
  tasks with finer spatial or temporal structure, which those
  point-thematic benchmarks cannot see. If you have such a task, we would
  very much like to hear how it behaves.
- **Cosine does not transfer across sources.** At matched reconstruction
  cosine this codec costs *less* downstream accuracy than the AlphaEarth
  one, because Tessera carries more independent per-pixel variation and
  much of it is weakly correlated with land cover. Do not port a cosine
  target from another embedding field.

Getting the weights, the full rate table and the format contract:
`docs/TESSERA_RELEASE.md`.""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

with open("demo_tessera_codec.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote demo_tessera_codec.ipynb ({len(CELLS)} cells)")
