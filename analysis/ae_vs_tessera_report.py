"""AE vs Tessera comparison report (PDF for a technical, non-specialist
audience). Pages: data intro/imagery, texture zoom, power spectra,
latent effective rank, noise-floor R-D, takeaways."""
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from georvq.config import load_config
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.train.figures import pca_rgb_basis, _pca_rgb

DEV = torch.device("cuda:1")
OUT = "/data/tessera-samples/reports/ae_vs_tessera.pdf"
INK, MUT = "#3d4451", "#7a8494"
C_AE, C_TS = "#4269d0", "#e05c5c"          # AE blue, Tessera red
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9ced6",
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK,
                     "figure.facecolor": "white"})

# ---------------- data ----------------
tcfg = load_config("configs/tessera_stage1_4x.yaml")
acfg = load_config("configs/ae_stage1_patch4.yaml")
tstore = ChipStore(tcfg.data.store_dir)
astore = ChipStore(acfg.data.store_dir)
_, tval = tstore.split(tcfg.data.val_frac, tcfg.data.split_seed)
_, aval = astore.split(0.04, 0)

def grab_windows(store, rows, src_cfg, n=6, px=192):
    """n windows of px*px, prepped to model space, PCA-RGB rendered."""
    toks = fixed_val_tokens(store, rows, token=px, per_chip=1, max_chips=64,
                            max_tokens=64)
    cfg = load_config(src_cfg)
    from georvq.sources import get_source
    src = get_source(cfg.data.source, cfg.model)
    x, valid = src.prep(toks.to(DEV), DEV)
    # rank windows by spatial variance, take a spread
    var = x.reshape(x.shape[0], x.shape[1], -1).var(-1).mean(-1)
    order = var.argsort(descending=True)
    sel = order[torch.linspace(0, min(30, len(order) - 1), n).long()]
    x, valid = x[sel], valid[sel]
    basis = pca_rgb_basis(x, valid)
    return x, _pca_rgb(x, basis)

with torch.no_grad():
    tx, timgs = grab_windows(tstore, tval, "configs/tessera_stage1_4x.yaml")
    ax_, aimgs = grab_windows(astore, aval, "configs/ae_stage1_patch4.yaml")

pdf = PdfPages(OUT)

# ---------------- page 1: intro + imagery ----------------
fig = plt.figure(figsize=(11, 8.5))
fig.text(0.06, 0.95, "AlphaEarth vs Tessera: two embedding fields,"
         " two compression problems", fontsize=17, weight="bold")
fig.text(0.06, 0.905,
    "Both datasets are 10 m/px 'embedding fields': every pixel is a vector\n"
    "(AlphaEarth: 64 numbers, Tessera: 128) produced by a foundation model\n"
    "from a year of satellite observations. We compress them with the same\n"
    "codec family. The pages that follow show why the two sources behave\n"
    "very differently under compression.", fontsize=10.5, va="top")
for i in range(4):
    axp = fig.add_axes([0.06 + i * 0.225, 0.42, 0.205, 0.28])
    axp.imshow(aimgs[i], interpolation="nearest")
    axp.set_xticks([]); axp.set_yticks([])
    if i == 0:
        axp.set_ylabel("AlphaEarth", fontsize=11, weight="bold", color=C_AE)
for i in range(4):
    axp = fig.add_axes([0.06 + i * 0.225, 0.10, 0.205, 0.28])
    axp.imshow(timgs[i], interpolation="nearest")
    axp.set_xticks([]); axp.set_yticks([])
    if i == 0:
        axp.set_ylabel("Tessera", fontsize=11, weight="bold", color=C_TS)
fig.text(0.06, 0.055,
    "1.9 km windows, embeddings projected to RGB via PCA (colors arbitrary,"
    " structure real). Same projection idea for both rows.\n"
    "Visually: AlphaEarth fields are smooth and region-like;"
    " Tessera fields carry much more pixel-scale texture.", fontsize=9,
    color=MUT)
pdf.savefig(fig); plt.close(fig)

# ---------------- page 2: texture zoom ----------------
fig = plt.figure(figsize=(11, 8.5))
fig.text(0.06, 0.95, "Zooming in: where the difficulty lives",
         fontsize=16, weight="bold")
for j, (imgs, name, col) in enumerate([(aimgs, "AlphaEarth", C_AE),
                                       (timgs, "Tessera", C_TS)]):
    im = imgs[0]
    axp = fig.add_axes([0.06, 0.14 + (1 - j) * 0.38, 0.34, 0.33])
    axp.imshow(im, interpolation="nearest")
    axp.add_patch(Rectangle((100, 100), 48, 48, fill=False, ec="w", lw=1.5))
    axp.set_xticks([]); axp.set_yticks([])
    axp.set_ylabel(name, fontsize=11, weight="bold", color=col)
    axz = fig.add_axes([0.44, 0.14 + (1 - j) * 0.38, 0.33, 0.33])
    axz.imshow(im[100:148, 100:148], interpolation="nearest")
    axz.set_xticks([]); axz.set_yticks([])
    axz.set_title("48 px crop (480 m)", fontsize=9, color=MUT)
fig.text(0.06, 0.125, va="top", s=
    "The white boxes mark 480 m crops, shown enlarged. AlphaEarth stays\n"
    "coherent when magnified; Tessera reveals per-pixel variation. A codec\n"
    "must spend bits on everything that is real signal — the question the\n"
    "rest of this report answers is how much of that texture is learnable\n"
    "structure vs. effectively unpredictable.", fontsize=10)
pdf.savefig(fig); plt.close(fig)

# ---------------- page 3: power spectra ----------------
def spectrum(x, W=64):
    f = torch.fft.fftfreq(W, device=DEV)
    kb = (f[:, None]**2 + f[None, :]**2).sqrt().mul(W).round().long() \
        .clamp(max=32).flatten()
    P = torch.zeros(33, device=DEV)
    cnt = torch.bincount(kb, minlength=33).float()
    n = 0
    for i in range(0, x.shape[0], 16):
        w = x[i:i+16, :, :W, :W]
        F = torch.fft.fft2(w, norm="ortho").abs().pow(2).mean((0, 1))
        P.index_add_(0, kb, F.flatten()); n += 1
    return (P / n).cpu().numpy()   # index_add already sums cells -> bin totals

def spec_windows(store, rows, src_cfg):
    toks = fixed_val_tokens(store, rows, token=64, per_chip=4,
                            max_chips=256, max_tokens=512)
    cfg = load_config(src_cfg)
    from georvq.sources import get_source
    src = get_source(cfg.data.source, cfg.model)
    x, _ = src.prep(toks.to(DEV), DEV)
    return x

with torch.no_grad():
    Pt = spectrum(spec_windows(tstore, tval, "configs/tessera_stage1_4x.yaml"))
    Pa = spectrum(spec_windows(astore, aval, "configs/ae_stage1_patch4.yaml"))
fig, axp = plt.subplots(figsize=(11, 8.5))
fig.subplots_adjust(left=0.1, right=0.65, top=0.82, bottom=0.12)
k = np.arange(1, 32)
axp.plot(k, Pa[1:32] / Pa.sum(), color=C_AE, lw=2.2, label="AlphaEarth")
axp.plot(k, Pt[1:32] / Pt.sum(), color=C_TS, lw=2.2, label="Tessera")
axp.set(xscale="log", yscale="log",
        xlabel="spatial frequency (cycles per 640 m window)",
        ylabel="fraction of signal power",
        title="Tessera carries far more high-frequency signal")
axp.legend(frameon=False)
fig.text(0.68, 0.55,
    "Power spectra of the embedding\nfields (share of signal variance\n"
    "by spatial scale).\n\n"
    "Fine-scale content (right side)\nis the expensive part of any\n"
    "compression problem: it cannot\nbe summarized, only stored.\n\n"
    "Above ~8 cycles/window Tessera\nholds roughly an order of\n"
    "magnitude more power than\nAlphaEarth.", fontsize=10, va="top")
pdf.savefig(fig); plt.close(fig)

# ---------------- page 4: latent effective rank ----------------
fig, axp = plt.subplots(figsize=(11, 8.5))
fig.subplots_adjust(left=0.1, right=0.62, top=0.82, bottom=0.12)
tiers = ["coarse\n(640 m cell)", "mid\n(160 m)", "fine\n(40 m)"]
d95_ae, dim_ae = [19, 34, 28], [256, 192, 64]
d95_ts, dim_ts = [51, 91, 71], [512, 512, 256]
xpos = np.arange(3)
axp.bar(xpos - 0.18, d95_ae, 0.32, color=C_AE, label="AlphaEarth")
axp.bar(xpos + 0.18, d95_ts, 0.32, color=C_TS, label="Tessera")
for i in range(3):
    axp.text(xpos[i] - 0.18, d95_ae[i] + 1.5, str(d95_ae[i]),
             ha="center", fontsize=9, color=C_AE)
    axp.text(xpos[i] + 0.18, d95_ts[i] + 1.5, str(d95_ts[i]),
             ha="center", fontsize=9, color=C_TS)
axp.set(xticks=xpos, xticklabels=tiers,
        ylabel="dimensions holding 95% of variance",
        title="How many numbers does each region really need?")
axp.legend(frameon=False)
fig.text(0.65, 0.60,
    "We trained identical multi-scale\nencoders on both sources and\n"
    "measured the effective\ndimensionality of the learned\nsummaries.\n\n"
    "AlphaEarth content is intrinsically\nlow-dimensional (~20-35 numbers\n"
    "per region). Tessera needs 2-3x\nmore in every tier — its pixels\n"
    "vary in more independent ways.\n\n"
    "This is the representation-space\nview of the spectral difference on\n"
    "the previous page.", fontsize=10, va="top")
pdf.savefig(fig); plt.close(fig)

# ---------------- page 5: noise-floor R-D ----------------
fig, axp = plt.subplots(figsize=(11, 8.5))
fig.subplots_adjust(left=0.1, right=0.62, top=0.82, bottom=0.12)
ts_r = [4, 2, 1.333]; ts_d = [0.0182, 0.0146, 0.0088]
ae_r = [4, 2]; ae_d = [0.0054, 0.0036]
axp.plot(ts_r, ts_d, "o-", color=C_TS, lw=2, ms=7, label="Tessera probes")
axp.plot(ae_r, ae_d, "o-", color=C_AE, lw=2, ms=7, label="AlphaEarth probes")
axp.scatter([7.1], [0.0199], marker="*", s=260, color=C_TS, zorder=5)
axp.scatter([13.3], [0.002], marker="*", s=260, color=C_AE, zorder=5)
axp.set_ylim(0.0014, 0.032)
axp.annotate("deployed Tessera codec\n(7x compression)", (7.1, 0.0199),
             textcoords="offset points", xytext=(-118, -8), fontsize=9)
axp.annotate("deployed AlphaEarth\ncodec (13x)", (13.3, 0.002),
             textcoords="offset points", xytext=(-64, 16), fontsize=9)
axp.set(xscale="log", yscale="log", xlabel="compression factor",
        ylabel="reconstruction error (1 - cosine similarity)",
        title="The gap between the codecs is the gap between the sources")
axp.set_xticks([1.33, 2, 4, 7, 13]); axp.set_xticklabels(
    ["1.33x", "2x", "4x", "7x", "13x"])
axp.legend(frameon=False, loc="center right")
fig.text(0.65, 0.62,
    "Circles: 'best possible' probes —\nheavyweight models compressing\n"
    "single 4x4-pixel patches with no\ntask other than fidelity.\n\n"
    "At every matched rate, Tessera's\nerror is 3.5-4x AlphaEarth's; both\n"
    "fall smoothly with rate (no hard\nnoise wall).\n\n"
    "Stars: the actual deployed multi-\nscale codecs. Both beat their own\n"
    "per-patch bound — spatial context\nis worth about one compression\n"
    "octave — and each sits where its\nsource's curve says it must.",
    fontsize=10, va="top")
pdf.savefig(fig); plt.close(fig)

# ---------------- page 6: takeaways ----------------
fig = plt.figure(figsize=(11, 8.5))
fig.text(0.06, 0.93, "Takeaways", fontsize=17, weight="bold")
fig.text(0.06, 0.86, """
1.  The two sources pose genuinely different compression problems.
    AlphaEarth fields are smooth and low-dimensional; Tessera fields carry
    2-3x more independent variation per pixel and ~10x more fine-scale
    signal power.

2.  The codec is close to optimal on both. Deployed models beat their own
    context-free per-patch bounds at matched rates — multi-scale context
    buys roughly one compression octave. The quality difference between
    the deployed codecs (cos ~0.998 vs ~0.980) is a property of the
    sources, not of the method.

3.  Error is the right axis, not similarity. In error terms (1 - cos, log
    scale), both sources trade error for rate along smooth, similarly-shaped
    curves; AlphaEarth's sits ~4x lower. "0.996 vs 0.998" is a factor-of-two
    difference in error.

4.  Roadmap: Tessera quality targets should be set per-rate from its curve
    (e.g. ~0.99 ceiling requires roughly 2.5-3x naive-rate latents, now in
    training as a 4-tier model with a 10 m detail tier). The productive
    engineering axis is rate-at-fixed-quality: quantized codebooks (int8,
    ~72 MB download), residual VQ ladders, and entropy coding.
""", fontsize=11.5, va="top", family="sans-serif")
fig.text(0.06, 0.08, "georvq / Tessera integration - August 2026",
         fontsize=9, color=MUT)
pdf.savefig(fig); plt.close(fig)
pdf.close()
print("saved ->", OUT)
