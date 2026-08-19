"""Variable-compression GIF panels for embedding-field codecs.

Renders original vs reconstruction under two PCA colourings (PCs 0-2 and
PCs 3-5 as RGB; each a 50/50 blend of a global and a per-patch PCA -- canonical
tone + individual structure) plus a per-pixel all-channel cosine-similarity
quality map on a fixed scale. The rate sweep merges the first checkpoint's
full operating range with any further checkpoint's high-bitrate tail, and
animates it over the styled compression bar with end pauses.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from georvq.eval.codec_ab import load_codec
from georvq.eval.extract import dequant_np, l2norm_np
from georvq.model.codec import tuple_bits
from georvq.sources.ae import dequant
from georvq.viz.style import (COS_CMAP, FRAME_MS, HOLD_MS, MUTED, TEXT,
                              apply_style, axes_center, compression_bar,
                              cos_colorbar, frame_axes, save_gif,
                              set_bar_ratio)

QUADS = [(0, 0), (0, 1), (1, 0), (1, 1)]
GROUPS = [(0, 1, 2), (3, 4, 5)]
BLEND = 0.5                                              # global vs per-patch PCA


# --------------------------------------------------------------------------
# PCA-RGB rendering
# --------------------------------------------------------------------------

def chip_px(emb1):
    """(C, H, W) embedding plane -> (H*W, C) unit-norm pixel matrix."""
    C = emb1.shape[0]
    return l2norm_np(emb1.transpose(1, 2, 0).reshape(-1, C))


def pick_chips(path, n=4, pool=300, seed=2):
    """The n highest-structure chips from a random pool of a windows array."""
    arr = np.load(path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    cand = np.sort(rng.permutation(len(arr))[:pool])
    sub = np.asarray(arr[cand]).astype(np.float32)
    score = sub.std(axis=(2, 3)).mean(axis=1)
    return np.asarray(arr[np.sort(cand[np.argsort(score)[-n:]])])


def fit_pca(px):
    """6-component PCA + per-group 2/98-percentile display ranges."""
    pca = PCA(6).fit(px[::4] if len(px) > 70000 else px)
    proj = pca.transform(px)
    lohi = [(np.percentile(proj[:, g], 2, 0), np.percentile(proj[:, g], 98, 0))
            for g in GROUPS]
    return pca, lohi


def rgb_mapper(chips_i8):
    """50/50 blend of a global (all-patch) PCA and a per-patch PCA -> two RGBs
    per patch (PCs 0-2, PCs 3-5). Same mapping for a patch's orig & recons."""
    n, _, H, Wd = chips_i8.shape
    o = dequant_np(chips_i8)
    gpca, glohi = fit_pca(np.concatenate([chip_px(o[i]) for i in range(n)], 0))
    ppca = [fit_pca(chip_px(o[i])) for i in range(n)]

    def norm(proj, lohi, gi):
        lo, hi = lohi[gi]
        return np.clip((proj[:, GROUPS[gi]] - lo) / (hi - lo), 0, 1)

    def to_rgb(emb):
        """Embedding chips -> RGB, blending a GLOBAL PCA basis with each
        chip's own local basis.

        The global part keeps colours comparable across chips; the local
        part restores within-chip contrast that a global basis flattens.
        Returned as a closure so a reconstruction can be rendered through
        exactly the basis its original was.
        """
        out = np.zeros((n, 2, H, Wd, 3), np.float32)
        for i in range(n):
            px = chip_px(emb[i])
            gp, pp = gpca.transform(px), ppca[i][0].transform(px)
            for gi in range(2):
                blend = (BLEND * norm(gp, glohi, gi)
                         + (1 - BLEND) * norm(pp, ppca[i][1], gi))
                out[i, gi] = blend.reshape(H, Wd, 3)
        return out
    return to_rgb(dequant_np(chips_i8)), to_rgb


def global_pca(windows_path, max_chips=160, step=2):
    """Fit the canonical global PCA on a spread of stored eval windows."""
    arr = np.load(windows_path, mmap_mode="r")
    px = np.concatenate([chip_px(dequant_np(np.asarray(arr[k:k + 1]))[0])
                         for k in range(0, min(max_chips, len(arr)), step)], 0)
    return fit_pca(px)


def panel_mapper(gpca, glohi, chip_i8):
    """Global+per-slice blended PCA mapper for one large chip. Returns
    (orig two-group RGB, to_rgb fn for reconstructions)."""
    ppca, plohi = fit_pca(chip_px(dequant_np(chip_i8)))
    H, Wd = chip_i8.shape[1], chip_i8.shape[2]

    def norm(proj, lohi, gi):
        lo, hi = lohi[gi]
        return np.clip((proj[:, GROUPS[gi]] - lo) / (hi - lo), 0, 1)

    def to_rgb(emb):                                      # (C,H,W)->(2,H,W,3)
        p = chip_px(emb)
        gp, pp = gpca.transform(p), ppca.transform(p)
        out = np.zeros((2, H, Wd, 3), np.float32)
        for gi in range(2):
            out[gi] = (BLEND * norm(gp, glohi, gi)
                       + (1 - BLEND) * norm(pp, plohi, gi)).reshape(H, Wd, 3)
        return out
    return to_rgb(dequant_np(chip_i8)), to_rgb


# --------------------------------------------------------------------------
# reconstruction across a rate sweep
# --------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_chips(model, chips_i8, tuples, W, device):
    """Reconstruct (B, C, S, S) chips at each rate tuple; chips twice the
    codec window are split into quadrant windows and reassembled."""
    chips = torch.from_numpy(chips_i8).to(device)
    B, C, S = chips.shape[0], chips.shape[1], chips.shape[-1]
    if W == S:
        x = dequant(chips)
    else:
        q = torch.stack([chips[:, :, r*W:(r+1)*W, c*W:(c+1)*W]
                         for r, c in QUADS], 1).reshape(B*4, C, W, W)
        x = dequant(q)
    _, cums = model.level_cumsums(x)
    outs = []
    for t in tuples:
        o = model.eval_tuple(cums, t, x).float()
        if W != S:
            o = o.reshape(B, 4, C, W, W)
            full = torch.zeros(B, C, S, S, device=device)
            for k, (r, c) in enumerate(QUADS):
                full[:, :, r*W:(r+1)*W, c*W:(c+1)*W] = o[:, k]
            o = full
        outs.append(o.cpu().numpy())
    return outs


@torch.no_grad()
def reconstruct_slice(model, chip_i8, tuples, W, device):
    """Tile one (C, S, S) slice into the codec's native windows, encode once,
    decode every tuple, and reassemble to (C, S, S) per tuple."""
    C, S = chip_i8.shape[0], chip_i8.shape[-1]
    T = S // W
    c = torch.from_numpy(chip_i8).to(device)
    wins = c.reshape(C, T, W, T, W).permute(1, 3, 0, 2, 4).reshape(T*T, C, W, W)
    x = dequant(wins)
    _, cums = model.level_cumsums(x)
    outs = []
    for t in tuples:
        o = model.eval_tuple(cums, t, x).float()
        outs.append(o.reshape(T, T, C, W, W).permute(2, 0, 3, 1, 4)
                    .reshape(C, S, S).cpu().numpy())
    return outs


def _merge_entries(md, tuples):
    """Rate-sweep order: model 0 over its whole range, then any further
    model's tail below model 0's lowest ratio; sorted high -> low ratio."""
    base_min = md[0]["ratio"].min()
    entries = [(0, j) for j in range(len(tuples))]
    for m in range(1, len(md)):
        entries += [(m, j) for j in range(len(tuples))
                    if md[m]["ratio"][j] < base_min]
    entries.sort(key=lambda e: -md[e[0]]["ratio"][e[1]])
    return entries


# --------------------------------------------------------------------------
# GIF builders
# --------------------------------------------------------------------------

def build_grid_gif(chips, models, specs, tuples, out, device):
    """Multi-patch grid GIF: rows of patches, columns ORIGINAL /
    RECONSTRUCTION (two PCA groups each) / COSINE SIMILARITY, animated over
    the merged rate sweep. `models`: [(name, ckpt_path)]."""
    apply_style()
    orig_rgb, to_rgb = rgb_mapper(chips)
    n = chips.shape[0]
    ox = dequant_np(chips)
    ox = ox / np.maximum(np.linalg.norm(ox, axis=1, keepdims=True), 1e-12)

    bands = None
    md = []                                              # per-model precompute
    for name, ck in models:
        model, W, _ = load_codec(ck, device)
        bands = model.bands
        bpp = np.array([tuple_bits(specs, t) / (W * W) for t in tuples])
        recons = reconstruct_chips(model, chips, tuples, W, device)
        md.append(dict(name=name, W=W, bpp=bpp, ratio=8.0 * bands / bpp,
                       rgb=[to_rgb(r) for r in recons],
                       cos=[(ox * r).sum(axis=1) for r in recons]))

    entries = _merge_entries(md, tuples)
    allcos = np.concatenate([md[m]["cos"][j].ravel() for m, j in entries])
    vmin, vmax = float(np.percentile(allcos, 1.0)), 1.0
    rr = [md[m]["ratio"][j] for m, j in entries]

    fig = plt.figure(figsize=(9.4, 8.4), dpi=120)
    gs = fig.add_gridspec(n + 1, 6, width_ratios=[1, 1, 1, 1, 1, 0.07],
                          height_ratios=[1]*n + [0.16], hspace=0.05,
                          wspace=0.05, left=0.014, right=0.93, top=0.86,
                          bottom=0.115)
    im, cos_im, ta = {}, [], []
    for i in range(n):
        for g in range(2):                                # ORIGINAL: cols 0,1
            ax = fig.add_subplot(gs[i, g]); ax.imshow(orig_rgb[i, g])
            frame_axes(ax)
            if i == 0:
                ta.append(ax)
        for g in range(2):                                # RECONSTRUCTION: 2,3
            ax = fig.add_subplot(gs[i, 2 + g])
            im[(i, g)] = ax.imshow(orig_rgb[i, g]); frame_axes(ax)
            if i == 0:
                ta.append(ax)
        ax = fig.add_subplot(gs[i, 4])                    # COSINE SIM: col 4
        cos_im.append(ax.imshow(md[0]["cos"][0][i], cmap=COS_CMAP,
                                vmin=vmin, vmax=vmax)); frame_axes(ax)
        if i == 0:
            ta.append(ax)

    for x, lab in [(axes_center(ta[0], ta[1]), "ORIGINAL"),
                   (axes_center(ta[2], ta[3]), "RECONSTRUCTION"),
                   (axes_center(ta[4]), "COSINE SIMILARITY")]:
        fig.text(x, 0.9, lab, ha="center", fontsize=10.5, weight="bold",
                 color=TEXT)
    for ax, lab in zip(ta[:4], ["1st order PCA", "2nd order PCA",
                                "1st order PCA", "2nd order PCA"]):
        fig.text(axes_center(ax), 0.879, lab, ha="center", fontsize=7.5,
                 color=MUTED)
    fig.text(0.014, 0.965, "AlphaEarth Variable Compression", ha="left",
             fontsize=17, weight="bold", color=TEXT)
    fig.text(0.014, 0.937, f"{bands}-channel embedding reconstruction across "
             "the compression–quality curve", ha="left", fontsize=9.5,
             color=MUTED)

    cax = fig.add_subplot(gs[0:n, 5])
    cos_colorbar(fig, cax, vmin, vmax)

    bar = fig.add_subplot(gs[n, 0:5])
    fill, dot, rmin = compression_bar(bar, rr, lw=15)

    seq = entries + entries[-2:0:-1]
    ends = {entries[0], entries[-1]}
    frames, durations = [], []
    for m, j in seq:
        d = md[m]
        for (i, g), h in im.items():
            h.set_data(d["rgb"][j][i, g])
        for i in range(n):
            cos_im[i].set_data(d["cos"][j][i])
        set_bar_ratio(bar, fill, dot, rmin, d["ratio"][j], d["W"],
                      d["bpp"][j], fontsize=12.5)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        durations.append(HOLD_MS if (m, j) in ends else FRAME_MS)
    plt.close(fig)
    save_gif(frames, out, durations)
    print(f"wrote {out}  ({len(frames)} frames, {max(rr):,.0f}x->{min(rr):,.0f}x"
          f", cos {vmin:.3f}-1.0, {len(entries)} rate points)", flush=True)


def build_panel(chip_i8, orig_rgb, to_rgb, specs, tuples, models, out, device):
    """Single-slice panel GIF: column ORIGINAL (two PCA groups stacked),
    column RECONSTRUCTION, column COSINE SIMILARITY, animated over the merged
    rate sweep. `models`: [(name, ckpt_path)]."""
    apply_style()
    S = chip_i8.shape[-1]
    ox = dequant_np(chip_i8)
    ox = ox / np.maximum(np.linalg.norm(ox, axis=0, keepdims=True), 1e-12)
    md = []
    for name, ck in models:
        model, W, _ = load_codec(ck, device)
        bpp = np.array([tuple_bits(specs, t) / (W * W) for t in tuples])
        rec = reconstruct_slice(model, chip_i8, tuples, W, device)
        md.append(dict(W=W, bpp=bpp, ratio=8.0 * model.bands / bpp,
                       rgb=[to_rgb(r) for r in rec],
                       cos=[(ox * r).sum(axis=0) for r in rec]))
    entries = _merge_entries(md, tuples)
    allc = np.concatenate([md[m]["cos"][j].ravel() for m, j in entries])
    vmin = float(np.percentile(allc, 1.0))
    rr = [md[m]["ratio"][j] for m, j in entries]

    fig = plt.figure(figsize=(8.6, 6.7), dpi=135)
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.045],
                          height_ratios=[1, 1, 0.13], hspace=0.05, wspace=0.05,
                          left=0.075, right=0.95, top=0.84, bottom=0.115)
    cells = {}
    for r in range(2):
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(orig_rgb[r]); frame_axes(ax)
        cells[("o", r)] = ax
        ax = fig.add_subplot(gs[r, 1]); cells[("r", r)] = ax
        ax.imshow(orig_rgb[r]); frame_axes(ax)
    axc = fig.add_subplot(gs[0:2, 2])
    cosim = axc.imshow(md[0]["cos"][0], cmap=COS_CMAP, vmin=vmin, vmax=1.0)
    frame_axes(axc)

    for a, lab in [(cells[("o", 0)], "ORIGINAL"), (cells[("r", 0)],
                   "RECONSTRUCTION"), (axc, "COSINE SIMILARITY")]:
        fig.text(axes_center(a), 0.86, lab, ha="center", fontsize=11,
                 weight="bold", color=TEXT)
    for r, lab in enumerate(["1st order PCA", "2nd order PCA"]):
        p = cells[("o", r)].get_position()
        fig.text(0.052, (p.y0 + p.y1)/2, lab, ha="center", va="center",
                 rotation=90, fontsize=9, color=MUTED)
    fig.text(0.075, 0.95, "AlphaEarth Variable Compression", ha="left",
             fontsize=16, weight="bold", color=TEXT)
    fig.text(0.075, 0.915, f"{S}px embedding slice · reconstruction across "
             "the compression–quality curve", ha="left", fontsize=9,
             color=MUTED)

    cax = fig.add_subplot(gs[0:2, 3])
    cos_colorbar(fig, cax, vmin, 1.0)

    bar = fig.add_subplot(gs[2, 0:3])
    fill, dot, rmin = compression_bar(bar, rr, lw=14)

    seq = entries + entries[-2:0:-1]
    ends = {entries[0], entries[-1]}
    frames, durations = [], []
    for m, j in seq:
        d = md[m]
        cells[("r", 0)].images[0].set_data(d["rgb"][j][0])
        cells[("r", 1)].images[0].set_data(d["rgb"][j][1])
        cosim.set_data(d["cos"][j])
        set_bar_ratio(bar, fill, dot, rmin, d["ratio"][j], d["W"],
                      d["bpp"][j], fontsize=12)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        durations.append(HOLD_MS if (m, j) in ends else FRAME_MS)
    plt.close(fig)
    save_gif(frames, out, durations)
    print(f"wrote {out}  ({len(frames)} frames, {max(rr):,.0f}x->{min(rr):,.0f}x)",
          flush=True)
