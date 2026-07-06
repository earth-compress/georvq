"""Shared dark-theme styling for rendered panels and GIFs.

Palette constants, font registration + rcParams, styled image-cell frames,
the cosine-similarity colorbar, the animated log-scale compression bar, and
palette-quantized GIF assembly.
"""
from __future__ import annotations

import glob

import matplotlib
import numpy as np

BG, TRACK, TEXT, MUTED = "#0f1116", "#272b35", "#ededed", "#8b909b"
ACCENT, REDC, BORDER = "#5aa9f5", "#eb5e55", "#33373f"
COS_CMAP = "Reds_r"                                      # white=100% -> red=low
HOLD_MS, FRAME_MS = 2000, 280


def apply_style():
    """Register Lato (if installed) and apply the dark figure theme."""
    from matplotlib import font_manager
    for f in glob.glob("/usr/share/fonts/truetype/lato/Lato-*.ttf"):
        try:
            font_manager.fontManager.addfont(f)
        except Exception:                                # noqa: BLE001
            pass
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Lato", "DejaVu Sans"],
        "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
        "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": BORDER,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    })


def frame_axes(ax):
    """Styled image cell: no ticks, thin border-coloured spines."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BORDER); sp.set_linewidth(0.9)


def axes_center(a, b=None):
    """Figure-fraction x centre of one axes (or the span of two)."""
    pa, pb = a.get_position(), (b or a).get_position()
    return (pa.x0 + pb.x1) / 2


def cos_colorbar(fig, cax, vmin, vmax=1.0):
    """Percent-labelled cosine-similarity colorbar on `cax`."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cb = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), COS_CMAP), cax=cax)
    cb.set_ticks([vmin, (vmin + 1) / 2, 1.0])
    cb.set_ticklabels([f"{t*100:.0f}%" for t in (vmin, (vmin + 1) / 2, 1.0)])
    cb.outline.set_edgecolor(BORDER); cb.outline.set_linewidth(0.9)
    cax.tick_params(colors=MUTED, labelsize=8, length=0)
    return cb


def compression_bar(bar, ratios, lw=15):
    """Style `bar` as a log-scale compression-ratio track with a fill and a
    marker dot. Returns (fill, dot, rmin) for set_bar_ratio updates."""
    lo_x, hi_x = min(ratios) / 1.6, max(ratios) * 1.6
    ticks = [t for t in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000,
                         50000) if lo_x < t < hi_x]
    bar.set_facecolor("none")
    for s in bar.spines.values():
        s.set_visible(False)
    bar.set_xscale("log"); bar.set_xlim(lo_x, hi_x); bar.set_ylim(0, 1)
    bar.set_yticks([]); bar.set_xticks(ticks)
    bar.set_xticklabels([f"{t//1000}k" if t >= 1000 else f"{t}" for t in ticks])
    bar.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0, pad=7)
    rmin = min(ratios)
    bar.plot([rmin, max(ratios)], [0.5, 0.5], color=TRACK, lw=lw,
             solid_capstyle="round", zorder=1)
    fill, = bar.plot([rmin, rmin], [0.5, 0.5], color=ACCENT, lw=lw,
                     solid_capstyle="round", zorder=2)
    dot, = bar.plot([rmin], [0.5], "o", color=TEXT, ms=6, mec=ACCENT,
                    mew=1.5, zorder=3)
    return fill, dot, rmin


def set_bar_ratio(bar, fill, dot, rmin, ratio, window, bpp, fontsize=12.5):
    """Move the bar fill/dot to `ratio` and set the rate caption."""
    fill.set_xdata([rmin, ratio]); dot.set_xdata([ratio])
    bar.set_xlabel(f"{ratio:,.0f}× compression      "
                   f"{window}px  ·  {bpp:.3f} bits/pixel",
                   color=TEXT, fontsize=fontsize, fontweight="bold",
                   labelpad=9)


def save_gif(frames, out, durations):
    """Palette-quantized looping GIF from (H,W,3) uint8 frames."""
    from PIL import Image
    try:
        nodither, medcut = Image.Dither.NONE, Image.Quantize.MEDIANCUT
    except AttributeError:
        nodither, medcut = 0, 2
    pal = Image.fromarray(np.concatenate(frames[::3], 0)).quantize(
        colors=256, method=medcut)
    q = [Image.fromarray(f).quantize(palette=pal, dither=nodither)
         for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=durations,
              loop=0, disposal=1)
