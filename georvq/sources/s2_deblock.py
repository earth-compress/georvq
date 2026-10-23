"""Seam deblocker for Sentinel-2 window codecs, on the three native grids.

This is the boundary deblocker of the ring-space beta_v4 release
(`ring_space/deblock.py` there, ~139K parameters at width 32), adapted
from its 42 px tiles in 252 px chips to a georvq S2 codec's window. It
runs after the decoder on the stitched mosaic, adds nothing to the
bitstream, and enters the `decoder_id` with its tiling recipe
(georvq.model.deblock, georvq.inference.release.decoder_id).

The network is a small UNet over the 10, 20 and 60 m grids: one conv
stem per grid taking that grid's bands and two seam masks (window seams
and fine-cell seams, average-pooled onto the coarser grids), a stride-2
and a stride-3 conv down to the 60 m grid with the 20 m and 60 m stems
fused in by 1x1 convs, one FiLM residual block at 20 m on the way down,
one at 60 m and one at 20 m on the way up, and a zero-initialised 1x1
head per grid. Normalisation is BatchNorm, which is per pixel in eval
mode. The decode depth enters through a learned embedding of the fine
tier's depth (the coarse tier is sent in full at every rung) feeding
the FiLM layers, which are zero-initialised, so the untrained network
is the identity.

Each frame is standardised per band before the stems, and the heads'
residual is made zero-mean over each window on each grid, after which
every window's mean and standard deviation are reset to those of the
decoded input (`preserve_block_stats`). The deblocker can therefore
move detail across a seam but cannot change any window's mean or
spread, so a step between window means, whether from the decoder or
from the coded stats, passes through it unchanged.

Tiling follows georvq.model.deblock.deblock_mosaic with the frame and
halo in `RECIPE`: 5 x 5 windows per frame and one window of halo for a
36 px window, so every window seam lies inside some frame's input and
not on a frame edge. Because each frame is standardised over its own
extent, the frame and halo are part of the decoder's definition.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.model.deblock import deblock_mosaic, load_deblocker
from georvq.sources.s2 import GROUPS

#: frame and halo on the 10 m grid, for a 36 px window
RECIPE = {"frame": 180, "halo": 36, "blend": "crop", "norm": "frame",
          "version": 1}
SCALES = {g: r for g, (_, r) in GROUPS.items()}


def block_mean(x, b):
    """Mean of each b x b block of x, broadcast back to x's shape."""
    return F.interpolate(F.avg_pool2d(x, b), scale_factor=b, mode="nearest")


def block_mean_std(x, b):
    m = F.avg_pool2d(x, b)
    s = (F.avg_pool2d(x * x, b) - m * m).clamp_min(1e-12).sqrt()
    up = lambda t: F.interpolate(t, scale_factor=b, mode="nearest")  # noqa
    return up(m), up(s)


def preserve_block_stats(modified, original, b):
    """`modified` with each b x b block's mean and std set to those of
    the same block of `original`."""
    om, os_ = block_mean_std(original, b)
    mm, ms = block_mean_std(modified, b)
    return (modified - mm) / ms.clamp_min(1e-6) * os_ + om


def seam_masks(h, w, window, cell, device):
    """(1, 2, h, w) on the 10 m grid: window seams, and fine-cell seams
    that are not window seams. A seam between pixels k-1 and k is marked
    at k, so a frame cut on the window grid has its seams at multiples
    of `window` from its own origin."""
    def on(n, p):
        i = torch.arange(n, device=device)
        return (i % p == 0) & (i > 0)
    win = (on(h, window)[:, None] | on(w, window)[None, :]).float()
    cel = (on(h, cell)[:, None] | on(w, cell)[None, :]).float() * (1 - win)
    return torch.stack([win, cel])[None]


def to_windows(x, window):
    """{group: (B, C, H/r, W/r)} cut on the window grid ->
    {group: (B * nh * nw, C, window/r, window/r)}, windows row-major
    within each frame."""
    out = {}
    for g, t in x.items():
        p = window // SCALES[g]
        B, C, h, w = t.shape
        out[g] = t.reshape(B, C, h // p, p, w // p, p) \
            .permute(0, 2, 4, 1, 3, 5).reshape(-1, C, p, p)
    return out


def from_windows(x, nh, nw, window):
    """Inverse of `to_windows` for frames of nh x nw windows."""
    out = {}
    for g, t in x.items():
        p = window // SCALES[g]
        C = t.shape[1]
        out[g] = t.reshape(-1, nh, nw, C, p, p).permute(0, 3, 1, 4, 2, 5) \
            .reshape(-1, C, nh * p, nw * p)
    return out


def _cbg(cin, cout, k=3, stride=1, pad=1):
    return [nn.Conv2d(cin, cout, k, stride=stride, padding=pad),
            nn.BatchNorm2d(cout), nn.GELU()]


class FiLMResBlock(nn.Module):
    """Pre-activation 3x3 residual block with BatchNorm and FiLM between
    the convs; FiLM starts at zero."""

    def __init__(self, width, cond_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        g, b = self.film(cond).chunk(2, dim=-1)
        h = self.conv1(F.gelu(self.norm1(x)))
        h = h * (1.0 + g[..., None, None]) + b[..., None, None]
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


class S2BoundaryDeblocker(nn.Module):
    """beta_v4's BoundaryDeblocker on a georvq S2 window.

    forward(x, Ls): x is {"10m": (B, 4, H, W), "20m": (B, 6, H/2, W/2),
    "60m": (B, 3, H/6, W/6)} decoded reflectance (DN / 10000) with H, W
    multiples of `window`, cut on the window grid; Ls is (B, n_tiers)
    decode depths, of which the last (the fine tier) selects the
    conditioning. Returns a dict of the same shapes.

    `forward_parts` also returns the zero-mean residuals in the
    standardised space, which is where the training loss is taken.
    """

    def __init__(self, width=32, cond_dim=64, depth=8, window=36, cell=6):
        super().__init__()
        assert window % 6 == 0 and window % cell == 0
        self.window, self.cell, self.depth = window, cell, depth
        self.bands = {g: n for g, (n, _) in GROUPS.items()}
        self.l_embed = nn.Embedding(depth + 1, cond_dim)
        nb = self.bands
        self.stem_10m = nn.Sequential(*_cbg(nb["10m"] + 2, width))
        self.stem_20m = nn.Sequential(*_cbg(nb["20m"] + 2, width),
                                      *_cbg(width, width))
        self.stem_60m = nn.Sequential(*_cbg(nb["60m"] + 2, width),
                                      *_cbg(width, width))
        self.down_10_to_20 = nn.Sequential(*_cbg(width, width, stride=2))
        self.fuse_20 = nn.Conv2d(2 * width, width, 1)
        self.res_20m_down = FiLMResBlock(width, cond_dim)
        self.down_20_to_60 = nn.Sequential(*_cbg(width, width, stride=3,
                                                 pad=0))
        self.fuse_60 = nn.Conv2d(2 * width, width, 1)
        self.res_60m_bottleneck = FiLMResBlock(width, cond_dim)
        self.up_60_to_20 = nn.Sequential(
            nn.Upsample(scale_factor=3, mode="nearest"), *_cbg(width, width))
        self.skip_fuse_20 = nn.Conv2d(2 * width, width, 1)
        self.res_20m_up = FiLMResBlock(width, cond_dim)
        self.up_20_to_10 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), *_cbg(width, width))
        self.skip_fuse_10 = nn.Conv2d(2 * width, width, 1)
        self.head_10m = nn.Conv2d(width, nb["10m"], 1)
        self.head_20m = nn.Conv2d(width, nb["20m"], 1)
        self.head_60m = nn.Conv2d(width, nb["60m"], 1)
        for h in (self.head_10m, self.head_20m, self.head_60m):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    @staticmethod
    def frame_stats(t):
        """Per (frame, band) mean and unbiased std over the frame."""
        f = t.flatten(2)
        return (f.mean(-1)[..., None, None],
                f.std(-1).clamp_min(1e-6)[..., None, None])

    def forward_parts(self, x, Ls):
        cond = self.l_embed(Ls[:, -1].long())
        stats = {g: self.frame_stats(x[g]) for g in GROUPS}
        z = {g: (x[g] - stats[g][0]) / stats[g][1] for g in GROUPS}
        B, _, H, W = x["10m"].shape
        m = seam_masks(H, W, self.window, self.cell, x["10m"].device)
        m = m.to(x["10m"].dtype).expand(B, -1, -1, -1)
        mk = {g: (m if r == 1 else F.avg_pool2d(m, r))
              for g, r in SCALES.items()}

        s10 = self.stem_10m(torch.cat([z["10m"], mk["10m"]], 1))
        s20 = self.stem_20m(torch.cat([z["20m"], mk["20m"]], 1))
        s60 = self.stem_60m(torch.cat([z["60m"], mk["60m"]], 1))
        d20 = self.res_20m_down(
            self.fuse_20(torch.cat([self.down_10_to_20(s10), s20], 1)), cond)
        d60 = self.res_60m_bottleneck(
            self.fuse_60(torch.cat([self.down_20_to_60(d20), s60], 1)), cond)
        u20 = self.res_20m_up(
            self.skip_fuse_20(torch.cat([self.up_60_to_20(d60), d20], 1)),
            cond)
        u10 = self.skip_fuse_10(torch.cat([self.up_20_to_10(u20), s10], 1))
        raw = {"10m": self.head_10m(u10), "20m": self.head_20m(u20),
               "60m": self.head_60m(d60)}

        deltas, out = {}, {}
        for g, r in SCALES.items():
            b = self.window // r
            d = raw[g] - block_mean(raw[g], b)
            y = preserve_block_stats(z[g] + d, z[g], b)
            deltas[g] = d
            out[g] = y * stats[g][1] + stats[g][0]
        return out, deltas, stats

    def forward(self, x, Ls):
        return self.forward_parts(x, Ls)[0]


def residual_loss(net, recon, target, Ls, valid=None):
    """beta_v4's loss: MSE between the predicted zero-mean residual and
    the target residual made zero-mean over each window, both in the
    frame-standardised space, summed over the three grids. Where `valid`
    ({group: (B, 1, h, w) bool}) is False the target is taken to be the
    input, so nodata pixels ask for no change."""
    out, deltas, stats = net.forward_parts(recon, Ls)
    parts = {}
    for g, r in SCALES.items():
        t = target[g]
        if valid is not None:
            t = torch.where(valid[g], t, recon[g])
        td = (t - recon[g]) / stats[g][1]
        td = td - block_mean(td, net.window // r)
        parts[g] = F.mse_loss(deltas[g], td)
    return sum(parts.values()), parts, out


def deblock_s2_mosaic(net, x, Ls, recipe=None, batch=1):
    """Deblock a stitched S2 mosaic {group: (1, C, H / r, W / r)} whose
    window grid starts at (0, 0), by the frame/halo recipe."""
    recipe = recipe or RECIPE
    return deblock_mosaic(net, x, Ls, frame=recipe["frame"],
                          halo=recipe["halo"], scales=SCALES, batch=batch)


def load_s2_deblocker(path, device="cpu"):
    """Saved S2 deblocker -> (eval-mode net, entry dict), in the layout
    of georvq.model.deblock.load_deblocker."""
    return load_deblocker(path, device, cls=S2BoundaryDeblocker)
