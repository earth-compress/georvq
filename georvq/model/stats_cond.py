"""Stats conditioning: transmit the patch stats, code what is left.

Ported from the Sentinel-2 work in ring-space (`ring_space/compression.py`,
`_normalize_bands` / `stats_proj`). The channel carries the per-window
mean, and with `use_std` also the per-window std, in which case the field
is fully standardized before the encoder.

WHICH STATS is a property of the field, not of the mechanism
(lab/findings/stats-channel-is-per-field.md):

  * The mean paid on Tessera: 85.5% of a window's energy, cheap and
    accurate to send (a 6-bit/band channel reaches cos 0.9767 against
    0.9769 for the exact mean).
  * The std HURT on Tessera (full-depth cosine 0.9700 -> 0.9678 on the
    k256 checkpoint): the decoder's output is smoother than the truth,
    and rescaling to the true spread amplifies its error instead of
    restoring detail. Tessera's dims are a learned embedding already on
    comparable scales, so there was no scale variation to remove.
  * Physical fields with real per-window scale variation standardise by
    both: S2 (per-scene illumination) and DEM (relief spans orders of
    magnitude; `DEMNormalizer` in georvq/sources/dem.py does it in the
    source, with a std floor against near-flat windows).

The stats are removed from the field before the encoder and restored
after the decoder, so the loss and every metric stay on the FULL field
and remain directly comparable to a run without conditioning. Encoder and
decoder are both told the stats: the encoder no longer sees them once
they are removed, and without them cannot know where the patch sits.

Rate: the channel is charged separately (`bits_per_band` x bands, doubled
by `use_std`), not by the pyramid's index accounting. The ring-space
lesson is that this is not optional -- raw fp16 stats there were 34% of
the bitstream at L=1. The inference-time codec (georvq/inference/
mean_codec.py) covers the mean channel only; a use_std model needs a std
quantizer before it can ship a real bitstream.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _split(stats, bands):
    """(B, bands) -> (mu, None); (B, 2*bands) -> (mu, sd)."""
    if stats.shape[1] == bands:
        return stats, None
    return stats[:, :bands], stats[:, bands:]


@torch.no_grad()
def patch_mean(x, valid=None, eps=1e-6):
    """Per-window mean over VALID pixels. x: (B, C, W, W) -> (B, C).

    nodata pixels are excluded: they are exactly zero in the source, so
    including them would drag the mean toward zero by the nodata fraction
    and make the conditioning signal depend on cloud/edge geometry."""
    if valid is None:
        return x.mean(dim=(2, 3))
    vm = valid.unsqueeze(1).to(x.dtype)
    return (x * vm).sum((2, 3)) / vm.sum((2, 3)).clamp_min(eps)


@torch.no_grad()
def patch_stats(x, valid=None, use_std=False, std_floor=0.05, eps=1e-6):
    """The stats vector the channel transmits: (B, C) mean, or with
    `use_std` (B, 2C) as [mean ‖ std], both over valid pixels. The floor
    keeps a near-uniform window from amplifying noise when divided."""
    mu = patch_mean(x, valid, eps)
    if not use_std:
        return mu
    d = x - mu[:, :, None, None]
    if valid is None:
        var = d.pow(2).mean(dim=(2, 3))
    else:
        vm = valid.unsqueeze(1).to(x.dtype)
        var = (d.pow(2) * vm).sum((2, 3)) / vm.sum((2, 3)).clamp_min(eps)
    return torch.cat([mu, var.sqrt().clamp_min(std_floor)], dim=1)


def center(x, stats, valid=None):
    """Remove the transmitted stats: subtract the mean and, when the
    vector carries one, divide by the std. Nodata pixels stay at zero.

    Keeping nodata at exactly zero matters: `Source.prep` marks them by
    all-bands-zero, and shifting them by -mu would turn every nodata pixel
    into a spurious constant the encoder has to model."""
    mu, sd = _split(stats, x.shape[1])
    xc = x - mu[:, :, None, None]
    if sd is not None:
        xc = xc / sd[:, :, None, None]
    if valid is not None:
        xc = xc * valid.unsqueeze(1).to(xc.dtype)
    return xc


def uncenter(out, stats, valid=None):
    mu, sd = _split(stats, out.shape[1])
    if sd is not None:
        out = out * sd[:, :, None, None]
    o = out + mu[:, :, None, None]
    if valid is not None:
        o = o * valid.unsqueeze(1).to(o.dtype)
    return o


class StatsEmbed(nn.Module):
    """Stats vector -> a `dim`-wide conditioning bias. `width` is the
    vector's length: bands for the mean, 2x bands with `use_std`.

    LayerNorm first, mirroring ring-space's `stats_norm`: the raw mean has
    a large, band-dependent offset and feeding it unnormalised makes the
    first Linear's job scale-dependent."""

    def __init__(self, width, dim):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))
        # zero-init the last layer so training starts exactly where an
        # unconditioned run starts, and the conditioning has to earn its
        # contribution (same trick as ring-space's zero-init FuseDec)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, mu):
        return self.mlp(self.norm(mu))


def channel_bits(cfg_stats, bands):
    """Bits per window for the stats side channel (rate accounting)."""
    return bands * cfg_stats.bits_per_band * (2 if cfg_stats.use_std else 1)
