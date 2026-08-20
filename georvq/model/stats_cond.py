"""Stats conditioning: transmit the patch mean, code what is left.

Ported from the Sentinel-2 work in ring-space (`ring_space/compression.py`,
`_normalize_bands` / `stats_proj`), narrowed to the mean only.

WHY MEAN ONLY, when the S2 version standardises by mean AND std:

  * The mean is 85.5% of a Tessera window's energy and is cheap and
    accurate to send (a 6-bit/band channel reaches cos 0.9767 against
    0.9769 for the exact mean).
  * The std is the badly-coded half -- 0.13 relative error even at 512
    bits -- and forcing a reconstruction to match it MEASURABLY HURTS
    (full-depth cosine 0.9700 -> 0.9678 on the k256 checkpoint). The
    decoder's output is smoother than the truth, so rescaling to the true
    spread amplifies its error instead of restoring detail.
  * S2 bands have different physical scales and per-scene illumination, so
    standardising is obviously right there. Tessera's 128 dims are a
    learned embedding already on comparable scales.

The mean is subtracted from the field before the encoder and added back
after the decoder, so the loss and every metric stay on the FULL field and
remain directly comparable to a run without conditioning. Both encoder and
decoder are told the mean: the encoder no longer sees it once it is
subtracted, and without it cannot know where in embedding space the patch
sits.

Rate: the channel is charged separately (`bits_per_band` x bands), not by
the pyramid's index accounting. The ring-space lesson is that this is not
optional -- raw fp16 stats there were 34% of the bitstream at L=1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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


def center(x, mu, valid=None):
    """Subtract the per-window mean, leaving nodata pixels at zero.

    Keeping nodata at exactly zero matters: `Source.prep` marks them by
    all-bands-zero, and shifting them by -mu would turn every nodata pixel
    into a spurious constant the encoder has to model."""
    xc = x - mu[:, :, None, None]
    if valid is not None:
        xc = xc * valid.unsqueeze(1).to(xc.dtype)
    return xc


def uncenter(out, mu, valid=None):
    o = out + mu[:, :, None, None]
    if valid is not None:
        o = o * valid.unsqueeze(1).to(o.dtype)
    return o


class StatsEmbed(nn.Module):
    """Mean vector -> a `dim`-wide conditioning bias.

    LayerNorm first, mirroring ring-space's `stats_norm`: the raw mean has
    a large, band-dependent offset and feeding it unnormalised makes the
    first Linear's job scale-dependent."""

    def __init__(self, bands, dim):
        super().__init__()
        self.norm = nn.LayerNorm(bands, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(bands, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))
        # zero-init the last layer so training starts exactly where an
        # unconditioned run starts, and the conditioning has to earn its
        # contribution (same trick as ring-space's zero-init FuseDec)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, mu):
        return self.mlp(self.norm(mu))


def channel_bits(cfg_stats, bands):
    """Bits per window for the mean side channel (rate accounting)."""
    return bands * cfg_stats.bits_per_band
