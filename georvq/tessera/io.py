"""Getting Tessera data into the codec.

`georvq.sources.tessera` reads the project's own chip store, whose raw
batches pack the per-pixel scale plane into two trailing int8 bands so
one array carries everything the loader needs. That is a storage detail,
and nobody outside this repo should have to reproduce it.

What `geotessera` actually hands you is what `prep` takes: an int8
embedding array and a float scale plane, dequantized as `x = q * scale`.
The Tessera v2 release (`georvq.tessera.v2`) prepares tiles through it.
"""
from __future__ import annotations

import numpy as np
import torch

from georvq.sources.tessera import TesseraSource


def prep(embedding, scale=None, device="cpu", channels_last=None,
         bands=128):
    """geotessera arrays -> the model-space tensors the codec encodes.

    embedding: (bands, H, W) or (B, bands, H, W), int8 codes or already
               dequantized floats. `geotessera` hands out (H, W, bands),
               which is accepted too -- see `channels_last`.
    scale:     (H, W) or (B, H, W) per-pixel dequantization scale.
               Required for int8 input, forbidden for float input.
    channels_last: None auto-detects, which is right whenever the band
               axis is unambiguous. Pass it explicitly for a tile whose
               height or width happens to equal `bands`, where no rule
               could tell the layouts apart.

    Returns (x, valid) exactly as `TesseraSource.prep` would, so anything
    downstream cannot tell which door the data came through. nodata is
    the source's own convention -- an all-zero vector, or a scale that is
    zero, negative, or absurd.
    """
    e = torch.as_tensor(np.asarray(embedding))
    if channels_last is None:
        last, first = e.shape[-1] == bands, e.shape[-3 if e.ndim > 3 else 0] == bands
        if last and first:
            raise ValueError(
                f"cannot tell the layout of {tuple(e.shape)}: both the band "
                f"axis and a spatial axis are {bands}. Pass channels_last=.")
        channels_last = last
    if channels_last:
        e = e.movedim(-1, -3) if e.ndim > 3 else e.movedim(-1, 0)
    if e.ndim == 3:
        e = e[None]
    if e.dtype == torch.int8:
        if scale is None:
            raise ValueError(
                "int8 embeddings need their scale plane: geotessera returns "
                "it alongside the codes, and x = q * scale")
        s = torch.as_tensor(np.asarray(scale)).float()
        if s.ndim == 2:
            s = s[None]
        if s.shape[0] != e.shape[0] or s.shape[-2:] != e.shape[-2:]:
            raise ValueError(
                f"scale {tuple(s.shape)} does not match embedding "
                f"{tuple(e.shape)}: expected one (H, W) plane per tile")
        q = e.to(device).float()
        s = s.to(device)
        valid = (e.to(device).ne(0).any(1) & (s > 0)
                 & (s < TesseraSource.SCALE_MAX))
        x = q * torch.where(valid, s / TesseraSource.XNORM,
                            torch.zeros((), device=s.device)).unsqueeze(1)
        return x, valid
    if scale is not None:
        raise ValueError("float embeddings are already dequantized; "
                         "pass scale only with int8 codes")
    x = e.to(device).float() / TesseraSource.XNORM
    valid = x.ne(0).any(1)
    return x, valid

