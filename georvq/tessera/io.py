"""Getting Tessera data into and out of the codec.

`georvq.sources.tessera` reads the project's own chip store, whose raw
batches pack the per-pixel scale plane into two trailing int8 bands so
one array carries everything the loader needs. That is a storage detail,
and nobody outside this repo should have to reproduce it.

What `geotessera` actually hands you is what these helpers take: an int8
embedding array and a float scale plane, dequantized as `x = q * scale`.
They also do the windowing, since the codec works on 64 px windows and a
tile is much larger than one.

    from georvq.tessera import load, encode_tile, decode_tile
    R = load(BUNDLE, device="cuda")
    packets, grid = encode_tile(R, embedding, scale)      # (128, H, W), (H, W)
    recon = decode_tile(R, packets, grid, depths=[20, 20, 5])
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


def to_windows(x, valid, window=64):
    """(B, bands, H, W) -> (N, bands, w, w) windows plus the grid shape.

    Whole windows only: a tile whose side is not a multiple of `window`
    is cropped at the bottom/right rather than padded, because padding
    would invent data the encoder would then spend rate describing. Crop
    or pad upstream if you need the last partial window.
    """
    B, C, H, W = x.shape
    gh, gw = H // window, W // window
    if gh == 0 or gw == 0:
        raise ValueError(f"tile {H}x{W} is smaller than one {window}px window")
    x = x[:, :, :gh * window, :gw * window]
    valid = valid[:, :gh * window, :gw * window]
    xw = (x.reshape(B, C, gh, window, gw, window)
           .permute(0, 2, 4, 1, 3, 5).reshape(-1, C, window, window))
    vw = (valid.reshape(B, gh, window, gw, window)
               .permute(0, 1, 3, 2, 4).reshape(-1, window, window))
    return xw, vw, (B, gh, gw)


def from_windows(xw, grid, window=64):
    """Inverse of `to_windows`: windows back to (B, bands, H, W)."""
    B, gh, gw = grid
    C = xw.shape[1]
    return (xw.reshape(B, gh, gw, C, window, window)
              .permute(0, 3, 1, 4, 2, 5)
              .reshape(B, C, gh * window, gw * window))


class TileGrid(tuple):
    """The (B, gh, gw) window grid `encode_tile` returns, carrying the
    tile's pixel validity per window as `valid` ((N, w, w) bool), which
    `decode_tile` needs to restore nodata."""

    def __new__(cls, grid, valid=None):
        g = super().__new__(cls, grid)
        g.valid = valid
        return g


def encode_tile(release, embedding, scale=None, batch=64, refine=None,
                channels_last=None):
    """Encode a whole tile. Returns (packets, grid) for `decode_tile`;
    `grid` carries the tile's validity. channels_last: as for `prep`."""
    x, valid = prep(embedding, scale, device=release.codec.device,
                    channels_last=channels_last)
    xw, vw, grid = to_windows(x, valid, release.window)
    packets = []
    for i in range(0, xw.shape[0], batch):
        packets += release.encode(xw[i:i + batch], valid=vw[i:i + batch],
                                  refine=refine)
    return packets, TileGrid(grid, vw)


def decode_tile(release, packets, grid, depths=None, batch=64, valid=None):
    """Decode what `encode_tile` produced, back to (B, bands, H, W).

    Pixels that were nodata come back as zero vectors. `valid` ((N, w, w)
    per window) defaults to the validity `grid` carries; a grid without
    it (a plain tuple) decodes every pixel, nodata ones as the window
    mean."""
    if valid is None:
        valid = getattr(grid, "valid", None)
    outs = []
    for i in range(0, len(packets), batch):
        v = None if valid is None else valid[i:i + batch]
        outs.append(release.decode(packets[i:i + batch], depths=depths,
                                   valid=v))
    return from_windows(torch.cat(outs), grid, release.window)


def requantize(x, scale=None):
    """Model-space floats -> the source's own units.

    Undoes the XNORM rescaling `prep` applied. With `scale`, also returns
    int8 codes on the source's quantization grid, so a reconstruction can
    be written back in the layout it arrived in.
    """
    orig = torch.as_tensor(x).float() * TesseraSource.XNORM
    if scale is None:
        return orig
    s = torch.as_tensor(np.asarray(scale)).float().to(orig.device)
    if s.ndim == 2:
        s = s[None]
    q = torch.round(orig / s.clamp_min(1e-12).unsqueeze(1))
    return orig, q.clamp(-127, 127).to(torch.int8)
