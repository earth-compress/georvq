"""Validity-masked reconstruction metrics for dense embedding fields."""
from __future__ import annotations

import torch


def masked_recon_stats(out, x, valid):
    """out, x: (B,C,t,t) unit-norm fields; valid: (B,t,t) bool.
    Returns (mse_per_unit_pixel, cosine) over valid pixels."""
    mse = ((out - x) ** 2).sum(1)[valid].mean()
    cos = (out * x).sum(1)[valid].mean()
    return mse, cos


def centered_cos(out, x, valid, eps=1e-8):
    """Spatial-structure probe: per-pixel cosine after subtracting each
    token's own valid-pixel mean from both sides. A constant (block-mean)
    reconstruction scores ~0 here regardless of its plain cosine."""
    vm = valid.unsqueeze(1).float()
    n = vm.sum((2, 3)).clamp_min(1)
    xc = x - ((x * vm).sum((2, 3)) / n)[:, :, None, None]
    oc = out - ((out * vm).sum((2, 3)) / n)[:, :, None, None]
    num = (xc * oc).sum(1)
    den = xc.norm(dim=1) * oc.norm(dim=1) + eps
    return (num / den)[valid].mean()
