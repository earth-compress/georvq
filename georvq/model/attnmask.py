"""Hierarchical local attention masks for the pyramid trunks.

Per-tier locality radii (in that tier's own cells; None = global):
queries and keys from radius-None tiers (coarse/mid) attend/are attended
globally; between two spatial tiers, a pair is allowed iff the Chebyshev
distance between cell centers (in pixels) is within the larger of the
two tiers' pixel radii. This encodes e.g.
    local_attn: [null, null, 2, 4]
= coarse->all, mid->all, fine->(coarse+mid+5x5 fine+local ultrafine),
ultrafine->(coarse+mid+local fine+9x9 ultrafine) — the structure trained
full-attention models converge to (see the AE attention probe): sharp
local neighborhoods plus a low-rank global read supplied by the coarse
tiers.

Returned mask is bool (N, N), True = attention DISALLOWED (the
nn.MultiheadAttention convention).
"""
from __future__ import annotations

import torch


def build_local_mask(specs, window, radii):
    """specs: tier specs (grid, cells); window: px; radii: per-tier cell
    radius or None (global). Returns bool (N, N), True = masked."""
    assert len(radii) == len(specs), "one radius (or null) per tier"
    # per-token spatial center in px, or None for global-tier tokens
    centers, r_px, tier_of = [], [], []
    for ti, (sp, r) in enumerate(zip(specs, radii)):
        cell = window / sp.grid
        for c in range(sp.cells):
            rr, cc = divmod(c, sp.grid)
            centers.append(((rr + 0.5) * cell, (cc + 0.5) * cell))
            r_px.append(None if r is None else r * cell)
            tier_of.append(ti)
    pos = torch.tensor(centers)
    d = (pos[:, None] - pos[None, :]).abs().amax(-1)      # (N, N) chebyshev px
    # a pair is allowed within max(query_r, key_r) px; any global tier on
    # either side (inf radius) keeps the pair visible
    rpx = torch.tensor([float("inf") if r is None else r for r in r_px])
    lim = torch.maximum(rpx[:, None], rpx[None, :])        # (N, N)
    return d > lim
