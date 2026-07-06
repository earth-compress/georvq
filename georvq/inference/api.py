"""The deployment codec API: model in, packets out.

A `Packet` is one window's compressed representation: per-tier code-index
arrays (cells, depth) plus optional side info (location for conditioned
models, packed stats bytes for normalized sources). Any monotone depth
tuple can be decoded from the same packet by truncating levels — the
matryoshka property.

The bitstream is decoder-independent: packets are defined by (tier specs,
codebooks, stats codec), so decoder upgrades never require re-encoding
(see docs/BITSTREAM.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class Packet:
    indices: list  # per tier: (cells, depth) int64 ndarray
    loc: Optional[np.ndarray] = None      # location embedding (conditioned models)
    stats: Optional[bytes] = None         # packed normalization stats


def load_codec(ckpt_path, device="cpu"):
    """Model from a checkpoint (either format), wrapped in a Codec."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" in ck:
        from georvq.factory import from_checkpoint
        model = from_checkpoint(ckpt_path, device)
    else:
        from georvq.compat import from_legacy_checkpoint
        model, _ = from_legacy_checkpoint(ckpt_path, device)
    return Codec(model)


class Codec:
    """Encode/decode against a trained PyramidCodec."""

    def __init__(self, model, stats_packer=None):
        self.model = model.eval()
        self.specs = model.specs
        self.window = model.window
        self.stats_packer = stats_packer
        self.device = next(model.parameters()).device

    # ---- encode ----

    @torch.no_grad()
    def encode(self, x, loc=None) -> list[Packet]:
        """x: (B, C, W, W) prepared model-space input -> one Packet per
        window, at full depth (truncate at decode time for lower rates)."""
        m = self.model
        zs = m.encode_levels(x)
        B = x.shape[0]
        if m.selector is not None:
            scores = m.selector(loc)
            idxs = [m.rvqs[i].all_levels_gated(z.float(), scores[i],
                                               m.selector.subset_k(sp))[1]
                    for i, (z, sp) in enumerate(zip(zs, self.specs))]
        elif m.conditioner is not None:
            idxs = m.conditioner.full_depth_indices(
                m.rvqs, self.specs, [z.float() for z in zs], loc)
        else:
            idxs = []
            for z, rvq in zip(zs, m.rvqs):
                residual = z.float()
                per = []
                for lev in rvq.levels:
                    q, idx, _ = lev(residual)
                    residual = residual - q
                    per.append(idx)
                idxs.append(per)
        # idxs: per tier, list of depth (B, cells) tensors -> (B, cells, depth)
        stacked = [torch.stack(per, -1).cpu().numpy() for per in idxs]
        loc_np = (loc.cpu().numpy() if isinstance(loc, torch.Tensor)
                  else loc) if loc is not None else None
        return [Packet(indices=[t[b] for t in stacked],
                       loc=loc_np[b] if loc_np is not None else None)
                for b in range(B)]

    # ---- decode ----

    @torch.no_grad()
    def decode(self, packets: list[Packet], depths=None):
        """Decode packets at a depth tuple (default: full).
        Returns (B, C, W, W) model-space reconstructions."""
        m = self.model
        depths = depths or [sp.depth for sp in self.specs]
        B = len(packets)
        dev = self.device
        idx = [torch.from_numpy(np.stack([p.indices[i] for p in packets]))
               .to(dev) for i in range(len(self.specs))]     # (B, cells, depth)
        loc = None
        if packets[0].loc is not None:
            loc = torch.from_numpy(np.stack([p.loc for p in packets])).to(dev)

        zqs = []
        if m.conditioner is not None:
            recons, tier_depths = [], []
            full = [sp.depth for sp in self.specs]
            for i, (rvq, sp, L) in enumerate(zip(m.rvqs, self.specs, depths)):
                g = self._prior(m, i, recons, tier_depths, loc, B)
                zq_full = self._lookup(rvq, idx[i], sp.depth) + g
                zq = self._lookup(rvq, idx[i], L) + g
                zqs.append(zq)
                recons.append(zq_full)   # full-depth context for finer tiers
                tier_depths.append(torch.full((B,), full[i], device=dev))
            Ls = torch.tensor(depths, device=dev)[None].expand(B, -1)
            return m.decode(zqs, Ls)
        for rvq, tier_idx, L in zip(m.rvqs, idx, depths):
            zqs.append(self._lookup(rvq, tier_idx, L))
        Ls = torch.tensor(depths, device=dev)[None].expand(B, -1)
        return m.decode(zqs, Ls)

    @staticmethod
    def _lookup(rvq, tier_idx, L):
        """(B, cells, depth) indices -> summed codebook vectors at depth L."""
        zq = 0
        for l in range(L):
            zq = zq + rvq.levels[l].codebook(tier_idx[..., l])
        if isinstance(zq, int):     # L == 0
            lev = rvq.levels[0]
            return torch.zeros(*tier_idx.shape[:2], lev.dim,
                               device=tier_idx.device)
        return zq

    @staticmethod
    def _prior(m, i, recons, tier_depths, loc, B):
        if i == 0:
            if m.conditioner.coarse_prior is not None and loc is not None:
                return m.conditioner.coarse_prior(loc).unsqueeze(1)
            return 0.0
        return m.conditioner._predict(i, recons, tier_depths, loc, B)

    # ---- rate accounting ----

    def packet_bits(self, depths=None) -> float:
        """Nominal (uncoded) bits per window at a depth tuple."""
        depths = depths or [sp.depth for sp in self.specs]
        return float(sum(sp.bits(L) for sp, L in zip(self.specs, depths)))

    @torch.no_grad()
    def entropy_bits(self, x) -> Optional[torch.Tensor]:
        """Entropy-model bits per window (B, total-levels summed); None if
        the model has no rate model. Real coded sizes come from
        georvq.inference.archive."""
        if self.model.rate is None:
            return None
        per_tier = self.model.entropy_level_bits(x)
        return torch.cat(per_tier, -1).sum(-1)
