"""The deployment codec API: model in, packets out.

A `Packet` is one window's compressed representation: per-tier code-index
arrays (cells, depth) plus optional side info (location for conditioned
models, packed stats bytes for normalized sources). Any monotone depth
tuple can be decoded from the same packet by truncating levels — the
matryoshka property.

The bitstream is decoder-independent: a packet is defined by everything
BOTH ends must agree on -- tier specs, codebooks, the stats codec or
transmitted-mean channel where a source uses one, and a refinement's
table and seed where one is carried -- and never by the decoder weights,
so decoder upgrades never require re-encoding. `bitstream_id` hashes
exactly that set (see docs/BITSTREAM.md).
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
    stats: Optional[bytes] = None
    # a normalizer's packed side channel (`stats_codec.StatsPacker`), for
    # sources that RESCALE their input. Not to be confused with `mean`
    # below, which is part of the reconstruction rather than a rescaling
    # of it -- the two are different mechanisms and never both in use.
    mean: Optional[np.ndarray] = None
    # (bands,) per-window mean, for models trained with `stats_cond`. It
    # is subtracted before the encoder and added back after the decoder,
    # so unlike `stats` (a normalizer's packed side channel) it is part
    # of the reconstruction, not a rescaling of it. Carried as floats
    # here; a release quantizes it on the way to disk and charges it.
    refinements: Optional[dict] = None
    # {tier: (cells, dim) float} — that tier's residual after its full RVQ
    # depth. Optional additive refinements: a decoder adds one to its
    # tier's tokens when that tier is decoded at full depth, buying back
    # the quantizer gap the codebooks left behind. Which tier is worth
    # refining differs by model (see georvq.inference.refine); dropping a
    # refinement always yields a valid decode.

    @property
    def coarse_residual(self):
        """The tier-0 refinement, under the name the AlphaEarth release
        gave it ('8*'). Retained because that release is frozen."""
        return (self.refinements or {}).get(0)

    @coarse_residual.setter
    def coarse_residual(self, v):
        if self.refinements is None:
            self.refinements = {}
        if v is None:
            self.refinements.pop(0, None)
        else:
            self.refinements[0] = v


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
    def encode(self, x, loc=None, refine_tiers=(), valid=None, mean=None,
               coarse_residual=False) -> list[Packet]:
        """x: (B, C, W, W) prepared model-space input -> one Packet per
        window, at full depth (truncate at decode time for lower rates).

        refine_tiers: tier indices whose post-RVQ residual to carry as an
        additive refinement (see georvq.inference.refine). Which tier is
        worth refining is model-specific. `coarse_residual=True` is the
        AlphaEarth release's name for `refine_tiers=(0,)`.

        valid/mean: for `stats_cond` models the per-window mean is part
        of the packet. It is measured from the input unless supplied —
        pass `mean` explicitly to encode against the QUANTIZED mean a
        release will actually transmit, so encoder and decoder agree.
        """
        m = self.model
        if m.stats_cond is not None and mean is None:
            # (B, bands) mean, or [mean ‖ std] for a use_std model
            mean = m.patch_stats(x, valid)
        zs = m.encode_levels(x, stats=mean, valid=valid)
        B = x.shape[0]
        want = set(refine_tiers) | ({0} if coarse_residual else set())
        if want and (m.selector is not None or m.conditioner is not None):
            raise NotImplementedError(
                "refinements are supported for plain codecs only")
        res = {}
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
            for t, (z, rvq) in enumerate(zip(zs, m.rvqs)):
                residual = z.float()
                per = []
                for lev in rvq.levels:
                    q, idx, _ = lev(residual)
                    residual = residual - q
                    per.append(idx)
                idxs.append(per)
                if t in want:
                    res[t] = residual.cpu().numpy()
        # idxs: per tier, list of depth (B, cells) tensors -> (B, cells, depth)
        stacked = [torch.stack(per, -1).cpu().numpy() for per in idxs]
        loc_np = (loc.cpu().numpy() if isinstance(loc, torch.Tensor)
                  else loc) if loc is not None else None
        mu_np = mean.cpu().numpy() if isinstance(mean, torch.Tensor) else mean
        return [Packet(indices=[t[b] for t in stacked],
                       loc=loc_np[b] if loc_np is not None else None,
                       mean=mu_np[b] if mu_np is not None else None,
                       refinements=({t: r[b] for t, r in res.items()}
                                    if res else None))
                for b in range(B)]

    # ---- decode ----

    @torch.no_grad()
    def decode(self, packets: list[Packet], depths=None, valid=None):
        """Decode packets at a depth tuple (default: full).
        Returns (B, C, W, W) model-space reconstructions."""
        m = self.model
        depths = depths or [sp.depth for sp in self.specs]
        B = len(packets)
        dev = self.device
        mean = None
        if packets[0].mean is not None:
            mean = torch.from_numpy(
                np.stack([p.mean for p in packets])).to(dev).float()
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
            return m.decode(zqs, Ls, stats=mean, valid=valid)
        for rvq, tier_idx, L in zip(m.rvqs, idx, depths):
            zqs.append(self._lookup(rvq, tier_idx, L))
        # additive refinements: applied per tier, and only where that tier
        # is at full depth — a refinement codes the residual left after
        # ALL of its tier's levels, so it is meaningless against a prefix
        for t in sorted(packets[0].refinements or {}):
            if depths[t] != self.specs[t].depth:
                continue
            r = torch.from_numpy(
                np.stack([p.refinements[t] for p in packets])).to(dev)
            zqs[t] = zqs[t] + r
        Ls = torch.tensor(depths, device=dev)[None].expand(B, -1)
        return m.decode(zqs, Ls, stats=mean, valid=valid)

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

    def packet_bits(self, depths=None, refinements=()) -> float:
        """Nominal (uncoded) bits per window at a depth tuple.

        Counts EVERYTHING the packet carries, not just the index stream:
        the stats side channel is charged at every depth because it is
        always transmitted, and any refinement is charged where its tier
        is at full depth. Omitting the side channel is not a rounding
        error — on a stats_cond model at its cheapest rung it is 83% of
        the true bitstream.
        """
        from georvq.model.codec import tuple_bits
        depths = depths or [sp.depth for sp in self.specs]
        b = tuple_bits(self.specs, depths, self.model.side_bits())
        for spec in refinements:
            if depths[spec.tier] == self.specs[spec.tier].depth:
                sp = self.specs[spec.tier]
                b += spec.cost_bits(sp.cells, sp.dim)
        return float(b)

    @torch.no_grad()
    def entropy_bits(self, x) -> Optional[torch.Tensor]:
        """Entropy-model bits per window (B, total-levels summed); None if
        the model has no rate model. Real coded sizes come from
        georvq.inference.archive."""
        if self.model.rate is None:
            return None
        per_tier = self.model.entropy_level_bits(x)
        return torch.cat(per_tier, -1).sum(-1)
