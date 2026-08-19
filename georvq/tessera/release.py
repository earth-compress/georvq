"""Tessera release policy: the ladder, the mean channel, the refinement.

The pieces here are choices about *this* model and *this* source. The
mechanism each one configures is generic and lives in
`georvq.inference`.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from georvq.inference.api import Codec, Packet
from georvq.inference.mean_codec import MeanChannel
from georvq.inference.refine import RefinementSpec
from georvq.inference import refine as _refine
from georvq.inference.release import bitstream_id, decoder_id

# --------------------------------------------------------------------------
# the refinement: which tier, and why this one
# --------------------------------------------------------------------------

#: Tessera's recoverable capacity sits in the FINE tier. Measured on the
#: release checkpoint (768 held-out windows, share of the ceiling's
#: centred-cosine gap recovered, against 10.85 b/px base):
#:
#:     tier 0, 1-2 bits   +0.13-0.25 b/px     0.0%
#:     tier 1, 1-2 bits   +2.00-4.00 b/px     1.9%
#:     tier 2, 1 bit      +16.0 b/px         56.2%
#:     tier 2, 2 bits     +32.0 b/px         81.2%
#:
#: The coarse tier returns *nothing* — the AlphaEarth release refines
#: exactly that tier, and transplanting its policy here would spend rate
#: for no measurable gain. Which tier holds a model's slack has to be
#: measured per model; it is not a property of the architecture.
REFINEMENT = RefinementSpec(tier=2, bits=1, quantizer="lloyd",
                            rotate_seed=0)

#: ...and it is OFF by default, which is the other way round from
#: AlphaEarth. There the sidecar costs +1.5% of rate and removes the
#: error tail, so it is free enough to always apply. Here it costs +16
#: b/px — it more than doubles the bitstream to recover 56% of a gap that
#: the downstream benchmarks cannot resolve at all. It is a
#: maximum-fidelity option, not a default.
REFINE_BY_DEFAULT = False


# --------------------------------------------------------------------------
# the rate ladder
# --------------------------------------------------------------------------

def ladder(specs, side_bits):
    """The efficient rate ladder: saturate coarse, then mid, then add fine
    levels one at a time.

    The level costs are wildly asymmetric — with cells 1/16/256 and 8-bit
    codes, one level costs 8 / 128 / 2,048 bits — so a level of the
    coarse tier is 256x cheaper than a level of the fine tier and there
    is never a reason to buy the expensive one first. Hand-picked
    "balanced" tuples are strictly dominated.
    """
    out = [(20, 0, 0), (20, 20, 0)]
    out += [(20, 20, k) for k in range(1, specs[2].depth + 1)]
    return [t for t in out if t[0] <= specs[0].depth]


def tuple_bits(specs, depths, side_bits, refine: RefinementSpec = None):
    b = sum(sp.cells * L * math.log2(sp.codes)
            for sp, L in zip(specs, depths)) + side_bits
    if refine is not None and depths[refine.tier] == specs[refine.tier].depth:
        sp = specs[refine.tier]
        b += refine.cost_bits(sp.cells, sp.dim)
    return float(b)


# --------------------------------------------------------------------------
# the release object
# --------------------------------------------------------------------------

class TesseraRelease:
    """A loaded Tessera release: codec + mean channel + refinement.

    The mean channel is the part that is easy to get wrong. The model was
    trained on a field with its mean removed and is conditioned on that
    mean, so encoder and decoder must agree on it *after* quantization —
    encoding against the float mean and decoding against the transmitted
    one would leave a mismatch the model never saw in training.
    `encode` therefore rounds the mean through the channel first and
    conditions on the rounded vector.
    """

    def __init__(self, codec: Codec, mean_channel: MeanChannel,
                 refine: RefinementSpec = None, refine_table=None,
                 bundle: dict = None):
        self.codec = codec
        self.model = codec.model
        self.specs = codec.specs
        self.window = codec.window
        self.mean_channel = mean_channel
        self.refine = refine
        self.refine_table = refine_table
        self.bundle = bundle or {}

    # ---- rate ----

    @property
    def side_bits(self) -> float:
        return float(self.mean_channel.cost_bits())

    def ladder(self):
        return ladder(self.specs, self.side_bits)

    def bits(self, depths=None, refine=False) -> float:
        depths = depths or [sp.depth for sp in self.specs]
        return tuple_bits(self.specs, depths, self.side_bits,
                          self.refine if refine else None)

    def bpp(self, depths=None, refine=False) -> float:
        return self.bits(depths, refine) / self.window ** 2

    # ---- code ----

    @torch.no_grad()
    def encode(self, x, valid=None, refine=None) -> list[Packet]:
        """x: (B, bands, W, W) prepared model-space input."""
        refine = REFINE_BY_DEFAULT if refine is None else refine
        mu = self.model.patch_mean(x, valid).cpu().numpy()
        codes = self.mean_channel.encode(mu)
        mu_q = self.mean_channel.decode(codes)       # what the decoder sees
        mu_t = torch.from_numpy(mu_q).to(x.device)
        tiers = (self.refine.tier,) if (refine and self.refine) else ()
        pk = self.codec.encode(x, valid=valid, mean=mu_t, refine_tiers=tiers)
        for p, c in zip(pk, codes):
            p.stats = c.astype(np.uint16).tobytes()   # the wire form
        if refine and self.refine is not None:
            t = self.refine.tier
            res = np.stack([p.refinements[t] for p in pk])
            q = _refine.unpack(*_refine.pack(res, self.refine,
                                             self.refine_table),
                               self.refine, self.specs[t].cells,
                               self.specs[t].dim, self.refine_table)
            for j, p in enumerate(pk):
                p.refinements = {t: q[j]}             # quantized, as sent
        return pk

    @torch.no_grad()
    def decode(self, packets, depths=None, valid=None):
        return self.codec.decode(packets, depths=depths, valid=valid)


# --------------------------------------------------------------------------
# building and loading a bundle
# --------------------------------------------------------------------------

def build_bundle(out_dir, model, cfg, means, refine_residuals=None,
                 step=0, notes=""):
    """Write a release bundle: weights, mean channel, refinement table.

    means:             (N, bands) window means from the training corpus
    refine_residuals:  (N, cells, dim) residuals of REFINEMENT.tier, from
                       held-out windows encoded with the SHIPPED weights
    """
    from georvq.repro import save_checkpoint
    os.makedirs(out_dir, exist_ok=True)
    save_checkpoint(os.path.join(out_dir, "model.pt"), model, cfg, step)

    bits = cfg.model.stats_cond.bits_per_band
    ch = MeanChannel.fit(means, bits)
    ch.save(os.path.join(out_dir, "mean_channel.npz"))

    table = None
    if refine_residuals is not None:
        table = _refine.fit_lloyd(refine_residuals, REFINEMENT)
        np.savez(os.path.join(out_dir, "refine_table.npz"), table=table,
                 **{"spec_" + k: v for k, v in REFINEMENT.to_json().items()
                    if v is not None})

    bid = bitstream_id(model)
    meta = {
        "bitstream_id": bid,
        "decoder_id": decoder_id(model),
        "decoder_version": 1,
        "source": "tessera",
        "window": model.window,
        "bands": model.bands,
        "tiers": [str(sp) for sp in model.specs],
        "mean_channel": {"bits_per_band": bits,
                         "bits_per_window": ch.cost_bits()},
        "refinement": (REFINEMENT.to_json() if table is not None else None),
        "refine_by_default": REFINE_BY_DEFAULT,
        "notes": notes,
    }
    with open(os.path.join(out_dir, "bundle.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load(release_dir, device="cpu") -> TesseraRelease:
    from georvq.factory import from_checkpoint
    model = from_checkpoint(os.path.join(release_dir, "model.pt"), device)
    ch = MeanChannel.load(os.path.join(release_dir, "mean_channel.npz"))
    spec, table = None, None
    rpath = os.path.join(release_dir, "refine_table.npz")
    if os.path.exists(rpath):
        z = np.load(rpath)
        table, spec = z["table"], REFINEMENT
    bundle = {}
    bpath = os.path.join(release_dir, "bundle.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            bundle = json.load(f)
    return TesseraRelease(Codec(model), ch, spec, table, bundle)


LADDER = ladder
