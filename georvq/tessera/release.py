"""Tessera release policy: the ladder, the mean channel, the refinement.

The pieces here are choices about *this* model and *this* source. The
mechanism each one configures is generic and lives in
`georvq.inference`.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from georvq.inference.api import Codec, Packet
from georvq.inference.mean_codec import MeanChannel
from georvq.inference.refine import RefinementSpec
from georvq.inference import refine as _refine
from georvq.inference.release import bitstream_id, decoder_id
from georvq.model.codec import tuple_bits as _codec_tuple_bits

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
    """Rate of one depth tuple, including the side channel and, where its
    tier is at full depth, the refinement.

    The pyramid + side-channel arithmetic belongs to
    `georvq.model.codec.tuple_bits` and is deliberately not repeated
    here: a second copy of a rate formula is how a channel ends up
    uncharged in one code path and charged in another.
    """
    b = _codec_tuple_bits(specs, depths, side_bits)
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
        """Cost of the transmitted mean, charged at every rung."""
        return float(self.mean_channel.cost_bits())

    def ladder(self):
        return ladder(self.specs, self.side_bits)

    def bits(self, depths=None, refine=False) -> float:
        """Bits per window at a depth tuple, side channel included and the
        refinement counted only when asked for and only at full depth.
        """
        depths = depths or [sp.depth for sp in self.specs]
        return tuple_bits(self.specs, depths, self.side_bits,
                          self.refine if refine else None)

    def bpp(self, depths=None, refine=False) -> float:
        """`bits` per pixel -- the number quoted in docs/TESSERA_RELEASE.md."""
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
        # The packet carries the DEQUANTIZED mean, not the codes. They are
        # not both needed: the mean already sits on the quantizer's
        # reconstruction grid, so `mean_channel.encode(p.mean)` returns the
        # transmitted codes exactly. Carrying both would be two fields for
        # one fact, free to disagree.
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
        """Decode packets at a depth tuple. `valid` restores the source's
        nodata geometry; omit it only when every pixel is valid.
        """
        return self.codec.decode(packets, depths=depths, valid=valid)


# --------------------------------------------------------------------------
# building and loading a bundle
# --------------------------------------------------------------------------

def _id_state(ch, table, spec):
    """The release state a decoder must agree on beyond the weights: the
    mean channel's constants and, when present, the refinement table and
    its spec. Written into `bitstream_id` by `build_bundle` and checked
    against it by `load`, so both go through here."""
    extra = [ch.mu0, ch.scale, np.int32(ch.bits), np.float32(ch.clip)]
    if table is not None:
        extra += [table, json.dumps(spec.to_json(), sort_keys=True).encode()]
    return extra


def build_bundle(out_dir, model, cfg, means, refine_residuals=None,
                 step=0, notes="", fp16=False):
    """Write a release bundle: weights, mean channel, refinement table.

    means:             (N, bands) window means from the training corpus
    refine_residuals:  (N, cells, dim) residuals of REFINEMENT.tier, from
                       held-out windows encoded with the SHIPPED weights
    """
    from georvq.inference.release import save_release
    os.makedirs(out_dir, exist_ok=True)

    bits = cfg.model.stats_cond.bits_per_band
    ch = MeanChannel.fit(means, bits)
    ch.save(os.path.join(out_dir, "mean_channel.npz"))

    table = None
    if refine_residuals is not None:
        table = _refine.fit_lloyd(refine_residuals, REFINEMENT)
        np.savez(os.path.join(out_dir, "refine_table.npz"), table=table,
                 **{"spec_" + k: v for k, v in REFINEMENT.to_json().items()
                    if v is not None})

    bid = save_release(
        out_dir, model, cfg, step=step, notes=notes,
        extra_id=_id_state(ch, table, REFINEMENT),
        fp16=fp16,
        meta={"source": "tessera",
              "bands": model.bands,
              "mean_channel": {"bits_per_band": bits,
                               "bits_per_window": ch.cost_bits()},
              "refinement": (REFINEMENT.to_json() if table is not None
                             else None),
              "refine_by_default": REFINE_BY_DEFAULT,
              "weight_dtype": "float16" if fp16 else "float32"})
    with open(os.path.join(out_dir, "bundle.json")) as f:
        return json.load(f)


def load(release_dir, device="cpu") -> TesseraRelease:
    """Load a Tessera release bundle into a ready `TesseraRelease`.

    The mean channel is required -- the model was trained on a
    mean-removed field, so weights alone cannot decode. The refinement
    table is optional and absent bundles simply cannot refine, which is
    the intended degraded mode rather than an error.
    """
    from georvq.factory import from_checkpoint
    model = from_checkpoint(os.path.join(release_dir, "model.pt"), device)
    ch = MeanChannel.load(os.path.join(release_dir, "mean_channel.npz"))
    bundle = {}
    bpath = os.path.join(release_dir, "bundle.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            bundle = json.load(f)
    spec, table = None, None
    rpath = os.path.join(release_dir, "refine_table.npz")
    if os.path.exists(rpath):
        table = np.load(rpath)["table"]
        # the bundle's OWN spec, not today's module constant: a table is
        # fitted for one (tier, bits, rotation) and decoding it under a
        # different one is silent garbage. REFINEMENT is only the
        # fallback for bundles cut before it was recorded.
        spec = (RefinementSpec.from_json(bundle["refinement"])
                if bundle.get("refinement") else REFINEMENT)
    want = bundle.get("bitstream_id")
    if want:
        # same layout with different codebooks, mean constants or table
        # decodes to plausible garbage rather than failing, so a bundle
        # whose files disagree with its own id is refused here
        got = bitstream_id(model, extra=_id_state(ch, table, spec))
        if got != want:
            raise ValueError(
                f"{release_dir}: bundle.json declares bitstream "
                f"{want} but its files hash to {got}; the weights, "
                "mean_channel.npz or refine_table.npz do not belong to "
                "this bundle")
    return TesseraRelease(Codec(model), ch, spec, table, bundle)


LADDER = ladder
