"""Release identities: `bitstream_id` and `decoder_id`.

`bitstream_id` hashes everything that defines the bitstream (tier specs,
codebook tensors, conditioning priors and any release-layer state passed
as `extra`). It leaves out the decoder weights, so that an improved
decoder can ship against the same id and decode existing archives (see
docs/BITSTREAM.md). `decoder_id` identifies the decoder half. Release
bundles are written and read by georvq.inference.bundle.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def bitstream_id(model, extra=()) -> str:
    """Hash of everything that DEFINES the bitstream.

    Decoder weights are deliberately excluded, so a better decoder ships
    against the same id and decodes existing archives. Everything a
    decoder must agree on to reconstruct the same field is included, and
    `extra` is how a source's release layer adds its own such state --
    a transmitted-mean channel's constants. Leaving those out would let two incompatible bundles
    claim one id, and mismatched-but-same-id state decodes to plausible
    garbage rather than to an error."""
    h = hashlib.sha256()
    for sp in model.specs:
        h.update(str((sp.grid, sp.dim, sp.codes, sp.depth)).encode())
    for rvq in model.rvqs:
        for lev in rvq.levels:
            h.update(lev.codebook.weight.detach().cpu().numpy().tobytes())
    if model.conditioner is not None:      # priors shape the coded residuals
        for p in model.conditioner.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
    for blob in extra:
        h.update(blob if isinstance(blob, bytes) else np.asarray(blob).tobytes())
    return h.hexdigest()[:16]


def decoder_id(model, extra_state=None, recipe=None) -> str:
    """Identity of the decoder half (bitstream_id deliberately excludes
    it): hashes the decoder's parameters. Decoder-specific artifacts
    (e.g. per-window quality rasters) are keyed by this, so upgraded or
    variant decoders publish alongside rather than overwriting.

    Variant decoders pass `extra_state` (the post-processor's state
    dict) and `recipe` (its canonical application constants, a small
    JSON-able dict): both DEFINE the output field, so both are part of
    the identity — changing either mints a new decoder_id."""
    h = hashlib.sha256()
    for sd in (model.decoder.state_dict(),) + \
            ((extra_state,) if extra_state is not None else ()):
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].detach().cpu().numpy().tobytes())
    if recipe is not None:
        h.update(json.dumps(recipe, sort_keys=True).encode())
    return h.hexdigest()[:16]
