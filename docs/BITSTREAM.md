# Bitstream format

This document defines what a georvq bitstream depends on and what a
packet contains. It is for anyone building a decoder, a release layer or
a container for a trained codec, who needs to know which changes keep
existing files readable.

## What defines the bitstream

A window's compressed representation (`georvq.inference.api.Packet`) is
fully determined by:

1. the tier specs (`grid:dim:codes:depth` per tier),
2. the codebook tensors (and conditioning priors, if any), and
3. the release layer's own decode state, when it has any: the
   transmitted-mean channel's tables, and a refinement's table and
   rotation seed (the `extra` argument of
   `georvq.inference.release.bitstream_id`).

The decoder weights are deliberately excluded. A release carries
`bitstream_id`, a hash of items 1-3, and any future decoder trained
against the same id (for example a bigger trunk, a seam deblocker or a
task-specialised head) decodes existing packets without re-encoding
them, so an improvement on the decode side needs no change to files
already written. Some published data is tied to one decoder, such as the
AE release's per-window quality column, which was measured with a
particular decoder. That data is keyed by a second hash, `decoder_id`,
taken over the decoder weights, so that variant decoders can publish
their own versions of it alongside each other.

Optional additive refinements extend a packet without changing the
indices. A refinement carries a tier's post-quantisation residual,
which is added to that tier's tokens when it decodes at full depth, and
removing or ignoring a refinement always gives a valid decode. Neither
the AE nor the Tessera release has a refinement. Their transmitted
per-window mean is part of the packet (item 3 above) rather than a
refinement, because the encoder codes the window with its mean removed,
so the indices cannot be decoded without it.

## Packet layout

A packet holds one window at full depth, and lower rates are decoded by
truncating levels at decode time (the matryoshka property). It contains:

- per tier, `(cells, depth)` code indices,
- an optional location embedding, for conditioned models (cascade
  decoding replays the priors),
- an optional per-window mean, for `stats_cond` models, carried as
  floats and quantised by the release on the way to disk,
- optional refinements, as described above.

Sources with a per-tile stats normaliser, such as S2 and DEM, charge
their stats in the rate figures (`side_bits`), but `Packet` does not
carry the stats. The S2 v1 release sends them in each window's base
record (`georvq.s2.v1.stats_channel`). The byte format S2's
stats are sized for is that of `georvq.inference.stats_codec.StatsPacker`,
`[n:1B][n component bytes]`, with two raw escapes, `n=255` for float16
and `n=254` for float32.

Nominal cost: `sum_tiers cells * depth_used * log2(codes)` bits
(`Codec.packet_bits`).

## Containers

`Packet` is an abstract unit, and each release decides how packets are
stored on disk and where the ids are recorded. The AlphaEarth release
stores them in one Zarr store per source tile, with an inner chunk per
RVQ level (`docs/GEORVQ_ZARR.md`, `docs/AE_RELEASE.md`), and records
`bitstream_id` in the store's header so that a mismatched model is refused.

## Entropy coding (not shipped)

Measured on the first AlphaEarth release, order-0 rANS recovered only
~0.6% over deflated raw indices, because the fine tier already ran at
~95% of codebook entropy. The shipped formats therefore store uncoded
indices, and georvq has no entropy coder.

Models trained with the context entropy model report their achievable
rate (`model.entropy_level_bits`, hard-index -log2 q). The gap between
that rate and the uncoded indices is the headroom a context-adaptive
coder, driven by the same `ContextRateModel`, would recover. Such a coder
would change the bytes a release stores, but not what a packet contains.
