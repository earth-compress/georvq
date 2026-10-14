# Bitstream format

## What defines the bitstream

A window's compressed representation (`georvq.inference.api.Packet`) is
fully determined by:

1. the tier specs (`grid:dim:codes:depth` per tier),
2. the codebook tensors (and conditioning priors, if any),
3. the stats side-channel codec (PCA basis), when the source uses one, and
4. the release layer's own decode state, when it has any: the
   transmitted-mean channel's tables, and a refinement's table and
   rotation seed (the `extra` argument of
   `georvq.inference.release.bitstream_id`).

The decoder weights are deliberately excluded. Releases carry
`bitstream_id` = hash(1–4): any future decoder trained against the same
id (bigger trunk, seam deblocker, task-specialised head) decodes
existing packets with no re-encoding, so decode-side
quality upgrades cost nothing. Decoder-*specific* artifacts (e.g. the AE release's
per-window quality column) are keyed by a second hash, `decoder_id`
(the decoder weights), so variant decoders publish alongside each other.

Optional additive refinements extend a packet without changing the
indices: the Tessera release's opt-in refinement carries the fine
tier's post-quantization residual and is added to the fine tokens when
that tier decodes at full depth. Removing or ignoring a refinement
always yields a valid decode. The AE release has no refinement; its
transmitted per-window mean is part of the packet (item 4 above).

## Packet layout

Per window, full depth (truncate levels at decode time for lower rates —
the matryoshka property):

- per tier: `(cells, depth)` code indices,
- optional location embedding (conditioned models; cascade decoding
  replays the priors),
- optional packed stats bytes (normalized sources): `[n:1B][n component
  bytes]`, with two raw escapes: `n=255` = float16, `n=254` = float32.

Nominal cost: `sum_tiers cells * depth_used * log2(codes)` bits
(`Codec.packet_bits`).

## Carriers

`Packet` is the abstract unit; how packets are containerized is a
per-release decision. The AlphaEarth release stores them in one
GeoParquet file per source tile, a column per RVQ level
(`docs/GEORVQ_PARQUET.md`, `docs/AE_RELEASE.md`), and stamps
`bitstream_id` in the file header so mismatched models are refused.

## Entropy coding (not shipped)

Measured on the first AlphaEarth release, order-0 rANS recovered only
~0.6% over deflated raw indices (the fine tier runs at ~95% of codebook
entropy), so the shipped formats store uncoded indices, and the rANS
archive writer has been removed.

Models trained with the context entropy model report their
*achievable* rate (`model.entropy_level_bits`, hard-index -log2 q). The gap
between that and the uncoded indices is the headroom a context-adaptive
coder (driven by the same `ContextRateModel`) would recover; it slots in
behind the same packet layout.
