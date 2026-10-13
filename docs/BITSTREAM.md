# Bitstream and archive format

## What defines the bitstream

A window's compressed representation (`georvq.inference.api.Packet`) is
fully determined by:

1. the tier specs (`grid:dim:codes:depth` per tier),
2. the codebook tensors (and conditioning priors, if any),
3. the stats side-channel codec (PCA basis), when the source uses one, and
4. the CDF tables, when an entropy-coded archive is used (none shipped;
   see below).

The decoder weights are deliberately excluded. Releases carry
`bitstream_id` = hash(1–4): any future decoder trained against the same
id (bigger trunk, seam deblocker, task-specialised head) decodes
existing packets and archives with no re-encoding, so decode-side
quality upgrades cost nothing. Decoder-*specific* artifacts (e.g. the AE release's
per-window quality rasters) are keyed by a second hash, `decoder_id`
(the decoder weights), so variant decoders publish alongside each other.

Optional additive refinements extend a packet without changing the
indices: the AE release's 8* sidecar carries the coarse tier's
post-quantization residual (2-bit scalar-quantized, per-window scale)
and is added to the coarse tokens whenever that tier decodes at full
depth. Removing or ignoring a refinement always yields a valid decode.

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
per-release decision. The AlphaEarth release stores them as
georeferenced COG assets (bands = RVQ levels — see `docs/AE_RELEASE.md`);
the patch dev tool uses a flat npz. Both stamp `bitstream_id` so
mismatched models are refused.

## Entropy-coded archives (research capability, not shipped)

Measured on the AlphaEarth release, order-0 rANS recovers only ~0.6%
over deflated raw indices (the fine tier runs at ~95% of codebook
entropy), so the shipped format stores uncoded indices. The machinery
remains for sources and coders which benefit from using it:

`georvq.inference.archive` writes real files: rANS (compressai backend)
over the flattened symbols (tier-major, level-major, cells within level)
with one calibrated, Laplace-smoothed CDF per (tier, level) at 16-bit
precision. Every bitstream is round-trip verified during writing;
`metadata.json` records the realized bytes against the naive bits, so
reported sizes are measured rather than estimated.

Models trained with the context entropy model additionally report their
*achievable* rate (`model.entropy_level_bits`, hard-index -log2 q). The gap
between that and the static-CDF archive is the headroom a context-adaptive
coder (driven by the same `ContextRateModel`) would recover; it slots in
behind the same packet layout.
