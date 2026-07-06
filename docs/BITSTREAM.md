# Bitstream and archive format

## What defines the bitstream

A window's compressed representation (`georvq.inference.api.Packet`) is
fully determined by:

1. the tier specs (`grid:dim:codes:depth` per tier),
2. the codebook tensors (and conditioning priors, if any),
3. the stats side-channel codec (PCA basis), and
4. the CDF tables (for entropy-coded archives).

The **decoder weights are deliberately excluded**. `bundle.json` in a
release carries `bitstream_id` = hash(1–4): any future decoder trained
against the same id (bigger trunk, seam deblocker, task-specialised head)
decodes existing packets and archives with zero re-encoding. Decode-side
quality upgrades are free.

A post-decode enhancement module (e.g. a boundary deblocker for window-seam
artifacts) slots into the bundle as an optional extra without touching the
format.

## Packet layout

Per window, full depth (truncate levels at decode time for lower rates —
the matryoshka property):

- per tier: `(cells, depth)` code indices,
- optional location embedding (conditioned models; cascade decoding
  replays the priors),
- optional packed stats bytes (normalized sources): `[n:1B][n component
  bytes]`, `n=255` = raw-f16 fallback.

Nominal cost: `sum_tiers cells * depth_used * log2(codes)` bits
(`Codec.packet_bits`).

## Entropy-coded archives

`georvq.inference.archive` writes real files: rANS (compressai backend)
over the flattened symbols (tier-major, level-major, cells within level)
with one calibrated, Laplace-smoothed CDF per (tier, level) at 16-bit
precision. Every bitstream is round-trip verified during writing;
`metadata.json` records realized bytes vs the naive bits — realized sizes,
not model estimates.

Models trained with the context entropy model additionally report their
*achievable* rate (`model.entropy_level_bits`, hard-index -log2 q). The gap
between that and the static-CDF archive is the headroom a context-adaptive
coder (driven by the same `ContextRateModel`) would recover; it slots in
behind the same packet layout.
