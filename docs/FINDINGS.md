# Empirical findings behind the design

Distilled results from the research cycles that produced this codebase.
They explain why the architecture is shaped the way it is; treat them as
strong priors, not laws.

## Architecture

- **ViT tokenizer + attention decoder beat conv trunks/FPN decisively** on
  dense embedding fields. Conv patch-embed *stems* remain (they are the
  per-source plug point); the deprecated part is conv feature mixing.
- **The decoder is usually the bottleneck, not the latent.** With a frozen
  encoder + bitstream, swapping in a larger transformer decoder gained
  +1–3 dB at identical rate. Hence the decoder-independent bitstream
  contract (docs/BITSTREAM.md).
- **Matryoshka min-depths matter**: training with per-tier minimum depths
  (e.g. 6-1-0) concentrates capacity on the operating points that get
  used; untrained tuples mislead evaluation.

## Quantization

- **VQ state must stay fp32** under bf16 autocast — assignment scores may
  run bf16 (boundary flips only), but residuals, EMA buffers and codebooks
  must not.
- Codebook health machinery (k-means/data init, EMA decay ramp, dead-code
  reset with traffic-scaled thresholds, commit loss) is what keeps large
  codebooks (16k–65k) alive; without it deep levels collapse.
- Warm-started codebooks must be "matured" (resets off, slow EMA) or
  resets destroy the loaded structure.
- **Rate-aware VQ**: a gentle rate weight recovers a few percent of rate
  for free, then plateaus if the coarse codebook starves — schedule rate
  pressure, don't crank it.

## Conditioning

- **Location explains a large fraction of coarse-embedding variance**
  (~47% on the flagship dataset); a pretrained location encoder beats raw
  spherical harmonics on the deployment (interpolation) regime. The
  marginal codec value concentrates on the coarse tier.
- **Co-adapting an encoder against a frozen location prior degrades it
  late in training** (the pre-quantization ceiling falls); freezing the
  encoder while conditioning trains avoids it. Watch the ceiling, not the
  codebook gap, to diagnose.
- Stats normalization (per-tile mean/std side channel) was very effective
  on multispectral imagery; embedding fields are already unit-norm and
  need none.

## Evaluation

- **Split geographically** (hashed lat/lon cells), never randomly — chips
  from one region leak.
- **Downstream probes** (frozen-embedding linear/kNN on held-out,
  contamination-checked windows) showed compression is nearly free for
  downstream regression across the whole R-D front — plain cosine
  understates usable quality. Also track *centered* cosine: a block-mean
  reconstruction scores high plain cosine while carrying no spatial
  structure.
- A flat low-rate tail in an R-D curve usually means a decoder floor, not
  an entropy gap — fix decode capacity before blaming the coder.
- Compare against classical baselines at *actual file bytes*, not nominal
  bits.

## Data

- Windowed codecs produce seam artifacts between neighbouring windows
  (inter-window error well above intra-window); a small post-decode
  deblocker conditioned on the boundary removes most of it at zero rate
  cost.
- If a source is upsampled onto a finer grid (period-N pixel structure),
  phase-aligned cropping lets the model reproduce the high-frequency
  structure instead of averaging it.
