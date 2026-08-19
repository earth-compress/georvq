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

## Residual refinements

A tier's RVQ stops at a finite depth and leaves a residual. Coding that
residual with a cheap scalar quantizer and adding it back at decode
(`georvq.inference.refine`) buys the gap back without touching the
codebooks or the index stream — but only under one precondition, and
only on one tier, and which tier is not something to inherit.

- **Only worth it when the ceiling is visibly above full depth.** The
  refinement can recover, at most, the distance between full-depth
  reconstruction and the same model decoding un-quantized latents. If
  that gap is small the tier is saturation-limited, not rate-limited,
  and there is nothing to buy — this is why the AlphaEarth mid and fine
  sidecars were worthless. Measure the ceiling first; it is the budget.

- **Which tier holds the slack differs by model, and the wrong choice
  returns exactly zero.** Measured on the Tessera release checkpoint
  (768 held-out windows, share of the centred-cosine ceiling gap
  recovered, base 10.85 b/px):

  | refined tier | rate | gap closed |
  |---|---:|---:|
  | 0 (coarse), 1–2 bits | +0.13–0.25 b/px | **0.0%** |
  | 1 (mid), 1–2 bits | +2.0–4.0 b/px | 1.9% |
  | 2 (fine), 1 bit | +16 b/px | 56.2% |
  | 2 (fine), 2 bits | +32 b/px | 81.2% |

  AlphaEarth refines tier 0 and recovers ~98% of its gap there; the same
  policy on Tessera recovers nothing at all. Tessera's coarse book is
  healthy and its fine tier is still rate-limited (residual decaying at
  level 20), so that is where the slack sits.

- **Cost scales with cells, so the same idea is cheap or ruinous
  depending on the tier.** A refinement costs `cells x dim x bits`. On a
  1/4/16 grid pyramid that is 1x, 16x, 256x cells — AlphaEarth's coarse
  sidecar is +1.5% of rate and can simply be left on, while Tessera's
  fine refinement more than doubles the bitstream. The rule generalises:
  a refinement is a default when it lands on a coarse tier and an opt-in
  when it lands on a fine one.

- **After a deep RVQ, rotate and use Lloyd; don't bother with PCA.** The
  post-RVQ residual is close to isotropic quantization noise, so a
  random rotation and a PCA rotation land within 1% of each other (rel
  MSE 0.1425 vs 0.1436 at ~1 bit/dim) — there is no structure left for
  PCA to exploit, which is exactly the regime where a Gaussian Lloyd-Max
  scalar quantizer is near-optimal. The rotation costs nothing in the
  bitstream (it is a seed), and equalising variance across coordinates is
  what a fixed levels-per-dimension coder wants. Random rotation even
  beats PCA on *raw* latents (0.4062 vs 0.4249): PCA concentrates
  variance and then quantizes the high-variance directions too coarsely,
  so it only pays with bit allocation.

- **A uniform quantizer's clip point must be retuned with its bit
  depth.** The convention that puts the outermost reconstruction level at
  `sigma x std` is good at 2 bits with sigma=2, but at 1 bit it places
  the two levels at ±2 std where the optimum is ±0.8, and the
  "refinement" adds error instead of removing it. One bit per dimension
  belongs to the Lloyd family.

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

## Release engineering (AlphaEarth cycle, 2026-07)

- **Entropy coding on healthy codebooks doesn't pay.** Order-0 rANS:
  ~0.6% over npz-deflated raw indices on real tiles (fine tier at ~95%
  of codebook entropy; deflate strips the uint16 padding to ~13.4
  effective bits of 14) — while dominating decode time. The remaining
  gap is cross-tier context (~-22% measured by probes), which needs the
  learned context model, not a better static coder.
- **The rate ladder is extremely asymmetric** (coarse 11 / mid 208 /
  fine 3,584 bits per level): the measured Pareto front always
  saturates coarse, then mid, then adds fine levels. Hand-picked
  "balanced" tuples like 6-4-4 are strictly dominated.
- **On AlphaEarth the coarse quantizer gap carries the outlier tail.**
  Worst-0.1% pixels under full-depth decode are coarse-codebook misses
  (rare land covers; window-scale bias), correlated 0.74 with the coarse
  residual norm — and a 2-bit scalar-quantized residual sidecar
  (66 B/window, per-window scale) recovers ~98% of the gap *including
  the tail*, for +1.5% rate. On that model mid/fine sidecars are
  worthless-to-harmful: those codebooks sit at their ceiling. See
  **Residual refinements** below for why that is a fact about the model
  rather than about the architecture.
- **bf16 belongs on decoder forwards only.** bf16 encoder latents flip
  ~24% of near-tie RVQ indices (quality-neutral but bitstream-unstable
  across runs/hardware); fp32 encoder + bf16 decoder keeps indices
  bit-stable at ~95% of the speed win.
- **Cheap decoders hit a quality wall.** Tier-subset decoders with
  low-res heads + bilinear upsampling (0.76M/4.25M params) run 17-28x
  faster but land 0.006-0.1 cos below the 31M trunk at matched rate —
  parked below the release bar (`docs/PARKED.md`).
- **Encoder-side self-measurement is nearly free and very valuable**:
  decoding the rate ladder during encode (fused pass: shared encode +
  cumsums, ~2x the base encode cost) yields per-blob and per-window true
  error stats that make quality-controlled delivery possible with no
  predictor network.

## Tessera stage-2 / codebook investigation (2026-08-10)

- **Tier-redundancy collapse is real and silent.** In the 4-tier `uf`
  model (grids 1/4/16/32 over 64 px) the fine tier carries only **0.2%**
  of its variance *within* a window — its 256 cells are near-copies of
  one window-level vector. The same tier is healthy in every 3-tier
  model (96.8% in `s1_4x`, 97.9% in `s1_2x`); it collapses only once the
  2 px ultrafine tier is added and takes over its job. Costs ~62% of the
  rate budget (256 cells x 8 lvl x 14 b) and is invisible in the ceiling,
  because a duplicated tier wastes rate, not quality. **Check
  within-window latent variance per tier before trusting any tier.**
- **Codebook perplexity is not a health metric — it is gameable.** An
  isolated agent, given only vectors and a fixed scorer, reached 11.94 /
  11.96 / 11.97 bits of 12 with all 4096 codes used at every level, at
  **0.014% of energy explained** (verified fp64, independent re-score);
  decoded cosine 0.45 vs 0.98 for k-means. Utilization is bought by
  quantizing directions where the data has no structure: 8.24 bits on
  the top-12 PCA directions vs 11.90 deep in the tail. Judge codebooks
  on rate-distortion, never on usage.
- **EMA codebook training can degrade a good fit, source-dependently.**
  Frozen encoder, static latents, no resets firing: `ema_decay 0.99`
  took residual 0.033 -> 0.047 by 2400 steps (damage concentrated in
  deep levels = cascade drift); `0.9995` was flat. On Tessera a fresh
  8-iter k-means beat 60k EMA steps end-to-end (**0.9766 vs 0.9744**).
  On AE the same test is a tie (**0.9963 vs 0.9964**) — AE's latents are
  much lower-rank (d95 19/34/28 vs 51/91/71), so its books are better
  sampled and stable. Prefer fit-once-and-freeze; it is never worse.
- **At matched rate, bigger K beats more levels** (healthy tier, ~112
  b/cell): K=16384 x8 -> 0.2252, 4096 x9 -> 0.2354, 1024 x11 -> 0.2411,
  256 x14 -> 0.2539. The opposite conclusion drawn from a *collapsed*
  tier was an artefact. Keep the AE ladder (2048/8192/16384).
- **Gain-shape VQ loses at matched rate**: plain VQ 0.2251 vs 0.2537
  (K=4096+2 gain bits) and 0.3087 (K=1024+4). Implemented and tested
  (`vq.gain_shape`), kept for reference. Its premise (books wasting
  capacity on magnitude) does not hold; the shape-codebook capacity it
  costs exceeds the magnitude precision it buys.
- **Cross-fitted refinement levels help only degenerate tiers.** +33% on
  the collapsed tier (where levels 1-2 contributed 1.3%), **-13% on a
  healthy one** (0.2556 vs 0.2262) — the fold models are weaker, so the
  residuals they produce mismatch what the shipped book will see.
- **MSE-optimal codebooks are ~decoder-optimal.** Freezing AE's encoder
  AND decoder and optimising only the codewords against the *decoded*
  loss gained **+0.00007 cos** (0.99585 -> 0.99592, monotone, held-out)
  — a real but negligible degree of freedom. Not worth a pipeline stage.
- **Measurement gotchas.** (a) Split latent datasets by *window*, never
  by cell: adjacent cells are near-duplicates (mean NN distance ~6% of
  vector norm) and a random split makes k-means look ~500x better than
  it is. (b) Latent residual is a weak proxy for quality — 0.22 residual
  maps to ~0.001 cos via the sigma ladder, because the decoder is
  trained for noise robustness. Always decode.
