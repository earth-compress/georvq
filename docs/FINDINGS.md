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
- **Past the throughput knee, batch size buys codebook health, not
  speed.** On the S2 pyramid (A100 80GB, 96 px windows, grad checkpointing)
  windows/s saturates from batch 768 upward — 445 / 628 / 631 / 634 / 570
  at 384 / 768 / 1024 / 1536 / 2048 — so a larger batch there costs
  proportional wall-clock per step and returns only better gradients and
  more vectors per codeword. Pick the batch from the sample-budget rule,
  then the step count from the wall-clock you have.

## Codebook sizing

- **K is derived, not chosen.** A codebook needs roughly
  **450 vectors per codeword** inside its EMA window, or fitting it is
  worse than not fitting it:

      vectors_per_code = batch x cells_in_tier x ema_window / K  >=  ~450

  K=16384 at 8 levels sits far below that (18-54 vectors/code) and loses
  90% of its codebook; K=256 at 20 levels gives thousands, dead-code
  resets fall from 6,500-8,200 per event to 1-5, and the download is 12x
  smaller for equal R-D. Since cells rise with tier, this makes codebooks
  something to grade UPWARD with tier rather than fix globally.

- **Depth must be sampled along the truncation path, not nested.**
  Drawing the coarse depth uniformly and each finer tier conditionally
  makes deep fine-tier levels vanishingly rare -- measured
  P(active)=0.006 for tier-2 level 14, i.e. those books saw 0.6% of rows.
  Spreading uniformly along the monotone path (one step per codebook)
  raises the same level to 0.197. See `path_tuples` in
  `georvq/model/codec.py`.

- **Cells dominate rate.** One tier-2 level costs ~64x one tier-1 level
  on a 1/4/16 pyramid, so depth should be allocated inversely to cell
  count, and the efficient ladder always saturates coarse, then mid, then
  adds fine levels one at a time.

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
- **A normalizer's stats are rate, and they hide.** `side_bits` once
  charged only the transmitted-mean channel (`stats_cond`), so a source
  that instead *normalizes* — Sentinel-2 removes 13 per-band means and 13
  stds per window — had that channel counted nowhere. It hides because it
  is invisible at the top of the ladder and dominant at the bottom: on the
  S2 pyramid the raw channel is +1.6% of full depth but **3.31x** the
  coarse-only rung. Charge every side channel at every depth.

- **PCA-packing a stats vector is worth ~5x, and the basis is cheap to
  fit.** 26 per-band mean/std floats, PCA with tolerance-coded uint8
  components (`inference/stats_codec.py`): **9.73 bytes/window** measured
  on held-out S2 windows at tolerance 0.01, against 26 x f16 = 52 bytes —
  a **5.3x** reduction, with 5 components carrying 99% of the stats
  variance. Independently reproduces the ~11 B/tile the ring-space S2
  betas shipped, on a basis refitted for a different window size.

- **Whether a stats side channel pays is a property of the field, not of
  the modality.** Per-tile mean/std normalization was very effective on
  multispectral imagery. A unit-norm embedding field (AlphaEarth) needs
  none at all. An embedding field that is *not* unit-norm (Tessera,
  magnitude spread 1.57x) wants the mean transmitted but not the std:
  the mean is 85.5% of window energy and buys +0.0069 cosine for
  0.188 b/px, while forcing the std *hurts* (0.9700 -> 0.9678) because
  the decoder is already smoother than the truth, so rescaling amplifies
  its error rather than restoring detail.

## Metric discipline

Conventions that each cost a wrong conclusion before they were adopted.

- **Use relative gap closure** `d(1-m)/(1-m)`, not absolute deltas, for
  any bounded metric. It exposed a depth-graded improvement (coarse 20%,
  mid 9%, full 0.9% per 1k steps) that absolute deltas flattened into
  noise.
- **Match training budget before believing a crossover.** The
  AE-vs-linear crossover moved 200 -> 300 -> 500 -> 600 bits as data and
  steps were added, and every move favoured whichever arm had been
  under-resourced.
- **A transmitted mean inflates plain cosine.** Track `ccos` (centred
  cosine) on any model with a stats channel: a coarse-only decode read
  0.9154 plain cosine while contributing no spatial structure at all,
  and a block-mean reconstruction scores 0.92.
- **Don't extrapolate a flat stretch.** A coarse tier predicted to reach
  0.11 reached 0.35: its recovery was a phase transition, not a line.
- **Latent residual is a weak proxy for quality.** 0.22 latent residual
  maps to ~0.001 cosine through the sigma ladder, because the decoder is
  trained for noise robustness. Always decode before judging.

## Evaluation

- **Split geographically** (hashed lat/lon cells), never randomly — chips
  from one region leak.
- **Downstream probes** (frozen-embedding linear/kNN on held-out,
  contamination-checked windows) showed compression is nearly free for
  downstream regression across the whole R-D front — plain cosine
  understates usable quality.
- **Cosine does not transfer across sources.** At matched reconstruction
  cosine the Tessera codec costs *less* downstream accuracy than the
  AlphaEarth codec (LUCAS land cover: AE at cos 0.9532 loses 0.043
  balanced accuracy, Tessera at cos 0.9475 loses 0.005), because Tessera
  carries ~2.5x more independent variation per pixel and much of it is
  weakly correlated with land cover. Set a quality target from the task,
  never by transplanting another source's cosine.
- **Establish a probe's noise floor before reading it.** Re-probing every
  rate point under repeated split seeds gave +/-0.017 on LUCAS and
  +/-0.023 on LCMAP — so LCMAP cannot resolve anything below ~0.05, and
  its "compression beats the original" readings are artefacts.
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
- **A store is a choice, not a step.** A chip store exists to give the
  pool fixed-geometry chips; a source already laid out that way gains
  nothing from being resharded. Major-TOM Core-S2L1C (106k scene dirs on
  a 10 km grid, 1.2 TB) is read in place, and GLO-30 likewise.
- **Check the compression block size before designing a reader.** The
  Major-TOM rasters are LZW with 1024 px blocks over a 1068 px image, so a
  windowed read decompresses essentially the whole band: **4.7 ms** for a
  full band against 0.4 ms for a 96x96 window that hits one warm block.
  Reading whole scenes and slicing in the pool is simpler and no more I/O.
  Measured end to end: 64 ms and 12.1 MB per 13-band scene.
- **Pool size and refresh rate do different jobs.** Reuse per chip is
  `windows/s / refresh_per_s` and is *independent of pool size*; the pool
  buys instantaneous diversity — how many distinct chips one batch can
  draw from. A batch as large as the pool resamples the same chips no
  matter how fast the refresh runs.

## Release engineering

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
  parked below the release bar (see Deferred with evidence above).
- **Encoder-side self-measurement is nearly free and very valuable**:
  decoding the rate ladder during encode (fused pass: shared encode +
  cumsums, ~2x the base encode cost) yields per-blob and per-window true
  error stats that make quality-controlled delivery possible with no
  predictor network.

## Deferred with evidence

Ideas that were measured, found real, and not shipped. Recorded so they
are not re-derived; each was parked for cost or for missing a quality
bar, not for being wrong.

- **A decoder-side error predictor is feasible and cheap.** The 24
  per-(tier, level) RVQ increment norms -- pure bitstream features the
  decoder already sums -- predict log window-mean error at **r = 0.94**
  with a linear probe, better than the encode-side coarse-residual norm
  (0.898). 42% of predicted worst-5% windows are truly in the worst 5%.
  So "which pixels are likely bad" and "how much would deeper levels
  help here" are answerable from indices alone, with no original.
- **Cheap decoders hit a quality wall.** Tier-subset decoders with
  low-res heads and bilinear upsampling (0.76M / 4.25M params) run
  17-28x faster but land 0.006-0.1 cos below the 31M trunk at matched
  rate. Speed is available; quality at speed is not, on this
  architecture.
- **Blocked-halo attention is worth it if narrow local attention comes
  back.** Computing a fine tier's neighbourhood attention in blocks with
  a halo, rather than gathering per query, measured **7x faster**. The
  narrow-suffix stream it was built for was abandoned with the
  architectures that used it, but the kernel idea stands on its own.

## Codebooks: sizing, fitting and health

- **A per-window normalizer and a grid-1 coarse tier compete for the same
  information.** A normalizer removes per-window statistics before the
  encoder and ships them separately; a coarse tier's whole job is the
  window-level summary. Whether that collision matters depends on HOW MUCH
  the normalizer takes relative to what the window holds, and the two
  shipped cases sit at opposite ends:

  | | across-window latent var | coarse L1 perplexity/K |
  |---|---:|---:|
  | S2 (26 values removed: mean+std x 13 bands) | **0.0033** | **0.016** (~1 code of 64) |
  | DEM (2 values removed: mean+std x 1 band) | **6.12** | **0.295** (~76 of 256) |

  On S2 the normalizer takes the entire window-level spectral signature,
  the coarse latent is left encoding a near-constant (0.003% of its own
  energy is variance), and its first RVQ level collapses to a single code.
  On DEM the same structure is healthy: two scalars leave the window's
  relief SHAPE to encode, and every coarse level stays used. Measured at
  step 8000 of each run.

  The diagnosis is cheap and worth running early on any normalized source:
  compare a tier's across-window latent variance against its norm, and
  read the first level's perplexity. The fix is not to delete the coarse
  tier — its rate is negligible, 48 bits of 25,776 on the S2 pyramid — but
  to hand back what was taken (condition on the stats, as the ring-space
  S2 release did), or to leave enough in the input for the tier to encode.

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
  (K=4096+2 gain bits) and 0.3087 (K=1024+4). Implemented, measured, and
  since removed. Its premise -- that books waste capacity on magnitude --
  does not hold: the shape-codebook capacity the split costs exceeds the
  magnitude precision it buys.
- **Cross-fitted refinement levels help only degenerate tiers.** +33% on
  the collapsed tier (where levels 1-2 contributed 1.3%), **-13% on a
  healthy one** (0.2556 vs 0.2262) — the fold models are weaker, so the
  residuals they produce mismatch what the shipped book will see.
- **MSE-optimal codebooks are ~decoder-optimal.** Freezing AE's encoder
  AND decoder and optimising only the codewords against the *decoded*
  loss gained **+0.00007 cos** (0.99585 -> 0.99592, monotone, held-out)
  — a real but negligible degree of freedom. Not worth a pipeline stage.
- **Split latent datasets by *window*, never by cell.** Adjacent cells
  are near-duplicates (mean NN distance ~6% of vector norm), and a random
  split makes k-means look ~500x better than it is.
