# Onboarding a new source: from measurements to an architecture

The goal is to be able to finish this sentence from evidence rather than
habit:

> *"Build a pyramid with **L** latent dims on **P** tiers at grids **G**,
> codebooks of size **K** for **N** levels, window **W**, **X** attention
> blocks, with these tricks."*

Every measurement below exists to constrain one of those. Anything that
does not is diagnostics, not design, and belongs in `ANALYSIS_SUITE.md`.

Status key: **[have]** committed and used, **[partial]** exists but
source-specific or incomplete, **[missing]** not built.

---

## 1. Spectral — structure across bands

| measurement | decides | status |
|---|---|---|
| Global PCA of pixel vectors: eigenspectrum, participation ratio, d90/d95/d99 | **L** floor; whether a linear front-end suffices | [partial] `patch_intrinsic_dim` does this pooled |
| Band cross-correlation matrix | band redundancy; whether to decorrelate before coding; band grouping | **[missing]** |
| Single-pixel autoencoder R-D: dim x bits | **L** and bits-per-dim saturation, context-free | [have] `pixel_rd.py` |
| Normalisation sensitivity: raw vs unit-norm vs per-band standardised | the LOSS (cosine vs MSE), and whether to transmit a gain | [partial] `data_probe` measures spread only |
| Per-band variance / dynamic range | whether bands need per-band weighting | [partial] `data_probe` |

**Rules established so far.** A learned bottleneck saturates at **~4 bits
per latent dimension** — beyond that, extra rate on the same dim count
buys nothing (`pixel_rd`, confirmed across dim 16/32/64/128). And
magnitude spread decides the loss: AlphaEarth is exactly unit-norm
(p99/p1 = 1.00) so cosine is unarguable; Tessera spans 1.57x, so an
absolute loss is measuring something real.

## 2. Spatial — structure across pixels

| measurement | decides | status |
|---|---|---|
| 2D power spectrum (radially averaged) | **P** and the grids **G**: where the energy actually sits in frequency | **[missing]** |
| Autocorrelation vs lag | finest useful cell size; window **W** | [have] `data_probe`, `residual_struct` |
| Within-patch vs global PCA / clustering | whether per-patch conditioning can pay at all | [have] `patch_intrinsic_dim` |
| Neighbour-predictive entropy (rANS-achievable) | the LOSSLESS FLOOR the codec must beat | [have] `data_probe` |
| Patch-size sweep: between- vs within-window variance vs W | **W**, and whether a mean side-channel is worth it | **[missing]** |

**Rules established so far.** Correlation falling below 0.5 at lag 4 px on
Tessera is consistent with the deployed fine tier's 4 px cells (16x16 grid
on a 64 px window) — the first case where a grid choice is retrospectively
justified by a measurement. The pooled/local complexity ratio bounds what
per-patch conditioning can ever buy: 0.92x (S2, nothing), 1.44x (AE),
2.90x (Tessera).

## 3. Information-theoretic — rate and quantization

| measurement | decides | status |
|---|---|---|
| Naive per-pixel entropy of stored values | the trivial baseline | [have] `data_probe` |
| Gaussian R-D bound by reverse water-filling | total rate target; whether a learned coder is earning its keep | [have] `moment_rd` (moments only) |
| Value histograms; quantization sensitivity (int8/int6/int4) | **book_bits**, latent precision, whether int8 passthrough is lossless | [partial] `tier_rate_attrib` shows int8 is lossless for latents |
| Sample-budget check: vectors per codeword | **K** and **N** | [have] as a rule, not a tool |

**Rules established so far.** The **sample-budget crossover** is the
sharpest rule we have: a codebook needs **~450 vectors per codeword**
inside the EMA window or fitting is worse than not fitting. That makes K a
derived quantity:

```
vectors_per_code = batch x cells_in_tier x ema_window / K   >=  ~450
```

K=16384 at 8 levels gave 18–54 and lost 90% of its codebook; K=256 at 20
levels gives thousands, and dead-code resets fell from 6,500–8,200 per
event to 1–5, at 12x smaller download for equal R-D.

**N** follows from per-level residual decay: levels keep earning while
their contribution is a material fraction of level 1. Tessera tier 2 at
L20 still contributes 17% of L1, i.e. **not saturated — 20 is too few**,
and each further level costs 0.5 b/px.

## 4. Use-case drivers — what the codec is *for*

**The weakest axis, and the one that should arguably come first**, since
it decides which metric everything else optimises.

| measurement | decides | status |
|---|---|---|
| Downstream task eval on decoded vs original | the objective; the rate at which quality stops mattering | **[missing]** |
| Per-band reconstruction quality | whether users taking band subsets are served; per-band loss weighting | **[missing]** |
| Metric audit: cosine vs MSE vs task | whether `cos` is flattering us | **[missing]** |
| Structured-use patterns (band subsets, coarse-only decode) | whether the matryoshka ladder matches real access | **[missing]** |

**Why this matters concretely.** We currently optimise masked MSE in
absolute units and *report* cosine. Those disagree: at cos 0.959 the
relative L2 error is 0.256. If downstream users consume direction, we are
spending capacity on a magnitude nobody scores; if they consume the raw
vector, cosine is flattering us and 0.256 is the honest number. **We have
not checked**, and it changes three decisions at once: input
normalisation, absolute-vs-relative loss, and whether the scalar-gain
channel is a fix or a distraction.

---

## The derivation chain

Ordered so each step consumes the previous ones.

| # | parameter | derived from | current evidence |
|---|---|---|---|
| 0 | **objective** | use-case: task eval, band subsetting | **unmeasured** |
| 1 | **W** (window) | autocorrelation range; between/within variance vs W | 64 px, unjustified |
| 2 | **P**, **G** (tiers, grids) | power spectrum; autocorrelation scale breaks | fine cell 4 px matches lag-0.5 point |
| 3 | **L** (latent dims/tier) | global PCA d95; ~4 bits/dim saturation | tier 2 L=256 for 160 bits/cell — likely over-provisioned |
| 4 | **K** | sample-budget: batch x cells x window / K >= 450 | K=256 verified healthy |
| 5 | **N** (levels) | per-level residual decay to a target rate | tier 2 not saturated at 20 |
| 6 | **X** (attention) | ceiling vs capacity; where the ceiling stops moving | unmeasured |
| 7 | **tricks** | mean removal, gain, conditioning | see below |

**Trick selection, from measurements we already have:**

- **Transmit the patch mean** when it is a large share of energy *and*
  cheap. Tessera: 85.5% of per-window energy, +0.0069 cosine for 0.188
  b/px, ~60x more rate-efficient than pyramid bits. **Yes.**
- **Do not force the std.** Matching it *hurts* (0.9700 -> 0.9678) because
  the decoder is already smoother than the truth, so rescaling amplifies
  error. Condition on it instead.
- **Transmit a scalar gain** when the mean-removed residual has wide
  dynamic range. Tessera: input spans 1.57x but the residual spans
  **4.16x**, so a single scalar (std PC1 is a pure gain, cos 0.9953,
  corr 0.998 with ||sigma||) at ~8 bits = 0.002 b/px would flatten it.
  **Untested.**
- **Truncate during any VQ-free warmup.** A long full-depth warmup lets
  coarse tiers atrophy (tier 0 fell to 1.16 effective dims of 512, first
  level at perplexity 1) because nothing requires them to be informative.

---

## What to build next, in priority order

1. **Use-case axis (§4)** — nothing exists, and it sets the objective every
   other choice inherits.
2. **Power spectrum + patch-size sweep (§2)** — the only principled route
   to P, G and W, which are currently inherited by habit.
3. **Band cross-correlation + global pixel PCA (§1)** — cheap, and gives
   the L floor directly.
4. **Quantization sensitivity sweep (§3)** — closes `book_bits`.

Consolidating the remaining shell diagnostics (codebook health, latent
health, training dynamics) is *lower* priority than any of these: they
tell us whether a run is healthy, not what to build.
