<!-- DRAFT: restructured to cite the findings ledger; wording not yet
reviewed by the maintainer. -->

# Onboarding a new source

This document is for anyone designing an architecture for a new source.
It lists the measurements that can be made on a source before any model
is trained, says which architecture parameter each one constrains, and
records the working rules found so far. The aim is to be able to fill in
a sentence of the following form from evidence rather than habit:

> *"Build a pyramid with L latent dims on P tiers at grids G, codebooks
> of size K for N levels, window W, X attention blocks, with these
> tricks."*

Every measurement below exists to constrain one of those parameters.
Measurements that describe a source or diagnose a trained model without
constraining a parameter are kept in `analysis/README.md`, so that this
document lists only design inputs.

This is a work in progress. Determining a good architecture from data
statistics is a hard problem, and this document does not solve it. What
the measurements do is narrow the search and suggest a sensible starting
configuration. The designs that shipped were still reached by trial and
error per source, with these numbers as the starting point
(`docs/DESIGN.md` says the same). The "working rules" below are
hypotheses, each of which has survived one or a few experiments, and
their evidence is recorded in the ledger.

Status key: [have] committed and used, [partial] exists but
source-specific or incomplete, [missing] not built.

## 1. Spectral structure across bands

| measurement | decides | status |
|---|---|---|
| Validity: nodata fraction and convention | masking in the adapter; without it every later statistic is poisoned by fill values | [have] `data_probe` |
| Global PCA of pixel vectors: eigenspectrum, participation ratio, d90/d95/d99 | **L** floor; whether a linear front-end suffices | [partial] `patch_intrinsic_dim` does this pooled |
| Per-tier PCA of what each tier would transmit (block means, residual block means) | **L** per tier, from the data rather than a model | [have] `pyramid_sizing` |
| Band cross-correlation matrix | band redundancy; whether to decorrelate before coding; band grouping | **[missing]** |
| Single-pixel autoencoder R-D: dim x bits | **L** and bits-per-dim saturation, context-free | [have] `pixel_rd.py` |
| Normalisation sensitivity: raw vs unit-norm vs per-band standardised | the LOSS (cosine vs MSE), and whether to transmit a gain | [partial] `data_probe` measures spread only |
| Per-band variance / dynamic range | whether bands need per-band weighting | [partial] `data_probe` |

Two working rules have come out of these measurements so far. A learned
bottleneck saturates at ~4 bits per latent dimension
([`bits-per-dim-saturation`](../lab/findings/bits-per-dim-saturation.md)).
Magnitude spread decides the loss. AlphaEarth is exactly unit-norm
(p99/p1 = 1.00), so the length of a vector carries no information and
cosine loses nothing as a loss, whereas Tessera spans 1.57x, so an
absolute loss is measuring something real.

## 2. Spatial structure across pixels

| measurement | decides | status |
|---|---|---|
| 2D power spectrum (radially averaged) | **P** and the grids **G**: where the energy actually sits in frequency | **[missing]** |
| Autocorrelation vs lag | finest useful cell size; window **W** | [have] `data_probe`, `residual_struct` |
| Within-patch vs global PCA / clustering | whether per-patch conditioning can pay at all | [have] `patch_intrinsic_dim` |
| Neighbour-predictive entropy (rANS-achievable) | the LOSSLESS FLOOR the codec must beat, and how much simple prediction saves | [have] `data_probe` |
| Per-tier energy decomposition: window mean, block means per grid, sub-grid floor | how the budget should split across tiers; whether a mean side-channel is worth it; the finest useful grid | [have] `pyramid_sizing` |
| Patch-size sweep: between- vs within-window variance vs W | **W** | **[missing]** (single-W numbers exist via `pyramid_sizing`) |

There are two working rules here so far. Tessera's fine-cell size is the
first grid choice to be justified, in retrospect, by a measurement
([`tessera-fine-grid-matches-autocorr`](../lab/findings/tessera-fine-grid-matches-autocorr.md)).
The pooled/local complexity ratio bounds what per-patch conditioning can
ever buy, and it differs sharply by source
([`patch-conditioning-headroom`](../lab/findings/patch-conditioning-headroom.md)).

## 3. Rate and quantisation

| measurement | decides | status |
|---|---|---|
| Naive per-pixel entropy of stored values | the trivial baseline | [have] `data_probe` |
| Gaussian R-D bound by reverse water-filling | total rate target; whether a learned coder is earning its keep | [have] `moment_rd` (moments only) |
| Value histograms; quantization sensitivity (int8/int6/int4) | **book_bits**, latent precision, whether int8 passthrough is lossless | [partial] `tier_rate_attrib` shows int8 is lossless for latents |
| Sample-budget check: vectors per codeword | **K** and **N** | [have] as a rule, not a tool |

The first working rule here is the sample-budget crossover
([`codebook-sample-budget`](../lab/findings/codebook-sample-budget.md)).
A codebook needs ~450 vectors per codeword inside the EMA window, or
fitting it is worse than not fitting it, which makes K a quantity
derived from the training setup:

```
vectors_per_code = batch x cells_in_tier x ema_window / K   >=  ~450
```

N follows from per-level residual decay. Levels keep paying while their
contribution is a material fraction of level 1's, and on Tessera the
fine tier is not saturated at the shipped depth
([`tessera-fine-tier-not-saturated`](../lab/findings/tessera-fine-tier-not-saturated.md)).

## 4. What the codec is used for

Status: [have]. The downstream evaluation is `scripts/eval_rate_sweep.py`,
with datasets built for both sources on the group server (42 GB for
Tessera and 122 GB for AE, covering ASTER GED, LUCAS LC/LU and LCMAP
LC/LU), and its results are in `lab/findings/tessera-transparency-point.md`.
It has already shown that the default objective, reconstruction cosine,
cannot be carried from one source to another.

The protocol encodes each labelled pixel's 64x64 context window, decodes
it at a rate tuple, and reads the reconstructed embedding at the label
pixel. It then fits 1-NN, 3-NN and linear probes, with train and test
split by source tile (~82 km blocks) so that spatial autocorrelation
cannot leak, and reports the delta against the same probe on the
uncompressed embedding. Every rate point is re-probed under repeated
split seeds to establish the probe's noise floor first
([`probe-noise-floors`](../lab/findings/probe-noise-floors.md)).

It has established two things.

1. Reconstruction cosine does not transfer across sources, so a quality
   target must come from the task
   ([`cosine-not-transferable`](../lab/findings/cosine-not-transferable.md)).
2. Tessera is downstream-transparent above ~1 b/px on point-thematic
   tasks, degrades faster than AlphaEarth below ~0.5 b/px, and the
   fine-scale detail the high rungs buy is unmeasured by the current
   suite
   ([`tessera-transparency-point`](../lab/findings/tessera-transparency-point.md)).

This has two consequences for how a source is specified.

First, the transparency point is the rate to recommend for
point-thematic work, but it does not make the higher rungs useless.
These tasks stop improving around the transparency rate, which is where
a user doing point-thematic work should be pointed. The right rung still
depends on the task, and the benchmarks that exist today do not exercise
the fine-scale detail the high rungs buy. What the result does rule out
is transplanting a cosine target from one source to another.

Second, the latent width L is over-provisioned by a measurable margin.
The latent d95 (PCA of the trained model's latents, per tier) sits
several times below the deployed latent widths
([`cosine-not-transferable`](../lab/findings/cosine-not-transferable.md)
has the numbers), so the widths could narrow before the probes would
notice.

The main gap in this evaluation is that all five benchmarks are coarse
thematic labels at a single point, the regime AlphaEarth was designed
for. Tessera's per-pixel information may well pay on tasks with finer
spatial or temporal structure, and none are in the suite. A fine-spatial
or temporal benchmark is therefore the next addition this axis needs,
and the use-case entry in `lab/TODO.md` depends on it as well.

### A quality target for Sentinel-2, from practice

Reconstruction cosine and R2 are what S2 models are trained and
monitored on, but a remote-sensing user judges an S2 product by its
absolute reflectance error per band, relative to that band's mean. The
thresholds are:

- < 1% on the 10 m bands is the rough goal.
- 1-2% is borderline. This is about the level of Landsat/Sentinel
  harmonisation differences, so a codec sitting there is adding error
  comparable to a known cross-sensor bias.

This is the number to report alongside R2 for any S2 model, because the
two can move apart. R2 is computed in per-window standardised space,
where a band's error is measured in units of its own spread, whilst the
practitioner reads error against the band's absolute level.

## The five-source measurement table

The table below was produced by `analysis/data_probe.py` and
`analysis/pyramid_sizing.py`, with W=64 and grids 1/4/16, and it is the
output that the rest of the document uses. Every row
comes from a measurement listed in sections 1-3: bands and valid are
basic properties from `data_probe` (section 1), tier-0 energy, the
per-tier d95 and the sub-4px residual share come from
`pyramid_sizing`'s energy decomposition (sections 1-2), the lossless
floor and prediction savings from the neighbour-predictive entropy
(section 2), magnitude spread from the normalisation sensitivity
(section 1), and the autocorrelation range from section 2. The d95 row
is the data d95, the PCA of what each tier would transmit (block means
and residual block means), measured before any model exists. It is a
different quantity from the latent d95 quoted in section 4 and the
derivation chain, which is the PCA of a trained model's latents. The
data d95 is a design input and the latent d95 a diagnosis of a finished
model, and the gap between them measures how efficiently the encoder
used its width.

| | DEM | S1 RTC | S2 L1C | AlphaEarth | Tessera |
|---|---:|---:|---:|---:|---:|
| bands | 1 | 2 | 13 | 64 | 128 |
| valid | 100% | 100% | 99.5% | 81.5% | 95.9% |
| tier-0 energy | 99.8% | 97.2% | 97.1% | 97.7% | 87.3% |
| data d95 tier 0/1/2 | 1/1/1 | 2/2/2 | 3/3/3 | 7/27/30 | 32/75/86 |
| residual <4 px | 1.7% | 45.5% | 11.6% | 16.6% | 46.0% |
| lossless floor b/px | 2.47 | 11.2 | 72.3 | 131 | 672 |
| prediction saves | 58.4% | 17.8% | 42.5% | 44.9% | 9.7% |
| magnitude spread | — | — (dB) | 6.6x | 1.00x | 1.57x |
| autocorr <0.5 at | >32 px | 4 px | 32 px | >32 px | 4 px |

The lossless unit is a judgement call per source (int8 as stored,
integer metre, uint16 DN, dB at 0.1 dB for SAR) and is printed with
the figures. On S2 the `left` predictor beats `(left+up)/2`, the reverse
of every other source, because the 20 m and 60 m bands are
nearest-upsampled and the horizontal neighbour is often an exact
duplicate. Entropies are order-0 per band, so every floor is
conservative.

The `residual <4 px` row measures where the residual energy sits in
scale. It does not limit what the codec can represent, because the fine
tier's tokens decode through a head that emits every pixel of their cell
individually, so sub-cell structure can be reconstructed. It is the part
of the signal for which the model has the least spare capacity.

### What the table implies per source

| | DEM | S1 | S2 | AE | Tessera |
|---|---|---|---|---|---|
| transmit mean | yes | yes | yes | yes | **yes** (0.188 b/px, +0.0069 cos) |
| per-band std | no | no | no | n/a | **no — measured harmful** (0.9700->0.9678) |
| scalar gain | already does | likely | likely | none needed | **likely, untested** (residual spans 4.16x) |
| latent dims t0/t1/t2 | 16/32/32 | 16/32/32 | 32/64/64 | 32/64/96 | **64/128/128** (deployed 512/512/256) |
| K per tier | 64/256/1024 | 64/256/1024 | 64/512/4096 | 64/1024/8192 | **64/1024/16384** |
| train metric | MSE on standardised relief; report R2 | MSE in dB + multi-look | per-band weighted MSE + spectral angle | cosine | MSE on mean-removed field; report **ccos** |

## The derivation chain

The steps are ordered so that each one uses the results of those before
it. The chain describes how the parameters should be derived, and the
evidence column records how much of each step is established so far,
which for several steps is still only habit.

| # | parameter | derived from | current evidence |
|---|---|---|---|
| 0 | **objective** | use-case: task eval, band subsetting | point-thematic transparency at ~1 b/px, cos ~0.93; cosine is NOT transferable across sources; fine-scale use unmeasured |
| 1 | **W** (window) | autocorrelation range; between/within variance vs W | 64 px, unjustified |
| 2 | **P**, **G** (tiers, grids) | power spectrum; autocorrelation scale breaks | fine cell 4 px matches lag-0.5 point |
| 3 | **L** (latent dims/tier) | data d95; ~4 bits/dim saturation | latent d95 51/91/71 of 512/512/256 on Tessera — over-provisioned several fold (`cosine-not-transferable`) |
| 4 | **K** | sample-budget: batch x cells x window / K >= 450 | K=256 verified healthy (`codebook-sample-budget`) |
| 5 | **N** (levels) | per-level residual decay to a target rate | tier 2 not saturated at 20 (`tessera-fine-tier-not-saturated`) |
| 6 | **X** (attention) | ceiling vs capacity; where the ceiling stops moving | unmeasured |
| 7 | **tricks** | mean removal, gain, conditioning | see below |

The measurements made so far settle some of the tricks and leave others
open.

- Transmit the patch mean when it is a large share of energy and is
  cheap to send. On Tessera it is 85.5% of per-window energy, and ~60x
  more rate-efficient than pyramid bits at the rate in the table above,
  so the Tessera release transmits it.
- Do not force the std. Matching it hurts (table above), because the
  decoder is already smoother than the truth, so rescaling amplifies
  error. Condition on it instead.
- Transmit a scalar gain when the mean-removed residual has a wide
  dynamic range. On Tessera the input spans 1.57x but the residual spans
  4.16x, so a single scalar (std PC1 is a pure gain, cos 0.9953, corr
  0.998 with ||sigma||) at ~8 bits = 0.002 b/px would flatten it. This
  has not been tested.
- Truncate during any VQ-free warmup. A long full-depth warmup lets
  coarse tiers atrophy (tier 0 fell to 1.16 effective dims of 512, with
  the first level at perplexity 1), because nothing requires them to be
  informative.

## What to build next

Forward work is queued in `lab/TODO.md`. The gaps this document's
status columns mark [missing] are, in the order they block the
derivation chain: a fine-spatial or temporal downstream task, the power
spectrum and patch-size sweep (the measurements that would set P, G and
W, which are currently inherited by habit), band cross-correlation with
global pixel PCA (the L floor), and a quantization sensitivity sweep
(closes `book_bits`).
