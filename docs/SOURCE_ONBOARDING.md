<!-- DRAFT: restructured to cite the findings ledger; wording not yet
reviewed by the maintainer. -->

# Onboarding a new source: from measurements to an architecture

The goal is to be able to finish this sentence from evidence rather than
habit:

> *"Build a pyramid with **L** latent dims on **P** tiers at grids **G**,
> codebooks of size **K** for **N** levels, window **W**, **X** attention
> blocks, with these tricks."*

Every measurement below exists to constrain one of those. Anything that
does not is diagnostics, not design, and belongs in `analysis/README.md`.

This is a work in progress, and the ambition should be read accordingly.
Determining a good architecture from data statistics is a famously hard
problem, and this document does not solve it. What the measurements do
is narrow the search and suggest a sensible starting configuration; the
designs that shipped were still reached by trial and error per source,
with these numbers as the starting point (`docs/DESIGN.md` says the
same). The "working rules" below are hypotheses that have each survived
one or a few experiments, recorded with their evidence in the ledger --
not laws.

Status key: **[have]** committed and used, **[partial]** exists but
source-specific or incomplete, **[missing]** not built.

---

## 1. Spectral — structure across bands

| measurement | decides | status |
|---|---|---|
| Validity: nodata fraction and convention | masking in the adapter; without it every later statistic is poisoned by fill values | [have] `data_probe` |
| Global PCA of pixel vectors: eigenspectrum, participation ratio, d90/d95/d99 | **L** floor; whether a linear front-end suffices | [partial] `patch_intrinsic_dim` does this pooled |
| Per-tier PCA of what each tier would transmit (block means, residual block means) | **L** per tier, from the data rather than a model | [have] `pyramid_sizing` |
| Band cross-correlation matrix | band redundancy; whether to decorrelate before coding; band grouping | **[missing]** |
| Single-pixel autoencoder R-D: dim x bits | **L** and bits-per-dim saturation, context-free | [have] `pixel_rd.py` |
| Normalisation sensitivity: raw vs unit-norm vs per-band standardised | the LOSS (cosine vs MSE), and whether to transmit a gain | [partial] `data_probe` measures spread only |
| Per-band variance / dynamic range | whether bands need per-band weighting | [partial] `data_probe` |

**Working rules so far.** A learned bottleneck saturates at ~4 bits
per latent dimension
([`bits-per-dim-saturation`](../lab/findings/bits-per-dim-saturation.md)).
And magnitude spread decides the loss: AlphaEarth is exactly unit-norm
(p99/p1 = 1.00) so cosine is unarguable; Tessera spans 1.57x, so an
absolute loss is measuring something real.

## 2. Spatial — structure across pixels

| measurement | decides | status |
|---|---|---|
| 2D power spectrum (radially averaged) | **P** and the grids **G**: where the energy actually sits in frequency | **[missing]** |
| Autocorrelation vs lag | finest useful cell size; window **W** | [have] `data_probe`, `residual_struct` |
| Within-patch vs global PCA / clustering | whether per-patch conditioning can pay at all | [have] `patch_intrinsic_dim` |
| Neighbour-predictive entropy (rANS-achievable) | the LOSSLESS FLOOR the codec must beat, and how much simple prediction saves | [have] `data_probe` |
| Per-tier energy decomposition: window mean, block means per grid, sub-grid floor | how the budget should split across tiers; whether a mean side-channel is worth it; the finest useful grid | [have] `pyramid_sizing` |
| Patch-size sweep: between- vs within-window variance vs W | **W** | **[missing]** (single-W numbers exist via `pyramid_sizing`) |

**Working rules so far.** Tessera's fine-cell size is the first grid
choice retrospectively justified by a measurement
([`tessera-fine-grid-matches-autocorr`](../lab/findings/tessera-fine-grid-matches-autocorr.md)).
The pooled/local complexity ratio bounds what per-patch conditioning can
ever buy, and differs sharply by source
([`patch-conditioning-headroom`](../lab/findings/patch-conditioning-headroom.md)).

## 3. Information-theoretic — rate and quantization

| measurement | decides | status |
|---|---|---|
| Naive per-pixel entropy of stored values | the trivial baseline | [have] `data_probe` |
| Gaussian R-D bound by reverse water-filling | total rate target; whether a learned coder is earning its keep | [have] `moment_rd` (moments only) |
| Value histograms; quantization sensitivity (int8/int6/int4) | **book_bits**, latent precision, whether int8 passthrough is lossless | [partial] `tier_rate_attrib` shows int8 is lossless for latents |
| Sample-budget check: vectors per codeword | **K** and **N** | [have] as a rule, not a tool |

**Working rules so far.** The sample-budget crossover is the sharpest
rule we have: a codebook needs ~450 vectors per codeword inside the EMA
window or fitting is worse than not fitting, which makes K a derived
quantity:

```
vectors_per_code = batch x cells_in_tier x ema_window / K   >=  ~450
```

The evidence is in
[`codebook-sample-budget`](../lab/findings/codebook-sample-budget.md).
N follows from per-level residual decay: levels keep paying while their
contribution is a material fraction of level 1's, and on Tessera the
fine tier is not saturated at the shipped depth
([`tessera-fine-tier-not-saturated`](../lab/findings/tessera-fine-tier-not-saturated.md)).

## 4. Use-case drivers — what the codec is *for*

**[have] — and it already overturns the default objective.**
`scripts/eval_rate_sweep.py`, with datasets built for both sources
(`/data/tessera-samples/eval` 42 GB, `/data/AE-samples/eval` 122 GB:
ASTER GED, LUCAS LC/LU, LCMAP LC/LU). Reported in
`tessera_downstream_interim.pdf`.

Protocol: encode each labelled pixel's 64x64 context window, decode at a
rate tuple, read the reconstructed embedding at the label pixel; fit 1-NN,
3-NN and linear probes; split train/test by source tile (~82 km blocks) so
spatial autocorrelation cannot leak; report the delta against the same
probe on the uncompressed embedding. Every rate point is re-probed under
repeated split seeds to establish the probe's noise floor first
([`probe-noise-floors`](../lab/findings/probe-noise-floors.md)).

What it establishes:

1. Reconstruction cosine does not transfer across sources, so a quality
   target must come from the task
   ([`cosine-not-transferable`](../lab/findings/cosine-not-transferable.md)).
2. Tessera is downstream-transparent above ~1 b/px on point-thematic
   tasks, degrades faster than AlphaEarth below ~0.5 b/px, and the
   fine-scale detail the high rungs buy is unmeasured by the current
   suite
   ([`tessera-transparency-point`](../lab/findings/tessera-transparency-point.md)).

Two consequences for how a source is specified.

*The transparency point is a recommendation, not a ceiling.* These tasks
stop improving around the transparency rate, which is where a user doing
point-thematic work should be pointed. It does not make the rest of the
ladder wrong: the right rung depends on the task, and the benchmarks
that exist today do not exercise the fine-scale detail the high rungs
buy. What it does rule out is transplanting a cosine target from one
source to another.

*L is measurably over-provisioned.* The latent d95 (PCA of the trained
model's latents, per tier) sits several
times below the deployed latent widths
([`cosine-not-transferable`](../lab/findings/cosine-not-transferable.md)
has the numbers), so the widths could narrow before the probes would
notice.

**The gap that matters.** All five benchmarks are coarse thematic labels
at a single point, the regime AlphaEarth was designed for. Tessera's
per-pixel information may well pay on tasks with finer spatial or
temporal structure, and none are in the suite. Adding a fine-spatial or
temporal benchmark is the highest-value addition to this axis; the
use-case entry in `lab/TODO.md` depends on the same addition.

### A quality target for Sentinel-2, from practice

Reconstruction cosine and R2 are training signals, not acceptance
criteria. For S2 the criterion a remote-sensing user applies is
**absolute reflectance error per band, relative to that band's mean**:

- **< 1% on the 10 m bands** is the rough goal.
- **1-2% is borderline** -- about the level of Landsat/Sentinel
  harmonization differences, so a codec sitting there is adding error
  comparable to a known cross-sensor bias.

This is the number to report alongside R2 for any S2 model, because the
two can move apart: R2 is computed in per-window standardized space,
where a band's error is measured in units of its own spread, while the
practitioner reads error against the band's absolute level. See
`docs/FINDINGS.md` on uniform losses allocating error by dynamic range.

## The five-source measurement table

`analysis/data_probe.py` and `analysis/pyramid_sizing.py`, W=64, grids
1/4/16. This is the output the rest of the document consumes. Every row
comes from a measurement listed in sections 1-3: bands and valid are
basic properties from `data_probe` (section 1), tier-0 energy, the
per-tier d95 and the sub-4px residual share come from
`pyramid_sizing`'s energy decomposition (sections 1-2), the lossless
floor and prediction savings from the neighbour-predictive entropy
(section 2), magnitude spread from the normalisation sensitivity
(section 1), and the autocorrelation range from section 2. The d95 row
is the *data d95*: PCA of what each tier would transmit (block means
and residual block means), measured before any model exists. It is a
different quantity from the *latent d95* quoted in section 4 and the
derivation chain, which is PCA of a trained model's latents; the data
d95 is a design input, the latent d95 a diagnosis of a finished model,
and the gap between them measures how efficiently the encoder used its
width.

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
integer metre, uint16 DN, **dB at 0.1 dB** for SAR) and is printed with
the figures. On S2 the `left` predictor beats `(left+up)/2`, the reverse
of every other source, because the 20 m and 60 m bands are
nearest-upsampled and the horizontal neighbour is often an exact
duplicate. Entropies are order-0 per band, so every floor is
conservative.

The `residual <4 px` row is a statement about where residual *energy*
sits in scale, not about what the codec can represent: the fine tier's
tokens decode through a head that emits every pixel of their cell
individually, so sub-cell structure is reachable — it is simply the part
the model has the least capacity to spare for.

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

Ordered so each step consumes the previous ones. The chain is the
ambition, not the current state: the evidence column records how much
of each step is established, and several are still habit.

| # | parameter | derived from | current evidence |
|---|---|---|---|
| 0 | **objective** | use-case: task eval, band subsetting | point-thematic transparency at ~1 b/px, cos ~0.93; cosine is NOT transferable across sources; fine-scale use unmeasured |
| 1 | **W** (window) | autocorrelation range; between/within variance vs W | 64 px, unjustified |
| 2 | **P**, **G** (tiers, grids) | power spectrum; autocorrelation scale breaks | fine cell 4 px matches lag-0.5 point |
| 3 | **L** (latent dims/tier) | data d95; ~4 bits/dim saturation | latent d95 51/91/71 of 512/512/256 on Tessera — over-provisioned several fold |
| 4 | **K** | sample-budget: batch x cells x window / K >= 450 | K=256 verified healthy |
| 5 | **N** (levels) | per-level residual decay to a target rate | tier 2 not saturated at 20 |
| 6 | **X** (attention) | ceiling vs capacity; where the ceiling stops moving | unmeasured |
| 7 | **tricks** | mean removal, gain, conditioning | see below |

**Trick selection, from measurements we already have:**

- **Transmit the patch mean** when it is a large share of energy *and*
  cheap. Tessera: 85.5% of per-window energy, and ~60x more
  rate-efficient than pyramid bits at the rate in the table above.
  **Yes.**
- **Do not force the std.** Matching it *hurts* (table above) because
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

## What to build next

Forward work is queued in `lab/TODO.md`. The gaps this document's
status columns mark [missing] are, in the order they block the
derivation chain: a fine-spatial or temporal downstream task, the power
spectrum and patch-size sweep (the principled route to P, G and W,
which are currently inherited by habit), band cross-correlation with
global pixel PCA (the L floor), and a quantization sensitivity sweep
(closes `book_bits`).
