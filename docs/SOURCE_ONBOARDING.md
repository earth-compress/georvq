# Onboarding a new source: from measurements to an architecture

The goal is to be able to finish this sentence from evidence rather than
habit:

> *"Build a pyramid with **L** latent dims on **P** tiers at grids **G**,
> codebooks of size **K** for **N** levels, window **W**, **X** attention
> blocks, with these tricks."*

Every measurement below exists to constrain one of those. Anything that
does not is diagnostics, not design, and belongs in `analysis/README.md`.

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
repeated split seeds, giving a noise floor of +/-0.017 on LUCAS and
+/-0.023 on LCMAP — so LCMAP cannot resolve anything below ~0.05 and its
"compression beats the original" readings are artefacts.

**What it establishes, and it is the most consequential result we have:**

1. **Cosine does not transfer across sources.** At matched or slightly
   worse reconstruction cosine, the Tessera codec costs LESS downstream
   accuracy than the AlphaEarth codec. LUCAS land cover: AE at cos 0.9532
   loses 0.043 balanced accuracy; Tessera at cos 0.9475 loses 0.005.
2. **Tessera is downstream-transparent above ~1 b/px, at cos ~0.93.**
   AlphaEarth needs cos ~0.99 for the same verdict.
3. **Below ~0.5 b/px Tessera degrades about twice as fast** (LUCAS
   -0.108 vs -0.052), established on LUCAS only; LCMAP is noise-dominated.
4. **The mechanism.** Tessera carries ~2.5x more independent variation per
   pixel — per-tier PCA d95 of **51/91/71** against tier dims 512/512/256,
   versus AlphaEarth's **19/34/28** — much of it fine-scale and weakly
   correlated with land cover. Discarding it costs cosine but not task
   accuracy. AlphaEarth's field is smoother and more nearly low-rank, so a
   given cosine error eats the dimensions the probes actually read.

**Two consequences for how a source is specified.**

*The transparency point is a recommendation, not a ceiling.* These tasks
stop improving at ~1 b/px, which is what a user coding for point-thematic
work should be pointed at — around `20-20-1` on the released ladder. It
does not make the rest of the ladder wrong: the codec ships a range
(0.23–10.9 b/px) precisely because the right rung depends on the task,
and the benchmarks that exist today do not exercise the fine-scale detail
the high rungs buy. What it does rule out is transplanting a *cosine*
target across sources: AlphaEarth's 0.998 is not a meaningful
specification for Tessera, whose probes are already indifferent at
0.93.

*L is measurably over-provisioned.* d95 of 51/91/71 in tiers of
512/512/256 says the latent widths could be several times narrower before
the probes would notice.

**The gap that matters.** All five benchmarks are coarse thematic labels
at a single point — the regime AlphaEarth was designed for. Tessera's
per-pixel information may well pay on tasks with finer spatial or temporal
structure, and **none are in the suite**. So "transparent above 1 b/px" is
established for point-thematic use and simply unmeasured for anything
else. That is an argument for adding a fine-spatial or temporal
benchmark — the highest-value addition to this axis — not for assuming
the high-rate end is wasted.

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
1/4/16. This is the output the rest of the document consumes.

| | DEM | S1 RTC | S2 L1C | AlphaEarth | Tessera |
|---|---:|---:|---:|---:|---:|
| bands | 1 | 2 | 13 | 64 | 128 |
| valid | 100% | 100% | 99.5% | 81.5% | 95.9% |
| tier-0 energy | 99.8% | 97.2% | 97.1% | 97.7% | 87.3% |
| d95 tier 0/1/2 | 1/1/1 | 2/2/2 | 3/3/3 | 7/27/30 | 32/75/86 |
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

Ordered so each step consumes the previous ones.

| # | parameter | derived from | current evidence |
|---|---|---|---|
| 0 | **objective** | use-case: task eval, band subsetting | point-thematic transparency at ~1 b/px, cos ~0.93; cosine is NOT transferable across sources; fine-scale use unmeasured |
| 1 | **W** (window) | autocorrelation range; between/within variance vs W | 64 px, unjustified |
| 2 | **P**, **G** (tiers, grids) | power spectrum; autocorrelation scale breaks | fine cell 4 px matches lag-0.5 point |
| 3 | **L** (latent dims/tier) | global PCA d95; ~4 bits/dim saturation | measured d95 51/91/71 of 512/512/256 — over-provisioned several fold |
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

## What to build next, in priority order

1. **A fine-spatial or temporal downstream task (§4).** The existing
   suite is five coarse thematic point labels, so it cannot see the
   per-pixel detail that is Tessera's whole advantage. Everything else on
   this list optimises against a target that may not be the real one.
2. **Power spectrum + patch-size sweep (§2)** — the only principled route
   to P, G and W, which are currently inherited by habit.
3. **Band cross-correlation + global pixel PCA (§1)** — cheap, and gives
   the L floor directly.
4. **Quantization sensitivity sweep (§3)** — closes `book_bits`.

Consolidating the remaining shell diagnostics (codebook health, latent
health, training dynamics) is *lower* priority than any of these: they
tell us whether a run is healthy, not what to build.
