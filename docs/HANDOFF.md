# Handoff: Tessera stats conditioning + the source-onboarding suite

State as of cumulative step ~29,000 of the stats-conditioning run. Written
so the next session can continue without re-deriving anything. Companion
docs: `SOURCE_ONBOARDING.md` (measurement -> architecture),
`ANALYSIS_SUITE.md` (tool inventory), `shell_analysis_catalogue.md`
(triaged ad-hoc diagnostics).

---

## 1. What is running

`configs/tessera_vq8_k256_statscond_resume.yaml` on **GPU0**.
Log: `logs/tessera/vq8_k256_statscond_r5.log`. wandb run `jy4xh0f2`.

Step numbering restarts per resume. **Cumulative = resume step + 19,000.**
Target 71,000 cumulative (52,000 in this segment).

| checkpoint | where | note |
|---|---|---|
| step 19,000 | `checkpoints/_handoff/statscond_step19000.pt` | what the resume loads; do NOT delete |
| step 5,000 | `checkpoints/_handoff/statscond_step5000.pt` | pre-collapse, clean restart point |
| live | `checkpoints/tessera_vq8_k256_statscond/` | overwritten every 2,500 steps |

**The run_tag trap:** the resume reuses its `run_tag`, so its own saves
overwrite `model_best.pt` in the live dir. That is why `init_from` points
at a copy under `_handoff/`. Keep it that way.

**It has been OOM-killed twice** — by the kernel, not by any fault: a
114 GiB CPU-resident pool made it the largest RSS on a box shared with two
users running 200-300 processes each. Pool is now halved (9,575 slots,
~57 GiB). If it dies again with no traceback, check `free -g` before
anything else. `analysis/../launch_when_ram.sh` pattern: wait for
>=75 GB available twice in a row, then launch.

## 2. Where the science stands

**The hypothesis.** Transmit the per-window mean as a side channel so the
coarse tier stops spending its 160 bits locating the patch in embedding
space and can encode shape instead.

**It worked, after a detour.** Coarse-tier structure (`ccos_coarse_only`)
went 0.003 -> 0.352 between steps 12k and 22k, and `coarse_only` cosine
(0.9363) is now above the block-mean baseline (0.9223) where it began
*below* it. Against the k256 baseline at 54k steps: coarse-tier structure
**0.336 vs 0.144**, `ccos_full` **0.805 vs 0.785**, full depth 0.9792 vs
0.9726.

**Three defects found and fixed** (all committed, all would have recurred):
- No gradient clipping anywhere. An unclipped spike collapsed a run
  (centred ceiling cosine 0.807 -> 0.036) and Adam never recovered.
  `optim.grad_clip` now defaults to 1.0.
- `recon_figure` was `@torch.no_grad()` but never `model.eval()`, so
  drawing a figure INITIALISED the codebooks — from 8 val windows, 18k
  steps early, by the wrong method.
- The VQ-free warmup decoded at full depth, so a long warmup let the
  coarse tier atrophy to 1.16 effective dims of 512 with its first
  codebook level at perplexity 1. `vq.warmup_truncate` (default on) now
  keeps a random prefix of tiers and zeroes the rest.

**The uncomfortable finding.** `tessera_downstream_interim.pdf` shows
Tessera is downstream-transparent above **~1 b/px at cos ~0.93** on the
five benchmark tasks. This run sits at **10.85 b/px, cos 0.979** — about
10x past transparency — so the gains above may be downstream-invisible.
But all five benchmarks are coarse thematic point labels, and Tessera's
advantage is fine-scale detail, of which **46% of the within-patch
residual sits below 4 px and is unreachable by this pyramid at any
depth**. A fine-spatial or temporal benchmark is the highest-value
missing measurement in the project.

## 3. The five-source measurement table

`analysis/data_probe.py` and `analysis/pyramid_sizing.py`, W=64,
grids 1/4/16.

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

Notes that matter when re-running: the lossless unit is a judgement call
per source (int8 as stored / integer metre / uint16 DN / **dB at 0.1 dB**
for SAR) and is printed with the figures. On S2 the `left` predictor beats
`(left+up)/2`, the reverse of the others, because 20m/60m bands are
nearest-upsampled so the horizontal neighbour is often an exact duplicate.
Entropies are order-0 per band, so every floor is conservative.

## 4. Architecture recommendations derived from the above

| | DEM | S1 | S2 | AE | Tessera |
|---|---|---|---|---|---|
| transmit mean | yes | yes | yes | yes | **yes** (0.188 b/px, +0.0069 cos) |
| per-band std | no | no | no | n/a | **no — measured harmful** (0.9700->0.9678) |
| scalar gain | already does | likely | likely | none needed | **likely, untested** (residual spans 4.16x) |
| latent dims t0/t1/t2 | 16/32/32 | 16/32/32 | 32/64/64 | 32/64/96 | **64/128/128** (deployed 512/512/256) |
| K per tier | 64/256/1024 | 64/256/1024 | 64/512/4096 | 64/1024/8192 | **64/1024/16384** |
| train metric | MSE on standardised relief; report R2 | MSE in dB + multi-look | per-band weighted MSE + spectral angle | cosine | MSE on mean-removed field; report **ccos** |

**Two rules that generalise.**

*K is derived, not chosen:* `batch x cells x ema_window / K >= ~450`
vectors per codeword. At batch 1024, decay 0.98 that gives ceilings of
113 / 1,820 / 29,127 for grids 1/4/16 — so codebooks should be graded
UPWARD with tier. The original 2048/8192/16384 gave tier 0 just 25
vectors/code and it collapsed; even uniform K=256 leaves tier 0 at 200,
still under the crossover.

*Cells dominate rate:* one tier-2 level costs ~64x one tier-1 level, so
depth should be allocated inversely to cell count. Tier 2's residual is
still decaying at L20, so N there is rate-limited, not saturation-limited.

## 5. Open items, in priority order

1. **A fine-spatial or temporal downstream benchmark.** Everything else
   optimises against a target that may be the wrong customer.
2. **Test the scalar gain channel.** ~0.002 b/px, addresses a measured
   4.16x dynamic range in the mean-removed residual. Never tried.
3. **Retrain with the corrected geometry** — narrower latents
   (64/128/128), graded codebooks (64/1024/16384), `warmup_truncate` on,
   grad clipping on. All the fixes are committed but no run has used them
   together from scratch.
4. **Re-run `tier_rate_attrib.py`** once the coarse tier settles; the
   99.6%-of-gap-is-tier-2 result was measured at step 19,000 when tier 0
   was still dead.
5. Consolidate the remaining shell diagnostics (codebook health, latent
   health, training dynamics) — lowest priority; they say whether a run is
   healthy, not what to build.

## 6. Metric discipline (learned the hard way)

- **Use relative gap closure** `d(1-m)/(1-m)`, not absolute deltas, for
  any bounded metric. It exposed a depth-graded improvement (coarse 20%,
  mid 9%, full 0.9% per 1k steps) that absolute deltas flattened.
- **Match training budget before believing a crossover.** The AE-vs-linear
  crossover moved 200 -> 300 -> 500 -> 600 bits as data and steps were
  added; every move favoured the arm being under-resourced.
- **`cos` is inflated by the transmitted mean.** Use `ccos`. `coarse_only`
  read 0.9154 while contributing nothing.
- **Don't extrapolate a flat stretch.** I predicted the coarse tier would
  reach 0.11; it reached 0.35 because the recovery was a phase
  transition, not a line.
