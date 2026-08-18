# Analysis suite: understanding a new data source

An inventory of the diagnostic tooling in `analysis/`, organised by the
question each tool answers rather than by when it was written. The intent
is a repeatable procedure for onboarding a new source: characterise the
data, bound what is achievable, then locate where a trained codec is
actually losing.

Two eras are present. **Legacy (2026-08-11/12)** is 34 scripts from the
codebook-fitting work, nearly all hardcoded to one AE/w32hd latent store;
they contain reusable ideas but are not portable as-is. **Current
(2026-08-13 onward)** is 21 scripts, mostly `--ckpt`-generic, written
during the Tessera stats-conditioning investigation.

Findings are recorded next to each tool because a diagnostic is only worth
keeping if you remember what it once told you.

---

## A. Characterise the source, before any model

| tool | question | portability |
|---|---|---|
| `patch_intrinsic_dim.py` | Within one patch, is the pixel variation a few linear directions or a few prototype "species"? Local vs pooled. | **multi-source** (`--source tessera\|ae\|s2`) |
| `patch_moments_pca.py` | Across the dataset, how low-rank are the per-patch mean and std vectors? | tessera-hardcoded |
| `moment_sample_size.py` | Is the sample big enough — separating estimator convergence from *effective* N under correlated sampling. | generic |

**What they established.** Within-patch structure is linear, not
categorical: 8 PCA directions capture 77.5% of a Tessera patch's variance
where 8 k-means prototypes capture 49.8% (sub-pixel mixing is linear).
Effective dimensionality is a property of the *representation*, not the
ground — participation ratio 1.3/13 for Sentinel-2, 5.9/64 for AlphaEarth,
6.3/128 for Tessera, on the same landscapes. And the pooled/local ratio
orders how much per-patch conditioning can ever buy: 0.92× (S2), 1.44×
(AE), 2.90× (Tessera).

Across the dataset the moments are **not** low-rank: 107 of 128 components
for 99% of the mean's variance. `moment_sample_size.py` separates two
worries that landed on opposite sides — the Gaussian comparator was
converged at 4k chips while AE *training* was not (10× data improved every
config by ~14%).

**Still ad-hoc, worth promoting:** per-band dynamic range and nodata rates
(`|x|` spans only 1.52× p1→p99 for Tessera but the mean-removed residual
spans 4.16×; S2 DN==0 rate is 0.0000%), and the mean/std anti-correlation
(−0.533) that says coherent patches have long means and little spread.

## B. Bound what is achievable, before training a codec

| tool | question |
|---|---|
| `pixel_rd.py` | Single-pixel R-D: bottleneck dim × rate, FSQ vs RVQ, with the encoder *trained with* its quantizer. |
| `moment_rd.py` | R-D of the 256-d moment vector end-to-end, against the Gaussian bound by reverse water-filling. |
| `moment_hybrid.py` | Nonlinear stage first, PCA + entropy-coded scalar quantization on the residual. |
| `hybrid_compare.py` | Compare arms at **matched distortion**, interpolating in log-error space where R-D curves are straight. |
| legacy: `compand.py`, `lattice.py`, `pladder.py`, `budget*.py`, `waterfill.py` | transform-coding upper bounds, storage-budget allocations, greedy water-filling |

**What they established.** A learned bottleneck beats a linear-Gaussian
coder by 1.6–1.7× at 128 bits, converging toward parity only at ~600.
Bottlenecks saturate at **~4 bits per dimension**. The hybrid (small
nonlinear stage, then transform coding) wins only above ~450 bits — and
its premise is half wrong: the stage-1 residual is *whiter* (participation
ratio 21 → 121), which removes transform-coding gain rather than enabling
it; the hybrid wins anyway because stage 1 strips 95% of the energy for 32
bits.

**Methodological note that cost a day:** every comparison here is
confounded by optimisation effort. The AE arm improved with more data
(+14%) and more steps (+10%), moving the crossover 200 → 300 → 500 → 600
bits. Always match training budget across arms before believing a
crossover.

## C. Where is a trained codec actually losing?

| tool | question |
|---|---|
| `tier_rate_attrib.py` | Per-tier attribution of the quantization gap: decode every {quantized, int8-passthrough} combination against what it costs. |
| `tier_collapse.py` | Has a tier's latent collapsed to a constant? Across-window spread, effective rank, per-level RVQ contributions, per-level perplexity. |
| `tier0_spatial.py` | How much spatial variation does a depth tuple encode — magnitude *and* alignment *and* autocorrelation, so "too little" is distinguished from "wrong". |
| `tier0_moments.py` | What does the coarse tier do with the patch mean and std specifically? |
| `residual_struct.py` | What structure survives in the reconstruction error: spatial autocorrelation vs the source, effective rank, energy tail. |
| `moment_match.py` | What is a transmitted moment channel *worth* per pixel, and at what rate does it stop paying? |

**What they established.** Essentially the whole quantization gap is the
fine tier (99.6%); the coarse tiers are 2.5%, so sending them raw is poor
value (+16.3 b/px buys +0.00013 cosine). The coarse tier can silently
collapse — 1.16 effective dimensions of 512, first codebook level at
perplexity 1 — while full-depth metrics look healthy, because the fine
tiers carry everything. The codec systematically **under-produces** spatial
variance (73% of the source's even at full depth) and stays smoother than
the source at every lag.

The moment channel is worth +0.0069 cosine at full depth for 0.188 b/px,
~60× more rate-efficient than pyramid bits — but **only the mean**:
matching the std *hurts* (0.9700 → 0.9678), because the decoder's output is
already smoother than the truth, so rescaling amplifies error instead of
restoring detail. Below 4 bits/band the channel is worse than nothing.

## D. Quantizer health

| tool | question |
|---|---|
| `ema_vs_tree.py` | Learned EMA codebooks vs a balanced PCA tree at fixed K. |
| `level_fit.py` | Per-level iterative fitting against a frozen encoder, with held-out tracking. |
| `refit_sweep.py`, `refit_dryrun.py`, `store_refit.py` | How much data a refit phase needs; dry-run on a trained checkpoint. |
| `tsvq_probe.py` | What O(log K) tree-structured encoding costs vs exhaustive search. |
| `pcdir_probe.py` | Is the within-cell principal direction estimable at all (BBP transition), or is the power iteration fitting noise? |
| `plot_reset_diag.py` | Per-level reset diagnostics from a run's wandb history. |

**What they established.** There is a **sample-budget crossover** near
~450 vectors per codeword below which refitting is worse than not
refitting; K=16384 at 8 levels sits far under it (18–54 vectors/code) and
loses 90% of its codebook. K=256 × 20 levels makes the pathology
structurally impossible — dead-code resets find 1–5 codes instead of
6,500–8,200 — at 12× smaller download, for equal R-D.

## E. Training diagnostics — mostly still ad-hoc

These ran inline and were never committed. They are the ones most worth
promoting, because each caught a real defect:

- **Grad-norm outlier detection.** Median 0.08, p99 0.51, but genuine
  spikes reach 3.85 (48× median). No clipping existed at all; an unclipped
  spike collapsed a run (centred ceiling cosine 0.807 → 0.036) and Adam
  never recovered.
- **Loss-spike attribution.** Is a spike the schedule, the data, or an
  amplitude outlier? Decided by whether *relative* error moves with
  absolute: if `vec_rel_l2` and cosine move together it is content, not
  scale.
- **Batch/LR scaling efficiency.** d(log loss)/dstep per batch size, and
  per sample. Established that under AdamW a 4× batch mostly *denoises* a
  same-sized step, and that with weight decay on all params in a
  LayerNorm-heavy trunk the effective LR goes as √(lr·wd) — so number of
  updates is the undiluted lever.
- **Checkpoint introspection.** `initialized` flags, step counters,
  codebook std — this is what caught codebooks being initialised 18k steps
  early by a figure-drawing call that forgot `model.eval()`.
- **Cross-run comparison from wandb**, and **relative gap-closure trends**
  — `Δ(1−ccos)/(1−ccos)` per 1k steps, which is the right unit for a
  bounded metric and reveals gradients that absolute deltas flatten.

## F. Infrastructure

`build_store_tessera.py`, `build_latents.py`, `build_full*.py`,
`verify_latent_store.py`, `fit_corpus_books.py`.

---

## Gaps for the next stage

1. **Portability.** Only `patch_intrinsic_dim.py` takes `--source`. The
   source-characterisation tools (A) should all run on any store via the
   `Source` abstraction rather than inlining dequantisation.
2. **The training diagnostics (E) are not tools yet** — they are shell
   history. They are also the ones that caught the most defects.
3. **No single entry point.** There is no "characterise this new source"
   driver that runs A, then B, and emits a report.
4. **Metric discipline is implicit.** Relative-gap framing, matched-budget
   comparison, and matched-distortion interpolation are conventions that
   currently live in reviewers' heads.
