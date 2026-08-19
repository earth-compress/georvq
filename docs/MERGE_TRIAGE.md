# Tessera branch: merge triage

What `worktree-tessera-source` contains, and what of it should survive
into `main`. Written against `6ed40f4` vs `main` at `d7dda7d`.

**Scope:** 108 commits, 205 files, +19,121 / −204 lines.
**State:** worktree clean; `pytest -m "not slow and not gpu"` → **145
passed, 5 deselected** in 34.5 s; the golden-checkpoint gate
(`tests/test_compat.py`) passes, so the AlphaEarth release still decodes
bit-faithfully. Every new code path in `codec.py` is config-gated with
defaults off, which is why it does.

The keeper model is **`configs/tessera_vq8_k256_statscond.yaml`** —
3 tiers at grids 1/4/16, latents 512/512/256, **K=256 × 20 levels**,
int8 QAT codebooks, patch-mean side channel, geometry inherited from
`tessera_stage1_4x`. Everything below is categorised by whether it
serves that model, serves the *next* one, or served neither.

Headline numbers for it: 10.85 b/px, cos 0.979, ccos 0.805, coarse-tier
structure 0.336 (vs 0.144 for the K=256 baseline without the mean
channel).

---

## A. Merge — generic, tested, defaults-off

### A1. The correctness fixes

These are the highest-value commits on the branch and none of them are
Tessera-specific. Each one was a live defect on `main`:

| fix | commit | what it cost |
|---|---|---|
| **No gradient clipping existed at all.** `optim.grad_clip` now defaults to 1.0 | `5e413ee` | an unclipped spike collapsed a run (centred ceiling cos 0.807 → 0.036); Adam never recovered |
| `recon_figure` was `@torch.no_grad()` but never `model.eval()` — **drawing a figure initialised the codebooks**, from 8 val windows, 18k steps early | `4bff15f` | silently wrong books for an entire run |
| VQ-free warmup decoded at full depth, so a long warmup let the **coarse tier atrophy** to 1.16 effective dims of 512 (`vq.warmup_truncate`, default on) | `9255b58` | the richest tier at step 5k became the deadest |
| `ema_count` seeded as ones made **`ema_usage` a constant 100%** | `83f4183` | the codebook-health metric read healthy unconditionally |
| `reset_mode=split` was a **silent no-op on the QAT and gain-shape paths** | `8b6a077` | A/B tests measured nothing |
| FSQ delivered **L−1 levels for even L**, and only ONE at L=2 | `b2e83da` | every FSQ rate point mis-stated |
| stage-2 must quantize the **normalized** latent geometry (`model.normalize_latents`) | `acda544` | fitted books handed latents at a different scale |
| quantized validation ran **while codebooks were unfitted** | `9cdf955` | meaningless early val curves |
| `level_fit` convergence policy **never fired** | `612bd62` | — |
| `load_partial` zero-padded `to_patch` when it should not | `d1b0254` | — |
| dist: rank bound to device explicitly; 90-min NCCL timeout | `d0725c5`, `e31c330` | barrier hang / rank-skew crash |

**Merge all of these unconditionally.** They are bug fixes with tests.

### A2. Tessera source support (required for any release)

`georvq/sources/tessera.py` (81), `scripts/build_store_tessera.py` (242),
`tests/test_tessera.py` (228), `configs/tessera_pyr64.yaml`,
`scripts/baseline_tessera_vq.py`. Registration in
`georvq/sources/__init__.py` is two lines. Includes the exact
direction↔int8 reparametrization (`868a208`), modelling dequantized
vectors rather than unit directions (`bbe6bba`), and corrupt-scale-pixel
masking (`ce2d0b3`).

### A3. Quantizer health machinery

- **`reset_mode="split"`** (`georvq/model/vq.py`, `tests/test_reset_split.py`)
  — LBG distortion splitting instead of random reseeding. Note the care
  taken: the `ema_sse` buffer is `persistent=False`, so the state-dict
  layout and every existing checkpoint's strict load are untouched. The
  second iteration (splitting the centroid into c ± εδ rather than onto
  the worst-fitting point) is the one that works; the first was
  measured to evaporate ~100% of its gain before the next reset.
- **`georvq/model/vqinit.py`** (154) — balanced-PCA-tree codebook init,
  per-tier init budgets (`init_per_code`), `mature_warmstart`. Fixes the
  coupling where at `kmeans_sample=32768` the coarse tier saw 16 vectors
  per centroid while the finest saw 512 and cost 12 GiB.
- **Per-level reset diagnostics** (`d090b3d`, `88395b3`) — including the
  "report healthy levels too" fix, without which a level that never
  loses a code is indistinguishable from a broken one.

### A4. `georvq/model/vq_quant.py` — b-bit QAT codebooks

132 lines + `tests/test_vq_quant.py`. Projected-EMA quantization-aware
codebooks; int8 books are ~75 MiB against 302 MiB fp32 for the same
model. The release model uses `book_bits: 8`. Also
`QuantizedBookVQ`'s `index_add` EMA (`563d4f4`), which replaced a dense
one-hot that was 16.5 GiB at K=16k × 512k N.

### A5. `georvq/model/stats_cond.py` — the patch-mean side channel

93 lines + `tests/test_stats_cond.py` (195). The branch's clearest
scientific win: **+0.0069 cosine for 0.188 b/px**, ~60× more
rate-efficient than pyramid bits, and it moved coarse-tier structure
from 0.003 → 0.352. Rate is *charged*, not free (`tuple_bits(side_bits=)`)
— the ring-space lesson that an uncharged stats channel silently became
34% of the bitstream is encoded in the config docstring.

`use_std` is validator-blocked, correctly: matching the std was measured
to **hurt** (0.9700 → 0.9678).

### A6. `depth_sampling: path`

Nested depth sampling starved the deep fine-tier books —
P(active) = 0.006 for tier-2 level 14, i.e. those books saw 0.6% of
rows. Path sampling raises the same level to 0.197. Small change,
large effect, generic.

### A7. Multi-GPU + throughput

`georvq/train/dist.py` (139) + `tests/test_dist.py` (122);
`train.compile` (trunk-layer forwards only, so state-dict names are
untouched); decoder grad-checkpointing; one-deep batch prefetch thread;
`batch_ramp`; `optim.schedule: constant` (so a run can be stopped
without leaving a cosine schedule half-applied); `passthrough_frac`
(without it a stage-2 decoder specialises to quantized inputs and its
continuous ceiling decayed 0.9899 → 0.9830 over 25k steps).

### A8. Downstream evaluation

`georvq/eval/tessera_extract.py` (219),
`scripts/eval_build_dataset_tessera.py`, `eval_rate_sweep.py
--restrict-parquet` for matched-point cross-source A/B. This produced
the single most consequential result on the branch — **cosine does not
transfer across sources**: at matched reconstruction cosine the Tessera
codec costs *less* downstream accuracy than the AlphaEarth codec (LUCAS
land cover: AE at cos 0.9532 loses 0.043 balanced accuracy, Tessera at
cos 0.9475 loses 0.005). The tile-blocked split and repeated-seed noise
floor (±0.017 LUCAS, ±0.023 LCMAP) are the right protocol and should
become the standard one.

### A9. Tests

14 new test files. Keep every one whose subject survives triage; drop
`test_vq_gainshape.py`, `test_refit.py` and (if the suffix path goes)
`test_suffix.py` with their code.

---

## B. Keep the lesson, delete the code

These are honest, well-measured **negative results**. The finding is
worth more than the implementation, and `docs/FINDINGS.md` already
records most of them. Move the rest to `docs/PARKED.md` and delete.

| what | size | verdict |
|---|---|---|
| **Gain-shape VQ** (`vq_gainshape.py` 250, `test_vq_gainshape.py` 163, `probe_gain_shape.py` 119) | ~530 lines | **Loses at matched rate**: plain VQ 0.2251 vs 0.2537 (K=4096 + 2 gain bits) and 0.3087 (K=1024 + 4). Its premise does not hold. **No config uses it.** Delete. |
| **Periodic offline refit** (`train/refit.py` 245, `test_refit.py` 119, 6 config knobs, `configs/tessera_vq8_refit.yaml`) | ~450 lines | Superseded by the branch's own conclusion: *fit-once-and-freeze is never worse*. A fresh 8-iter k-means beat 60k EMA steps end-to-end (0.9766 vs 0.9744). The keeper run does not use it. Delete; keep `freeze_books`. |
| **`train/drift.py`** encoder-drift probe (151) | 151 lines | Served its purpose — it is what turned EMA decay from a hyperparameter into something derivable. One config uses it. **Judgement call:** small and generic; keeping it is defensible, but it is diagnostic, not design. Lean delete, record the derivation. |
| **`analysis/cb_challenge_agent/`** (19 files) | ~1,100 lines | Produced the single best lesson on the branch — *codebook perplexity is gameable*: 11.97 of 12 bits with all 4096 codes used at **0.014% of energy explained**, decoded cos 0.45 vs 0.98 for k-means. Judge codebooks on rate–distortion, never on usage. That sentence is already in FINDINGS.md. Delete the 19 files. |
| **Cross-fitted refinement levels** (`analysis/crossfit.py`) | — | +33% on a collapsed tier, **−13% on a healthy one**. Negative. Lesson only. |

---

## C. Delete — abandoned architecture lines

Six architecture families were explored; **one won**. The keeper geometry
is `tessera_stage1_4x` → `tessera_vq8` → `tessera_vq8_k256` →
`tessera_vq8_k256_statscond`. The rest are dead ends with their
verdicts already recorded:

- **`uf` (4-tier ultrafine, 2 px)** — killed by tier-redundancy
  collapse: the fine tier carried **0.2%** of its variance within a
  window while consuming ~62% of the rate budget. Healthy (96.8–97.9%)
  in every 3-tier model; collapses only once the ultrafine tier is
  added.
- **`w32` / `w32hd`** — the 0.99-ceiling probes.
- **`slim_eff` / `4x_eff`** — the efficiency trunks.
- **`patch4` / noise-floor brackets** (incl. the three `ae_stage1_patch4*`
  configs) — single-vector patch-AE probes.
- **`4tier_mk`** — matryoshka tier-truncation sampling.
- **`2x` / `4x` ladder rungs** — superseded by their own endpoint.

**Configs: 31 added, keep 4.** Keep `tessera_pyr64.yaml`,
`tessera_vq8.yaml`, `tessera_vq8_k256.yaml` (the baseline the release is
measured against) and `tessera_vq8_k256_statscond.yaml`. Delete the
other 27, including the three resume/r2 variants — a resume config is a
session artefact, and its content (the run_tag overwrite trap, the pool
sizing) belongs in prose, not a checked-in YAML.

**Code that exists only for the dead architectures:**

| module | used by | recommendation |
|---|---|---|
| `model/suffix.py` (149) + `tokenizer/decoder.suffix_dim` + `test_suffix.py` | `uf`, `slim_eff`, `4x_eff` only | **Delete.** The narrow-suffix stream is the most complex addition on the branch and serves only abandoned models. Park the blocked-halo result (7× faster than per-query gather) in PARKED.md. |
| `model/attnmask.py` (43) + `model.local_attn` + `test_attnmask.py` | 4 dead configs | **Keep** — small, generic, tested, and a plain ViT feature the next (narrower) model may want. |
| `tokenizer/decoder.late_join` + `test_late_fusion.py` | 6 dead configs | **Keep** — same reasoning, ~equally cheap. |
| `decoder.head_dw` | 1 dead config | Delete or keep; trivial either way. |
| `sigma.truncate` / `kept_max` + `test_sigma_trunc.py` | `4tier_mk` only | Keep — it is the correct matryoshka semantics (zero dropped tiers, don't just noise them) and stage-1 still uses sigma mode. |
| `data.limit_chips` | 3 probe configs | **Keep** — 3 lines, and data-scaling probes are a permanent need. |

**`reports/` — 34 committed binaries.** Delete 18 `reset_*.png`
diagnostics (per-event figures from one run), `branch_ladders.png`,
`embedded_ladders.png`, `level_fit/` (8 PNGs + a 1,192-line JSON),
`rd_compare.png`, `residual_diagnostic.png`, `ae_attention.png`. Keep
`ae_vs_tessera.pdf` only if the release doc cites it; otherwise it lives
in `logs/`. Committed run figures are exactly the junk to avoid.

**`analysis/` — 86 files, keep ~20.** `ANALYSIS_SUITE.md` already draws
the line: 34 "legacy" scripts hardcoded to one AE/w32hd latent store
(`budget*.py`, `bitdepth.py`, `masked.py`, `pca25.py`, `lvl4.py`,
`lvl8.py`, `sens.py`, `corr.py`, `wkm.py`, `cover.py`, `ladder.py`,
`embedded.py`, `branch.py`, `waterfill.py`, `hybrid.py`, `fit_sweep.py`,
`perlevel.py`, `rdcmp.py`, `hidden_test.py`, `challenge_eval.py`,
`ae_books.py`, `ae_cbopt.py`, `build_full*.py`, …). They contain reusable
ideas but are not portable as-is and their findings are written down.
Delete them.

---

## D. The thread worth continuing

**`docs/SOURCE_ONBOARDING.md` is the most valuable artefact on the
branch**, and it is exactly what you described wanting: the ad-hoc
analysis consolidated into a procedure for designing a codec for a new
source. It is honest about status (`[have]` / `[partial]` / `[missing]`
per measurement) and it already produced two rules that generalise:

- **K is derived, not chosen.** `batch × cells × ema_window / K ≥ ~450`
  vectors per codeword. K=16384 × 8 levels sat at 18–54 and lost 90% of
  its codebook; K=256 × 20 gives thousands, dead-code resets fell from
  6,500–8,200 per event to 1–5, at **12× smaller download for equal
  R-D**. Codebooks should be graded *upward* with tier.
- **Cells dominate rate.** One tier-2 level costs ~64× one tier-1 level,
  so depth allocates inversely to cell count.

Plus the five-source measurement table (DEM / S1 RTC / S2 L1C /
AlphaEarth / Tessera) — lossless floors 2.47 / 11.2 / 72.3 / 131 / 672
b/px, and the architecture recommendations derived from it.

**Merge this, and promote its ~14 supporting tools** out of `analysis/`
into a real package (`georvq/probe/` or a curated `analysis/` with a
README and `--source` support):

`data_probe.py`, `pyramid_sizing.py`, `patch_intrinsic_dim.py`,
`pixel_rd.py`, `moment_rd.py`, `moment_hybrid.py`, `hybrid_compare.py`,
`moment_sample_size.py`, `moment_match.py`, `residual_struct.py`,
`tier_collapse.py`, `tier_rate_attrib.py`, `tier0_spatial.py`,
`tier0_moments.py`, `tier_increment.py`, `run_compare.py`,
`level_fit.py`, `ema_vs_tree.py`, `tsvq_probe.py`, `plot_reset_diag.py`.

Its own stated gaps, in its priority order: a **fine-spatial or temporal
downstream benchmark** (all five current benchmarks are coarse thematic
point labels — the regime AlphaEarth was designed for, not Tessera's);
power spectrum + patch-size sweep (the only principled route to P, G, W,
currently inherited by habit); band cross-correlation + global pixel PCA;
quantization-sensitivity sweep.

`docs/ANALYSIS_SUITE.md` is the inventory behind it — merge, but fold
the "legacy era" section down to a paragraph once those files are gone.
`docs/shell_analysis_catalogue.md` (119 rows of triaged shell history) is
scaffolding: it has served its purpose, delete it after the promotions
it recommends are done.

---

## E. Gaps before "a released model for Tessera"

1. **There is no Tessera release layer.** AlphaEarth has `georvq/ae/`
   plus `docs/AE_RELEASE.md`; Tessera has neither. Per CLAUDE.md the
   source-specific release policy belongs in its own package with
   `georvq/inference/` staying source-agnostic. This needs writing.
2. **The run is mid-flight.** `tessera_vq8_k256_statscond_resume` is at
   ~29k of 71k cumulative steps on GPU0. The release waits on it. Note
   the run_tag trap: the resume reuses its
   `run_tag`, so `init_from` must keep pointing at the `_handoff/` copy.
3. **`docs/OVERVIEW.md`, `docs/ADDING_A_SOURCE.md`, `README.md` and
   `CLAUDE.md` were never touched** by 19k lines of new code. They are
   stale on: the Tessera source, `stats_cond`, `book_bits`, the new VQ
   init/reset modes, `dist.py`, and `depth_sampling`.
4. **The three newest findings are not in FINDINGS.md** — they postdate
   it. Most notably the **rotation + Lloyd refinement** result
   (`turboquant.py`, `refine_decode.py`): ~1 bit/dim on the post-RVQ
   residual closes **82% of the quantization gap at 36.8 b/px**, where
   sending tier 2 as raw int8 closes 99.6% but costs 128.9 b/px — 4/5
   of the benefit at 3.5× less rate, with **no additional trained
   codebooks** (a random seed and a Lloyd table fitted once). Random
   rotation beats PCA here because at fixed levels-per-dimension you
   want variance *equalised*, not concentrated. This is a genuine
   candidate feature, not just a lesson.
5. **`docs/HANDOFF.md` was not merged as-is (done).** It is a session
   document. Dissolve it: run state → deleted, science → FINDINGS.md,
   the measurement table and derived architecture → SOURCE_ONBOARDING.md,
   the metric-discipline section → FINDINGS.md or OVERVIEW.md (it is
   general and hard-won: use relative gap closure `Δ(1−m)/(1−m)` on
   bounded metrics; match training budget before believing a crossover;
   `cos` is inflated by the transmitted mean, use `ccos`; don't
   extrapolate a flat stretch).
6. **Open experiments worth naming in PARKED.md, not chasing now:** the
   scalar gain channel (~0.002 b/px against a measured 4.16× dynamic
   range in the mean-removed residual — never tried); a retrain with the
   corrected geometry (narrower latents 64/128/128, graded codebooks
   64/1024/16384) — every fix is committed but no run has used them
   together from scratch.

**The measurement to carry forward:** Tessera is downstream-transparent
above **~1 b/px at cos ~0.93** on all five current benchmarks. That fixes
which rung to *recommend* for point-thematic work — around `20-20-1` —
and nothing more: the release ships a ladder from 0.23 to 10.9 b/px
precisely because the right rung depends on the task. The open question
is not whether the high rungs are wasted but what they are worth, and no
benchmark in the suite can answer it, because all five are coarse
thematic labels at a single point. A fine-spatial or temporal benchmark
is the highest-value addition to the eval suite.

---

## Suggested sequence

1. **PR 1 — fixes.** §A1 alone, against `main`. Small, uncontroversial,
   independently valuable, and it de-risks everything after it.
2. **PR 2 — Tessera source + release.** §A2, the 4 surviving configs,
   `georvq/tessera/` + `docs/TESSERA_RELEASE.md`, weights artefact.
3. **PR 3 — codec features.** §A3–A8: `stats_cond`, `vq_quant`,
   `vqinit`, `reset_mode`, `depth_sampling`, `dist.py`, eval tooling.
   Update OVERVIEW/ADDING_A_SOURCE in the same PR.
4. **PR 4 — the onboarding suite.** SOURCE_ONBOARDING + ANALYSIS_SUITE +
   the ~20 promoted tools, as an explicitly work-in-progress subsystem.
5. **Deletions** (§B, §C) land with whichever PR touches the same files;
   the lessons move to FINDINGS.md / PARKED.md *first*, in the same
   commit as the code removal, so nothing is lost.

Net effect, counting lines: of the 19,121 added, roughly **10,000 merge**
— ~5,000 of library/tests (6,291 in `georvq/`+`scripts/`+`tests/`, less
~1,280 deleted with §B), ~4,160 of promoted analysis tools, ~600 of docs
and ~300 of surviving configs — and roughly **9,000 retire** with their
conclusions written down, along with ~30 committed run figures.
