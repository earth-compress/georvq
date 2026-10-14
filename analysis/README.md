# Source-characterisation and codec-diagnostic probes

The tools behind `docs/SOURCE_ONBOARDING.md`, which is the procedure;
this is the inventory. Each answers one question about a data source or
about where a trained codec is losing, and each is kept because its
answer changed a design decision — findings are in the ledger (`lab/findings/`).

Run them against a chip store and a checkpoint:

```bash
PYTHONPATH=. python analysis/data_probe.py --store <chip store> --source tessera
PYTHONPATH=. python analysis/tier_collapse.py --ckpt <checkpoint.pt>
```

Only `data_probe.py`, `pyramid_sizing.py` and `tier_collapse.py` are
source-generic, and `patch_intrinsic_dim.py` takes tessera, ae or s2. The
moment and pixel R-D tools
(`patch_moments_pca.py`, `moment_rd.py`, `moment_hybrid.py`,
`moment_sample_size.py`, `pixel_rd.py`) read Tessera chips directly
(128 bands, Tessera's own dequantization). `tier_rate_attrib.py`,
`refine_decode.py`, `tier0_moments.py`, `tier0_spatial.py` and
`moment_match.py` score reconstructions with per-pixel cosine or centred
cosine, which are measures for embedding fields, rather than with the
source's declared `ReportSpec`. Several checkpoint tools default `--ckpt` to Tessera checkpoints
(`checkpoints/tessera_vq8_split2`, `checkpoints/tessera_vq8_k256`) that
no longer exist, so pass `--ckpt` explicitly. Checkpoint tools take
`--device` (default `cuda`, which under Slurm is the allocated card).
The tools read stores and checkpoints and write nothing back to them.

## A. Characterise the source, before any model

| tool | question |
|---|---|
| `data_probe.py` | The entry point for a new source: validity, magnitude spread, lossless floor from spatial prediction, autocorrelation range. `--store` is required. |
| `cbp_sources.py` | Chip loaders for the ClimateBenchPress stores (CMIP6, NextGEMS, ESA CCI biomass, CAMS), imported by `data_probe.py` (and through it by `pyramid_sizing.py`). Not run on its own. |
| `pyramid_sizing.py` | Turns those measurements into candidate window, tier grids and latent widths. |
| `patch_intrinsic_dim.py` | Within one patch, is the variation a few linear directions or a few prototype "species"? Local vs pooled. (`--source tessera\|ae\|s2`) |
| `patch_moments_pca.py` | Across the dataset, how low-rank are the per-patch mean and std vectors? (Tessera only) |
| `moment_sample_size.py` | Is the sample big enough — estimator convergence vs *effective* N under correlated sampling. (Tessera only, reads the `patch_moments_pca.py` cache) |

## B. Bound what is achievable, before training a codec

| tool | question |
|---|---|
| `pixel_rd.py` | Single-pixel R-D: bottleneck dim x rate, with the encoder trained *with* its quantizer. (Tessera only) |
| `moment_rd.py` | R-D of the moment vector against the Gaussian bound by reverse water-filling. (Tessera only) |
| `moment_hybrid.py` | Nonlinear stage first, then transform coding on the residual. (Tessera only, reads the `patch_moments_pca.py` cache) |
| `hybrid_compare.py` | Compares arms at **matched distortion**, interpolating in log-error space. |

## C. Where is a trained codec actually losing?

| tool | question |
|---|---|
| `tier_rate_attrib.py` | Per-tier attribution of the quantization gap, against what each tier costs. |
| `tier_collapse.py` | Has a tier collapsed to a constant? Across-window spread, effective rank, per-level contributions. |
| `tier_increment.py` | What each tier *adds*: injected energy, alignment with the error it should correct, and the scale it works at. |
| `tier0_spatial.py` | How much spatial variation a depth tuple encodes — magnitude, alignment and autocorrelation, so "too little" is distinguished from "wrong". |
| `tier0_moments.py` | What the coarse tier does with the patch mean and std specifically. |
| `residual_struct.py` | What structure survives in the error: autocorrelation vs the source, effective rank, energy tail. |
| `moment_match.py` | What a transmitted moment channel is worth per pixel, and where it stops paying. |

## D. Quantizer health

| tool | question |
|---|---|
| `ema_vs_tree.py` | Learned EMA codebooks vs a balanced PCA tree at fixed K. |
| `level_fit.py` | Per-level iterative fitting against a frozen encoder, with held-out tracking. |
| `plot_reset_diag.py` | Per-level dead-code reset diagnostics from a run's wandb history. |
| `turboquant.py` | Rotation + Lloyd scalar coding of a post-RVQ residual, as replacement and as refinement. |
| `refine_decode.py` | Whether that refinement actually *decodes* better, per tier, through the real model. |

`turboquant.py` and `refine_decode.py` are what produced
`georvq/inference/refine.py`; use them to locate which tier of a new
model holds recoverable slack before configuring a refinement.

## E. Comparing runs

`run_compare.py` — `trace`, `rates`, `leaderboard` and `outliers` over
wandb history, which is the family of questions that otherwise gets
re-asked as one-off queries every time.

## What is deliberately absent

Diagnostics that only report whether a run is healthy, rather than what
to build, were not kept. Their conclusions are in the ledger. The
cached-latent DEM fitting chain is still in `scripts/`
(`build_latent_store.py`, `fit_codebooks.py`, `train_decoder_cached.py`,
all defaulting to the `/data/georvq-stores/copdem_lat128` latent store);
whether it stays there is an open question for the maintainer.
