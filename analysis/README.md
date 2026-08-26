# Source-characterisation and codec-diagnostic probes

The tools behind `docs/SOURCE_ONBOARDING.md`, which is the procedure;
this is the inventory. Each answers one question about a data source or
about where a trained codec is losing, and each is kept because its
answer changed a design decision — findings are in the ledger (`lab/findings/`).

Run them against a chip store and a checkpoint:

```bash
PYTHONPATH=. python analysis/data_probe.py --source tessera
PYTHONPATH=. python analysis/tier_collapse.py --ckpt <checkpoint.pt>
```

Most take `--ckpt` and are source-generic; the exceptions are noted.
They read stores and checkpoints and write nothing back to them.

## A. Characterise the source, before any model

| tool | question |
|---|---|
| `data_probe.py` | The entry point for a new source: validity, magnitude spread, lossless floor from spatial prediction, autocorrelation range. |
| `pyramid_sizing.py` | Turns those measurements into candidate window, tier grids and latent widths. |
| `patch_intrinsic_dim.py` | Within one patch, is the variation a few linear directions or a few prototype "species"? Local vs pooled. (`--source tessera\|ae\|s2`) |
| `patch_moments_pca.py` | Across the dataset, how low-rank are the per-patch mean and std vectors? (tessera-hardcoded) |
| `moment_sample_size.py` | Is the sample big enough — estimator convergence vs *effective* N under correlated sampling. |

## B. Bound what is achievable, before training a codec

| tool | question |
|---|---|
| `pixel_rd.py` | Single-pixel R-D: bottleneck dim x rate, with the encoder trained *with* its quantizer. |
| `moment_rd.py` | R-D of the moment vector against the Gaussian bound by reverse water-filling. |
| `moment_hybrid.py` | Nonlinear stage first, then transform coding on the residual. |
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

Diagnostics that only report whether a run is healthy — rather than what
to build — were not kept, nor were the fitting experiments hardcoded to
one historical latent store. Their conclusions are in
the ledger; the code is not worth carrying, because it was
written against stores and checkpoints that no longer exist.
