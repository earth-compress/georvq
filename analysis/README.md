# Analysis harnesses (tessera investigation, Aug 2026)

Probes and experiment scripts from the stage-2 / codebook investigation.
They read datasets from `/data/tessera-samples/` (chip store, latent
stores) and checkpoints from `ring-space/checkpoints/`; nothing here
writes to the dataset partition. Run with
`PYTHONPATH=<worktree> /home/ali/miniconda3/envs/ring-space/bin/python`.

Findings these produced are written up in `docs/FINDINGS.md`; figures are
in `reports/`; run logs and result artefacts in
`/home/ali/projects/georvq/logs/tessera/`.

| script | what it measures |
|---|---|
| `probe_vq_health.py` | per-tier/level codebook usage, perplexity, radial/tangential error split |
| `probe_gain_shape.py` | gain-shape vs unconstrained RVQ bake-off (negative result) |
| `hidden_test.py` | cross-tier: refits a *method* on all tiers, scores end-to-end decoded |
| `fit_sweep.py`, `perlevel.py`, `crossfit.py` | rate allocation, per-level decay, cross-fitted refinement |
| `lattice.py`, `compand.py`, `pladder.py` | transform coding (PCA + scalar quant) vs RVQ |
| `branch.py`, `waterfill.py`, `embedded.py` | branching R-D ladders; embedded (nested) quantizers |
| `budget.py`, `budget2.py`, `bitdepth.py` | codebook storage budget: dims vs bit depth |
| `masked.py`, `pca25.py`, `lvl4.py`, `lvl8.py` | masked / eigenbasis-truncated codebooks |
| `sens.py`, `corr.py`, `wkm.py` | decoder sensitivity per dimension and per principal direction |
| `rdcmp.py` | R-D of the trained phase-1 codec vs the new encoder pre-finetune |
| `build_full.py`, `build_full2.py` | full-corpus int8 latent store builder (2 GPUs) |
| `verify_latent_store.py` | store completeness: shard integrity, row agreement, chip coverage |
| `fit_corpus_books.py` | **stage-2 step 1**: corpus-scale int4 codebook fit -> merged `init_from` checkpoint |
| `build_latents.py` | sampled latent store (superseded by the full build) |
| `ae_books.py`, `ae_cbopt.py` | AE codebooks: fresh refit, and decoder-loss optimisation |
| `residual_diag.py`, `ae_attn_probe.py` | error spectra; AE attention locality probe |
| `cb_challenge_agent/` | isolated-agent codebook challenge: task, fixed scorer, agent's solutions |
