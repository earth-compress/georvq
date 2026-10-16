# Codebase overview

This document is a map of the georvq codebase for the people and agents
who are developing it or want to know more. It describes how the code is organised, 
which invariants protect released codecs, and the practical knowledge that cannot be read
from function signatures. `README.md` introduces the project, and
`docs/DESIGN.md` explains the method and the reasoning behind it.

## The pipeline

```
chip store (sharded memmaps + parquet index + manifest)
  → ChipPool                        # resident pool (single-plane or multi-group),
                                    #   background refresh
  → Source.prep                     # dequant + nodata mask, on device
  → PyramidCodec
      [stats_cond]                  # optional: remove the per-window mean,
                                    #   condition both ends on it, send it
      source stem                   # raw plane(s) → (B, C, W, W) features
      ViTTokenizer                  # CLS = coarse tier, conv patch embeds per tier
      per-tier ResidualVQ           # coarse→fine residual codebooks
      [TierPredictor]               # optional: predict-and-subtract conditioning
      [CodebookSelector]            # optional: location-gated codebook subset
      [PyramidRateModel]            # optional: learned entropy model
      PyramidAttnDecoder            # AdaLN attention trunk, all tiers → fine tokens
                                    #   (exploratory alternatives: decoder bank,
                                    #   ladder, tiersum)
      source head                   # fine tokens → pixel plane(s)
      [stats_cond]                  #   ...and add the mean back
  → Source.recon_stats              # loss + metrics in model space
```

A model variant is made by adding an optional submodule, rather than by
a config flag that is checked in several places. `PyramidCodec.quantize()`
is the only place that routes between variants, and it picks the plain,
skip-masked, cascade or gated quantisation path according to which
submodules exist. Everything is assembled from one validated `Config`
(`factory.build_model`), and `georvq.from_checkpoint(path)` rebuilds any
checkpoint exactly.

## Module map

| Path | Contents |
|---|---|
| `georvq/config.py` | Pydantic-v2 schema (`extra="forbid"`), YAML loader with `inherit:` + dot-overrides, `config_hash`, `config_diff` |
| `georvq/factory.py` | `build_model(cfg)`, `from_checkpoint(path)` |
| `georvq/compat.py` | Legacy (research-repo) checkpoint loader: prefix-renames state dicts, reconstructs a `Config` from argparse args |
| `georvq/metrics.py` | `masked_recon_stats`, `centered_cos` |
| `georvq/repro.py` | Seeding, git/env metadata, `save_checkpoint` (embeds config + hash) |
| `georvq/cli.py` | `georvq-train`, `georvq-train-stage1` entry points |
| `georvq/model/vq.py` | `VectorQuantizer`, `ResidualVQ`, data-sample init, dead-code revival (`reset_mode`). Its numerics are frozen (see Invariants) |
| `georvq/model/vqinit.py` | Codebook seeding: balanced-PCA-tree init, per-tier init budgets (`init_per_code`) |
| `georvq/model/vq_quant.py` | b-bit quantisation-aware codebooks (`vq.book_bits`), so that int8 books are ~4x smaller to download |
| `georvq/model/stats_cond.py` | The transmitted-mean side channel: centre before the encoder, condition both ends, restore after the decoder |
| `georvq/model/codec.py` | `PyramidCodec`, matryoshka depth sampling, `level_cumsums`/`eval_tuple` (encode once, decode any rate) |
| `georvq/model/tokenizer.py`, `decoder.py`, `layers.py` | ViT encoder, AdaLN attention decoder + `DenseHead` + `DecoderBank`, shared blocks |
| `georvq/model/ladder.py`, `tiersum.py` | Exploratory decoders with early exits per tier prefix (ERA5, Tessera presets) |
| `georvq/model/rate.py` | `ContextRateModel`, `PyramidRateModel`, `soft_rate`, `calib_tau` |
| `georvq/model/conditioning.py` | `TierPredictor` (cascade), `CodebookSelector` (gated subset) |
| `georvq/model/location.py` | SatCLIP clone-on-demand loader, location tables, location→latent prior MLPs |
| `georvq/model/deblock.py` | `Deblocker`, an exploratory seam-removal post-decoder (no release ships it) |
| `georvq/data/` | `store.py` chip-store v1 + `open_store` + `geographic_split`; `pool.py` resident pools and `GroupBatch`; `build.py` `ShardWriter`; live readers `live_dem.py`, `live_s2.py`, `era5.py`; `sampler.py` `WindowSampler`, the one sample geometry |
| `georvq/sources/` | `Source` interface + registry; `ae.py` (minimal reference), `tessera.py` (dense embeddings with a per-pixel scale), `s2.py` (multi-resolution example), `dem.py` (one band with a mean/std side channel), `era5.py` (gridded physical variables); `prep.py` fitted model-space transforms; `report.py` the per-source reporting surface (metrics, baselines, figures, previews) |
| `georvq/train/` | `loop.py` the one trainer (it also owns the dead-code reset schedule, `mark_resets`); `stage1.py` σ-conditioned pretrain; `grow.py` grown training; `dist.py` data parallelism for frozen codebooks; `signals.py` the SIGUSR1/SIGTERM stop flag; `validate.py`, `figures.py`, `report.py`, `warmstart.py` |
| `georvq/eval/` | Held-out extraction, probes, benchmark registry, codec A/B helpers |
| `georvq/inference/` | Source-agnostic deployment code: `api.py` Codec/Packet, `stats_codec.py`, `refine.py` additive residual refinements, `mean_codec.py` the `stats_cond` channel on the wire, `release.py` bundles + bitstream/decoder ids, `records.py` the release container (record layout, Zarr writer, zarr-python reader, record cache), `bitpack.py` fixed-width records, `onnx_stats.py` ONNX for transmitted-mean models |
| `georvq/ae/` | The AlphaEarth release codec: `v2.py` (codec, `georvq.ae.open`, CLI), `v2_bundle.py`, `v2_resolve.py` (finding the bundle a store names), `index.py`. See `docs/AE_RELEASE.md` |
| `georvq/tessera/` | The Tessera release codec, holding Tessera-specific policy (transmitted mean, rate ladder, fine-tier refinement). See `docs/TESSERA_RELEASE.md` |
| `scripts/` | Thin drivers over the above (train, store builders, eval, release builds); `scripts/viz/` panel/GIF styling and builders |
| `analysis/` | Source-characterisation and codec-diagnostic probes (`analysis/README.md`) |
| `lab/` | The findings ledger and kept experiments (`lab/README.md`); the maintainer's TODO queue |
| `configs/` | Presets, each of which reproduces a specific historical run (see Preset provenance) |

## Invariants

1. VQ numerics must not change. Files encoded with a released checkpoint
   can only be read for as long as that checkpoint decodes them to the
   same bits, so `model/vq.py`, the quantise paths in
   `codec.py`/`conditioning.py`, and `model/vq_quant.py` are guarded by
   golden tests. `tests/test_ae_v2_bundle.py` and
   `tests/test_tessera_bundle.py` require the released bundles to
   reproduce their recorded codes and decoded bits exactly, and skip
   themselves where the bundle is absent. Any change there must keep those tests exact, not
   approximately equal. The recorded hashes hold on torch 2.10 or later
   with numpy 2; older versions give different codes.
2. State-dict names are part of the checkpoint format. `compat.py` maps
   legacy checkpoint keys onto today's module names by prefix. Renaming
   modules or parameters breaks every existing checkpoint, so if a
   rename is unavoidable, extend the rename table and show that a legacy
   checkpoint still loads strictly.
   Config fields that are removed or renamed go in `RETIRED_FIELDS` or
   `RENAMED_FIELDS` (`config.py`), so checkpoints that embed them still
   load.
3. The bitstream excludes the decoder. A packet is defined by the tier
   specs, the codebooks, any conditioning priors and the release layer's
   own decode state, such as a mean channel's tables, which enters
   `bitstream_id` through its `extra` argument (`docs/BITSTREAM.md`).
   Decoder-side improvements must never require re-encoding. Anything
   that changes the bitstream changes `bitstream_id`, and
   decoder-specific artifacts are keyed by `decoder_id`. Additive
   refinements (`inference/refine.py`) sit outside this definition,
   because they extend a packet without touching its indices, so
   ignoring one always gives a valid decode at a lower rate.
4. A released quantiser's payload is frozen. Shipped files must stay
   readable even when the code behind them is generalised.
   `tests/test_ae_v2_bundle.py` reads back a golden Zarr tile
   (`tests/golden/ae_v2_tile.georvq.zarr`), and
   `tests/test_tessera_bundle.py` pins the bytes of the packed Tessera
   refinement. Any other shipped container should be treated the same
   way.
5. VQ state stays fp32 under bf16 autocast. Assignment scores may run in
   bf16 (`vq.bf16_scores`), but residuals, EMA buffers and codebooks must
   not.
6. Sources own all data-specific behaviour. The shared stack
   (tokeniser→decoder, trainer, evals, inference) must stay
   source-agnostic, so an `if source == ...` outside `sources/` belongs
   in the adapter instead. This includes what a run reports. The trainer
   reads a `ReportSpec` (`sources/report.py`) for its metric names,
   baselines, figures and preview, and names no metric itself, so
   hardcoding a metric key in `train/` is the same violation as
   hardcoding a dequantisation.
7. Configs carry reproducibility. Every training artifact embeds its
   resolved config, so anything that affects a run must be in the
   `Config` schema (validated, `extra="forbid"`), not in an environment
   variable or an edit to the source.

## Running things

```bash
pip install -e ".[train,eval,viz,onnx,era5,dev]"

georvq-train -c configs/ae/pyr64.yaml --gpu 0 log.run_tag=my_run   # train
georvq-train -c configs/ae/pyr64.yaml train.steps=2000 ...          # any override

pytest -m "not slow"    # fast suite incl. training smokes (~80 s on 8 CPUs) — run on every change
pytest                  # + the resume and real-checkpoint tests

python scripts/eval_rate_sweep.py --dataset <ds> --ckpt <ckpt> --sanity identity   # eval harness sanity
python scripts/eval_rate_sweep.py --dataset <ds> --ckpt <ckpt>   # Pareto-front A/B


# AlphaEarth release codec (georvq/ae — see docs/AE_RELEASE.md)
python -m georvq.ae.v2 encode src.tif out_dir/ --bundle <bundle>   # -> {stem}.georvq.zarr
python -m georvq.ae.v2 decode tile.georvq.zarr out.npy --quality 8-8-2
python -m georvq.ae.v2 export-onnx <onnx-dir> --bundle <bundle>   # torch-free decode

# Tessera release codec (georvq/tessera — see docs/TESSERA_RELEASE.md)
python scripts/build_release_tessera.py --ckpt <ckpt> --out <bundle>
```

`configs/smoke.yaml` runs the pipeline end to end on a synthetic store
and is cheap enough for a CPU. When validating a change, prefer it, or a
few hundred steps with overrides, over a long run. The fast pytest suite
already includes training smokes, and together with one short smoke run
it is usually enough to validate a change.

## Roles

A codec researcher, working on new sources or better models, works in
`georvq/{model, data, sources, train, eval}`, `configs/` and `scripts/`,
with `docs/ADDING_A_SOURCE.md` and the findings ledger (`lab/`). These
are the changes most likely to break the invariants above, so keeping
them intact is the researcher's job.

A release engineer, shipping a trained codec, works in
`georvq/inference/` (bundles, ids, refinements, ONNX export) and in a
source-specific release layer. `georvq/ae/` and `georvq/tessera/` are two
worked examples of the same contract, and reading them side by side is
the quickest way to see which decisions are policy. Both send a
transmitted mean, but they differ in how it is coded, in the container,
and in whether a refinement is offered (only Tessera has one, as an
opt-in). Each of these differences was measured, and the measurements
are in `docs/AE_RELEASE.md` and `docs/TESSERA_RELEASE.md`.

A data end user, decoding compressed data, needs `georvq.ae.open` and
the `{stem}.georvq.zarr` stores for AlphaEarth, with
`docs/AE_RELEASE.md` as the complete contract, or `georvq.tessera` and
`docs/TESSERA_RELEASE.md` for Tessera. None of the training code is
needed for this.

## Preset provenance

Each preset encodes a recipe that ran and worked during the research
cycle, with its settings checked against the logged argv of the original
runs.

- `pyr64.yaml` is the standard AE recipe, including its k-means warmup.
  The released AE weights came from a warm-start chain on this geometry
  that no single preset reproduces.
- `stage1.yaml` is the σ-conditioned continuous pretrain.
- `hybrid_rate.yaml` is a frozen stage-1 encoder with fresh, smaller and
  deeper codebooks and an entropy model (its small 256-slot pool is
  historical).
- `cascade.yaml` is the location prior with cross-tier predictors, with
  the encoder frozen for the whole run because co-adaptation degrades
  the ceiling (`lab/findings/coadaptation-degrades-ceiling.md`).
- `tessera_pyr64.yaml` is the 3-tier pyramid on Tessera embeddings.
- `tessera_vq8.yaml` is that geometry with int8 QAT codebooks.
- `tessera_vq8_k256.yaml` has graded-down codebooks (K=256 x 20 levels),
  the sizing that the sample-budget rule implies.
- `tessera_vq8_k256_statscond.yaml` is the released Tessera codec, which
  is the above with the transmitted patch mean added.
- `satclip_select.yaml` trains the selector only, on a frozen codec. The
  variant with 4× codebooks trained from scratch, described in its
  header comment, is an unvalidated research direction. It never trained
  past k-means historically, and a short attempt here diverged, so do not
  present it as a working recipe.

The S2, DEM and ERA5 presets are for sources still in development.

A preset records the run that reproduces a result, so it should change
only when that run changes, and an edit to one deserves the same review
as a change to the code.

## Practical gotchas

- `conda run` buffers stdout, so background training logs stay empty
  until the process exits. To check progress, read the run directory
  instead, with
  `georvq.repro.load_checkpoint(<run>/model_best.pt)["step"/"metrics"]`
  (plain `torch.load` needs `weights_only=False` on torch 2.6 or later),
  or `final_val.json` after completion.
- Console entry points `sys.exit()` the function's return value, so
  returning a string from a `cli.py` function makes the command exit
  nonzero after success. Return `None`.
- Huge codebooks need the k-means path. The `vq.warmup_steps: 0`
  data-init path materialises an N×K one-hot in the EMA update and runs
  out of memory around K≈65k. Set `vq.warmup_steps` to at least a few
  hundred, so that codebooks are initialised by residual k-means
  (`_chunked_assign` is memory-safe).
- SatCLIP is fetched on first use, as a pinned git clone into
  `~/.cache/georvq/satclip` plus HF weights. On an offline machine,
  clone it yourself and set `GEORVQ_SATCLIP_DIR`.
- The background pool refresh makes training runs seed-stable but not
  bit-identical, so expect about ±0.001 val-cos between identically
  configured runs. Claims of bit-exactness apply to the model on fixed
  inputs, not to the trainer.
- `train.resume=<run>/model.pt` continues a run from its own checkpoint.
  Weights, optimiser, scheduler, step, best score and RNG state are
  restored, and the schedule carries on from the saved step. It is
  exclusive with `train.init_from`, which starts a new run from another
  run's weights.
- SIGUSR1 or SIGTERM makes the trainer write a resumable `model.pt` at the end of the current step and exit 0, so
  Slurm's `--signal=B:USR1@300` ahead of the time limit, or `scancel`,
  never kills a run mid-step. The signal reaches the trainer only if the
  job script `exec`s python or forwards the signal to it.
- With `vq.warmup_steps > 0`, a warm start refits the codebooks by k-means after the warmup, discarding any
  shape-matching books loaded from `train.init_from`. The trainer
  refuses to start in that case unless `train.refit_books=true` says the
  refit is intended.
- Multi-group (S2-style) sources stream through the same `ChipPool` and
  train loop as single-plane ones, as one resident tensor per group
  sharing a slot index and a crop origin. That origin must land on the
  coarsest native grid (period 6 on S2), so multi-group crops are always
  aligned and `slot_rows` must divide by the period. The code asserts
  both. See
  `docs/ADDING_A_SOURCE.md`.

## Where to extend

A new dataset needs a builder, a `Source` adapter and a preset, and
`docs/ADDING_A_SOURCE.md` describes all three.

A better decoder or a post-decode enhancement can be added freely under
the bitstream contract (`docs/BITSTREAM.md`). The seam deblocker
(`georvq/model/deblock.py`, exploratory and not shipped) is a worked
example, and its canonical recipe shows one rule such work has to
follow: a post-processor whose output depends on the tiling must freeze
that tiling into the decoder identity.

For context-adaptive entropy coding, the achievable rate is already
measured (`model.entropy_level_bits`). Order-0 coding was measured at
~0.6% on AE and dropped, because the gain needs the context model. A
coder driven by the same `ContextRateModel` would change how packets are
stored, but not what a packet contains.

Shipping a new source means writing a release layer over
`georvq/inference`, and `georvq/ae/` with `docs/AE_RELEASE.md` is the
worked example.

The reasons the architecture is shaped as it is are in `docs/DESIGN.md`,
including the negative results, which are recorded so that they are not
repeated.
