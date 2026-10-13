Draft for review: the module-map additions for AE v2 and the install line are new and not yet approved.

# Codebase overview

Orientation for humans and agents working on georvq. `README.md` is the
front door and `docs/DESIGN.md` explains the method and the reasoning
behind it; this explains how the code is organized, what must not break,
and the practical knowledge that isn't visible from signatures.

## One pass through the pipeline

```
chip store (sharded memmaps + parquet index + manifest)
  → ChipPool / GroupChipLoader      # resident pool, background refresh
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

Variants are the *presence of optional submodules*, never flag routing.
`PyramidCodec.quantize()` is the single routing point that picks the
plain / skip-masked / cascade / gated quantization path from which
submodules exist. Everything is assembled from one validated `Config`
(`factory.build_model`); rebuild any checkpoint exactly with
`georvq.from_checkpoint(path)`.

## Module map

| Path | What lives there |
|---|---|
| `georvq/config.py` | Pydantic-v2 schema (`extra="forbid"`), YAML loader with `inherit:` + dot-overrides, `config_hash`, `config_diff` |
| `georvq/factory.py` | `build_model(cfg)`, `from_checkpoint(path)` |
| `georvq/compat.py` | Legacy (research-repo) checkpoint loader: prefix-renames state dicts, reconstructs a `Config` from argparse args |
| `georvq/metrics.py` | `masked_recon_stats`, `centered_cos` |
| `georvq/repro.py` | Seeding, git/env metadata, `save_checkpoint` (embeds config + hash) |
| `georvq/cli.py` | `georvq-train`, `georvq-train-stage1` entry points |
| `georvq/model/vq.py` | `VectorQuantizer`, `ResidualVQ`, k-means init, dead-code revival (`reset_mode`) — numerics frozen, see the invariants below |
| `georvq/model/vqinit.py` | Codebook seeding: balanced-PCA-tree init, per-tier init budgets (`init_per_code`) |
| `georvq/model/vq_quant.py` | b-bit quantization-aware codebooks (`vq.book_bits`) — int8 books are ~4x smaller to download |
| `georvq/model/stats_cond.py` | The transmitted-mean side channel: centre before the encoder, condition both ends, restore after the decoder |
| `georvq/model/codec.py` | `PyramidCodec`, matryoshka depth sampling, `level_cumsums`/`eval_tuple` (encode once, decode any rate) |
| `georvq/model/tokenizer.py`, `decoder.py`, `layers.py` | ViT encoder, AdaLN attention decoder + `DenseHead` + `DecoderBank`, shared blocks |
| `georvq/model/ladder.py`, `tiersum.py` | Exploratory decoders with early exits per tier prefix (ERA5, Tessera presets) |
| `georvq/model/attnmask.py` | Local attention masks (`model.local_attn`) |
| `georvq/model/rate.py` | `ContextRateModel`, `PyramidRateModel`, `soft_rate`, `calib_tau` |
| `georvq/model/conditioning.py` | `TierPredictor` (cascade), `CodebookSelector` (gated subset) |
| `georvq/model/location.py` | SatCLIP clone-on-demand loader, location tables, location→latent prior MLPs |
| `georvq/data/` | `store.py` chip-store v1 + `open_store` + `geographic_split`; `pool.py` resident pools and `GroupBatch`; `build.py` `ShardWriter`; live readers `live_dem.py`, `live_s2.py`, `era5.py`; `sampler.py` sample geometry |
| `georvq/sources/` | `Source` interface + registry; `ae.py` (minimal reference), `tessera.py` (dense embeddings with a per-pixel scale), `s2.py` (multi-resolution example), `dem.py` (one band with a mean/std side channel), `era5.py` (gridded physical variables); `prep.py` fitted model-space transforms; `report.py` the per-source reporting surface (metrics, baselines, figures, previews) |
| `georvq/train/` | `loop.py` the one trainer (it also owns the dead-code reset schedule, `mark_resets`); `stage1.py` σ-conditioned pretrain; `grow.py` grown training; `dist.py` data parallelism for frozen codebooks; `trap.py` a flight recorder and tripwire for gradient spikes; `validate.py`, `figures.py`, `report.py`, `warmstart.py` |
| `georvq/eval/` | Held-out extraction, probes, benchmark registry, codec A/B helpers |
| `georvq/viz/` | Panel/GIF styling and builders |
| `georvq/inference/` | Source-agnostic deployment machinery: `api.py` Codec/Packet, `stats_codec.py`, `archive.py` rANS (research tool), `refine.py` additive residual refinements, `mean_codec.py` the `stats_cond` channel on the wire, `release.py` bundles + bitstream/decoder ids, `onnx_backend.py` ONNX export + torch-free runtime, `georvq_parquet.py` the GeoParquet release container (layout, writer, range reader, record cache), `bitpack.py` fixed-width records, `onnx_stats.py` ONNX for transmitted-mean models |
| `georvq/ae/` | **The AlphaEarth release codec**: `v2.py` (codec, `georvq.ae.open`, CLI), `v2_bundle.py`, `v2_resolve.py` (finding the bundle a file names), `index.py`. See `docs/AE_RELEASE.md`. `cli.py`, `cog.py` and `deblock.py` are the earlier release, kept so its bundles still load |
| `georvq/tessera/` | **The Tessera release codec** — Tessera-specific policy (transmitted mean, rate ladder, fine-tier refinement). See `docs/TESSERA_RELEASE.md` |
| `scripts/` | Thin drivers over the above (train, store builders, eval, archive, GIFs) |
| `analysis/` | Source-characterisation and codec-diagnostic probes (`analysis/README.md`) |
| `lab/` | The findings ledger and kept experiments (`lab/README.md`); the maintainer's TODO queue |
| `configs/` | Presets — each reproduces a specific historical run (see Provenance) |

## Invariants — do not break casually

1. **VQ numerics must not change.** The repo's value rests on trained
   checkpoints decoding bit-faithfully. `model/vq.py` (and the quantize
   paths in `codec.py`/`conditioning.py`) are guarded by the
   golden-checkpoint gate (`tests/test_compat.py`: the released weights
   must reproduce recorded reconstruction quality exactly, self-skipping
   where the release bundle is absent). Any change there must keep that
   test exact, not approximately equal.
2. **State-dict names are part of the checkpoint format.** `compat.py` maps legacy
   checkpoint keys onto today's module names by prefix. Renaming modules
   or parameters breaks every existing checkpoint; if you must, extend
   the rename table and prove a legacy checkpoint still loads strict.
3. **The bitstream excludes the decoder.** A packet/archive is defined by
   tiers + codebooks + conditioning priors (+ stats codec / CDF tables
   where used — `docs/BITSTREAM.md`). Decoder-side improvements must
   never require re-encoding; anything that changes the bitstream
   changes `bitstream_id`, and decoder-specific artifacts are keyed by
   `decoder_id`. Additive refinements (`inference/refine.py`) sit
   *outside* this: they extend a packet without touching its indices, so
   ignoring one always yields a valid decode at a lower rate.
4. **A released quantizer's payload is frozen.** `georvq/ae`'s `star.tif`
   is public, so its bytes cannot change even when the code behind them
   is generalised; `tests/test_refine.py` pins the packing against an
   inlined copy of the original implementation. Treat any shipped
   container the same way.
5. **VQ state stays fp32 under bf16 autocast.** Assignment scores may run
   bf16 (`vq.bf16_scores`); residuals, EMA buffers and codebooks must not.
6. **Sources own all data-specific behavior.** The shared stack
   (tokenizer→decoder, trainer, evals, inference) must stay
   source-agnostic. If you're adding an `if source == ...` anywhere
   outside `sources/`, it belongs in the adapter instead. This includes
   what a run *reports*: the trainer reads a `ReportSpec`
   (`sources/report.py`) for its metric names, baselines, figures and
   preview, and no longer knows what a cosine is. Hardcoding a metric
   key in `train/` is the same violation as hardcoding a dequant.
7. **Configs carry reproducibility.** Every training artifact
   embeds its resolved config; anything that affects a run must live in
   the `Config` schema (validated, `extra="forbid"`), not in an env var
   or an edit-the-source knob.

## Running things

```bash
pip install -e ".[train,loc,archive,eval,viz,dev]"

georvq-train -c configs/ae/pyr64.yaml --gpu 0 log.run_tag=my_run   # train
georvq-train -c configs/ae/pyr64.yaml train.steps=2000 ...          # any override

pytest -m "not slow and not gpu"    # fast suite (~40 s) — run on every change
pytest                              # + training smokes (CPU, seconds) and GPU tests

python scripts/eval_run_ab.py --dataset <ds> --sanity identity   # eval harness sanity
python scripts/eval_rate_sweep.py --dataset <ds> --ckpt <ckpt>   # Pareto-front A/B
python scripts/build_archive.py --ckpt <ckpt> --store-dir <dir> --out <dir>  # research rANS archives


# AlphaEarth release codec (georvq/ae — see docs/AE_RELEASE.md)
python -m georvq.ae.v2 encode src.tif out_dir/ --bundle <bundle>   # -> {stem}.georvq.parquet
python -m georvq.ae.v2 decode tile.georvq.parquet out.npy --quality 8-8-2
python -m georvq.ae.v2 export-onnx <onnx-dir> --bundle <bundle>   # torch-free decode

# Tessera release codec (georvq/tessera — see docs/TESSERA_RELEASE.md)
python scripts/build_release_tessera.py --ckpt <ckpt> --out <bundle>
```

## Who touches what

- **Codec researcher** (new sources, better models): `georvq/{model,
  data, sources, train, eval}`, `configs/`, `scripts/`,
  `docs/ADDING_A_SOURCE.md` + the findings ledger (`lab/`). The
  invariants above are yours to uphold.
- **Release engineer** (ship a trained codec): `georvq/inference/`
  (bundles, ids, refinements, ONNX export) plus a source-specific release
  layer. `georvq/ae/` and `georvq/tessera/` are two worked examples of
  the same contract, and reading them side by side is the fastest way to
  see which decisions are policy: they differ on whether a stats channel
  is sent at all, on which tier carries a refinement, and on whether that
  refinement is a default or an opt-in — each difference measured, in
  `docs/AE_RELEASE.md` and `docs/TESSERA_RELEASE.md`.
- **Data end user** (decode compressed data): for AlphaEarth,
  `georvq.ae.open` and the `{stem}.georvq.parquet` files, with
  `docs/AE_RELEASE.md` as the complete contract; for Tessera, `georvq.tessera` and
  `docs/TESSERA_RELEASE.md`. Nothing in training concerns them.

Smoke config for cheap end-to-end checks: `configs/smoke.yaml` (synthetic
store, CPU-friendly). Prefer it — or a few hundred steps with overrides —
over long runs when validating a change; the fast pytest suite plus one
short smoke catches nearly everything.

## Preset provenance

Each preset encodes a recipe that actually ran and worked in the research
cycle, with its settings verified against the original runs' logged argv:

- `pyr64.yaml` — the standard AE recipe (incl. its k-means warmup). The
  released AE weights came from a warm-start chain on this geometry that
  no single preset reproduces.
- `stage1.yaml` — σ-conditioned continuous pretrain.
- `hybrid_rate.yaml` — frozen stage-1 encoder + fresh smaller-deeper
  codebooks + entropy model (its small 256-slot pool is historical).
- `cascade.yaml` — location prior + cross-tier predictors, encoder frozen
  for the whole run (co-adaptation degrades the ceiling — `lab/findings/coadaptation-degrades-ceiling.md`).
- `tessera_pyr64.yaml` — the 3-tier pyramid on Tessera embeddings.
- `tessera_vq8.yaml` — that geometry with int8 QAT codebooks.
- `tessera_vq8_k256.yaml` — graded-down codebooks (K=256 x 20 levels),
  the sizing the sample-budget rule implies.
- `tessera_vq8_k256_statscond.yaml` — the released Tessera codec: the
  above plus the transmitted patch mean.
- `satclip_select.yaml` — selector-only on a frozen codec. The
  4×-codebooks-from-scratch variant in its header comment is an
  **unvalidated research direction**: it never trained past k-means
  historically, and a short attempt here diverged. Don't present it as a
  working recipe.

The S2, DEM and ERA5 presets are for sources still in development.

Treat preset edits like code changes: they are claims about what
reproduces a result.

## Practical gotchas

- **`conda run` buffers stdout.** Background training logs stay empty
  until the process exits. To check progress, read the run dir instead:
  `torch.load(<run>/model_best.pt)["step"/"metrics"]`, or
  `final_val.json` after completion.
- **Console entry points `sys.exit()` the function's return value.**
  Returning a string from a `cli.py` function makes the command exit
  nonzero after success. Return `None`.
- **Huge codebooks need the k-means path.** The `vq.warmup_steps: 0`
  data-init path materializes an N×K one-hot in the EMA update and OOMs
  around K≈65k; set `vq.warmup_steps` ≥ a few hundred so codebooks init
  via residual k-means (`_chunked_assign` is memory-safe).
- **SatCLIP** is fetched on first use: pinned git clone into
  `~/.cache/georvq/satclip` + HF weights. Offline machines: clone it
  yourself and set `GEORVQ_SATCLIP_DIR`. Needs the `[loc]` extra.
- **Pool nondeterminism.** The background pool refresh makes training
  runs seed-stable but not bit-identical; expect ~±0.001 val-cos between
  identically-configured runs. Bit-exactness claims belong to the model
  (fixed inputs), not the trainer.
- **Multi-group (S2-style) sources** stream through the same `ChipPool`
  and train loop as single-plane ones, as one resident tensor per group
  sharing a slot index and a crop origin. That origin must land on the
  coarsest native grid (period 6 on S2), so multi-group crops are always
  aligned and `slot_rows` must divide by the period — both asserted. See
  `docs/ADDING_A_SOURCE.md`.

## Where to extend

- **New dataset** → builder + `Source` adapter + preset:
  `docs/ADDING_A_SOURCE.md`.
- **Better decoder / post-decode enhancement** → free under the
  bitstream contract (`docs/BITSTREAM.md`); the AE deblock decoder
  (`georvq/ae/deblock.py`, shipped) is the worked example — note its
  canonical-recipe pattern: post-processors whose output depends on
  tiling must freeze the tiling into the decoder identity.
- **Context-adaptive entropy coding** → the achievable rate is already
  measured (`model.entropy_level_bits`; order-0 coding was measured
  ~0.6% on AE and dropped — the win requires the context model); a coder
  driven by the same `ContextRateModel` slots behind the packet layout.
- **Shipping a new source** → a per-source release layer over
  `georvq/inference`; `georvq/ae/` + `docs/AE_RELEASE.md` are the worked
  example.
- **Why the architecture is shaped this way** → `docs/DESIGN.md`
  (empirical results, including the negative ones worth not repeating).
