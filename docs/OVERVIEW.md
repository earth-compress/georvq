# Codebase overview

Orientation for humans and agents working on georvq. `README.md` explains
the method; this explains how the code is organized, what must not break,
and the practical knowledge that isn't visible from signatures.

## One pass through the pipeline

```
chip store (sharded memmaps + parquet index + manifest)
  → ChipPool / GroupChipLoader      # resident pool, background refresh
  → Source.prep                     # dequant + nodata mask, on device
  → PyramidCodec
      source stem                   # raw plane(s) → (B, C, W, W) features
      ViTTokenizer                  # CLS = coarse tier, conv patch embeds per tier
      per-tier ResidualVQ           # coarse→fine residual codebooks
      [TierPredictor]               # optional: predict-and-subtract conditioning
      [CodebookSelector]            # optional: location-gated codebook subset
      [PyramidRateModel]            # optional: learned entropy model
      PyramidAttnDecoder            # AdaLN attention trunk, all tiers → fine tokens
      source head                   # fine tokens → pixel plane(s)
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
| `georvq/model/vq.py` | `VectorQuantizer`, `ResidualVQ`, k-means init — **parity-sacred, see below** |
| `georvq/model/codec.py` | `PyramidCodec`, matryoshka depth sampling, `level_cumsums`/`eval_tuple` (encode once, decode any rate) |
| `georvq/model/tokenizer.py`, `decoder.py`, `layers.py` | ViT encoder, AdaLN attention decoder + `DenseHead`, shared blocks |
| `georvq/model/rate.py` | `ContextRateModel`, `PyramidRateModel`, `soft_rate`, `calib_tau` |
| `georvq/model/conditioning.py` | `TierPredictor` (cascade), `CodebookSelector` (gated subset) |
| `georvq/model/location.py` | SatCLIP clone-on-demand loader, location tables, location→latent prior MLPs |
| `georvq/data/` | `store.py` chip-store v1 + `geographic_split`; `pool.py` resident pools; `build.py` `ShardWriter` |
| `georvq/sources/` | `Source` interface + registry; `ae.py` (minimal reference), `s2.py` (multi-resolution example) |
| `georvq/train/` | `loop.py` the one trainer; `stage1.py` σ-conditioned pretrain; `validate.py`, `figures.py`, `warmstart.py` |
| `georvq/eval/` | Held-out extraction, probes, benchmark registry, codec A/B helpers |
| `georvq/viz/` | Panel/GIF styling and builders |
| `georvq/inference/` | `api.py` Codec/Packet, `stats_codec.py`, `archive.py` rANS, `release.py` bundles, `onnx_backend.py` ONNX export + torch-free runtime, `cli.py` codec executables |
| `scripts/` | Thin drivers over the above (train, store builders, eval, archive, GIFs) |
| `configs/` | Presets — each reproduces a specific historical run (see Provenance) |

## Invariants — do not break casually

1. **VQ numerics are parity-sacred.** The repo's value rests on trained
   checkpoints decoding bit-faithfully. `model/vq.py` (and the quantize
   paths in `codec.py`/`conditioning.py`) are covered by parity tests
   against the original research implementation (`pytest -m ringspace`,
   self-skipping where that repo is absent). Any change there must keep
   those tests exact, not approximately equal.
2. **State-dict names are load-bearing.** `compat.py` maps legacy
   checkpoint keys onto today's module names by prefix. Renaming modules
   or parameters breaks every existing checkpoint; if you must, extend
   the rename table and prove a legacy checkpoint still loads strict.
3. **The bitstream excludes the decoder.** A packet/archive is defined by
   tiers + codebooks + conditioning priors + stats codec + CDFs
   (`docs/BITSTREAM.md`). Decoder-side improvements must never require
   re-encoding; anything that changes the bitstream changes
   `bitstream_id`.
4. **VQ state stays fp32 under bf16 autocast.** Assignment scores may run
   bf16 (`vq.bf16_scores`); residuals, EMA buffers and codebooks must not.
5. **Sources own all data-specific behavior.** The shared stack
   (tokenizer→decoder, trainer, evals, inference) must stay
   source-agnostic. If you're adding an `if source == ...` anywhere
   outside `sources/`, it belongs in the adapter instead.
6. **Configs are the reproducibility spine.** Every training artifact
   embeds its resolved config; anything that affects a run must live in
   the `Config` schema (validated, `extra="forbid"`), not in an env var
   or an edit-the-source knob.

## Running things

```bash
pip install -e ".[loc,archive,eval,viz,dev]"

georvq-train -c configs/pyr64.yaml --gpu 0 log.run_tag=my_run   # train
georvq-train -c configs/pyr64.yaml train.steps=2000 ...          # any override

pytest -m "not slow and not gpu"    # fast suite (~6 s) — run on every change
pytest -m "not slow"                # + ringspace parity (needs the research repo)
pytest                              # + training smokes (GPU, minutes)

python scripts/eval_run_ab.py --dataset <ds> --sanity identity   # eval harness sanity
python scripts/eval_rate_sweep.py --dataset <ds> --ckpt <ckpt>   # Pareto-front A/B
python scripts/build_archive.py --ckpt <ckpt> --store-dir <dir> --out <dir>

# deployment executables (AlphaEarth; --backend torch|onnx, --device cpu|cuda)
georvq-export-onnx --ckpt <ckpt|bundle> --out <onnx-dir>       # + parity verify
georvq-codec-patch encode --model <ckpt|bundle|onnx-dir> windows.npy out.gvq.npz
georvq-codec-tile  encode --model <...> tile.{npy,tif} out.gvq.npz
georvq-codec-tile  decode --model <...> out.gvq.npz recon.tif --tuple 6,4,4
```

Smoke config for cheap end-to-end checks: `configs/smoke.yaml` (synthetic
store, CPU-friendly). Prefer it — or a few hundred steps with overrides —
over long runs when validating a change; the fast pytest suite plus one
short smoke catches nearly everything.

## Preset provenance

Each preset encodes a recipe that actually ran and worked in the research
cycle, with its settings verified against the original runs' logged argv:

- `pyr64.yaml` — the flagship codec (incl. its k-means warmup).
- `stage1.yaml` — σ-conditioned continuous pretrain.
- `hybrid_rate.yaml` — frozen stage-1 encoder + fresh smaller-deeper
  codebooks + entropy model (its small 256-slot pool is historical).
- `cascade.yaml` — location prior + cross-tier predictors, encoder frozen
  for the whole run (co-adaptation degrades the ceiling — `FINDINGS.md`).
- `satclip_select.yaml` — selector-only on a frozen codec. The
  4×-codebooks-from-scratch variant in its header comment is an
  **unvalidated research direction**: it never trained past k-means
  historically, and a short attempt here diverged. Don't present it as a
  working recipe.

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
- **Multi-group (S2-style) sources** have model-level support and a
  RAM-resident loader; the streaming pool + standard train loop are
  single-plane only for now (`docs/ADDING_A_SOURCE.md` has the working
  pattern and the roadmap).

## Where to extend

- **New dataset** → builder + `Source` adapter + preset:
  `docs/ADDING_A_SOURCE.md`.
- **Better decoder / post-decode enhancement (e.g. seam deblocker)** →
  free under the bitstream contract: `docs/BITSTREAM.md`.
- **Context-adaptive entropy coding** → the achievable rate is already
  measured (`model.entropy_level_bits`); a coder driven by the same
  `ContextRateModel` slots behind the existing packet layout.
- **Why the architecture is shaped this way** → `docs/FINDINGS.md`
  (empirical results, including the negative ones worth not repeating).
