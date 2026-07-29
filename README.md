# georvq

Pyramidal residual-VQ compression for geospatial datasets.

One model, one bitstream, a whole rate–distortion front: georvq trains
hierarchical (coarse→fine) residual-VQ codecs over gridded geospatial data —
embedding fields (e.g. AlphaEarth), multispectral imagery (e.g. Sentinel-2),
or any source you write an adapter for.

## The method

- **Pyramid of tiers.** Each tier codes a grid of latent cells at
  increasing spatial resolution (`grid:dim:codes:depth`, e.g.
  `1:256:2048:8 / 4:192:8192:8 / 16:64:16384:8` over a 64px window); finer
  tiers code residual detail on top of coarser ones. A ViT tokenizer with
  per-tier conv patch stems encodes; an AdaLN attention decoder mixes all
  tiers back to pixels.
- **Matryoshka variable rate.** Training samples monotone per-tier depth
  tuples and conditions the decoder on the tuple, so a single checkpoint
  decodes at any point on its R-D front — truncating RVQ levels is the
  rate control.
- **Residual VQ with codebook health.** k-means/data init, EMA decay
  ramps, dead-code resets, fp32 VQ state under bf16 training — the
  machinery that keeps 16k–65k-entry codebooks alive.
- **Learned entropy model + honest bytes.** An optional per-tier context
  model makes codes cheaper (rate-aware training via soft-rate, gradient
  codebooks) and reports achievable bits; `georvq.inference.archive`
  writes real rANS files with byte-exact verification so claimed ratios
  are realized ones.
- **Stats side-channel.** Sources that need per-tile normalization ship
  their stats as ~11-byte adaptive-PCA packets.
- **Location conditioning (optional).** Predict-and-subtract cascades
  (frozen location→latent prior + cross-tier predictors) and
  location-gated codebook subsetting, via SatCLIP embeddings
  (`georvq[loc]`).
- **Staged training.** σ-conditioned continuous stage-1 pretraining,
  warm-start with shape reconciliation, encoder/codec freeze schedules.
- **Decoder-independent bitstream.** Packets are defined by tiers +
  codebooks + stats codec only; decoders can be upgraded forever without
  re-encoding (docs/BITSTREAM.md).

Everything is driven by validated pydantic configs; every checkpoint embeds
its full resolved config (plus git state and config hash) and is rebuilt
exactly by `georvq.from_checkpoint()`. Compare any two runs with
`georvq.config.config_diff`.

## Install

```bash
pip install -e .                              # core
pip install -e ".[loc,archive,eval,viz,dev]"  # everything
```

## Train

```bash
# standard 64px 3-tier pyramid on an AlphaEarth chip store
georvq-train -c configs/pyr64.yaml --gpu 0 log.run_tag=my_run

# override anything with dot paths
georvq-train -c configs/pyr64.yaml model.rate='{weight: 0.005}' train.steps=60000

# stage-1 continuous pretrain, then a rate-model fine-tune on top
georvq-train-stage1 -c configs/stage1.yaml --gpu 0
georvq-train -c configs/hybrid_rate.yaml --gpu 0 train.init_from=checkpoints/stage1/model.pt
```

Presets in `configs/`: `pyr64` (the standard recipe), `stage1`,
`hybrid_rate` (entropy model + gradient codebooks), `cascade` and
`satclip_select` (location conditioning), `smoke` (CI).

## Data

Sources are converted once into a standard chip store (sharded memmaps +
parquet index + versioned manifest); training streams it through a
GPU/CPU-resident pool with background refresh (whole-dataset FIFO sweeps
for multi-TB stores). Builders: `scripts/build_store_ae.py`,
`scripts/build_store_s2.py`. Adding a source = builder + `Source` adapter
+ YAML preset — see `docs/ADDING_A_SOURCE.md`.

## Evaluate

- `scripts/eval_rate_sweep.py` — downstream probe metrics across the whole
  R-D front (frozen-embedding kNN/linear probes on contamination-checked
  held-out windows, geographic splits).
- `scripts/eval_run_ab.py` — original-vs-reconstructed A/B on a benchmark
  (with identity/noise sanity harnesses).
- `scripts/baseline_classical.py` — classical codec baselines at actual
  file bytes.

## Ship

Release bundles (`georvq.inference.release`) are self-contained
directories (weights + embedded config + `bitstream_id`) that decode
anywhere georvq is installed:

```python
from georvq.inference.release import load_release
codec = load_release("releases/my_run", device="cuda")
xhat = codec.decode(codec.encode(x), depths=[8, 4, 2])
```

On top of the generic machinery, each shipped source gets its own
**release layer** with source-appropriate containers and executables.
The AlphaEarth release (`georvq/ae/`, `docs/AE_RELEASE.md`) turns a
2.25 GB source tile into ~58 MB of georeferenced COG assets with
progressive rate, quality-controlled decode against encode-time
measured error stats, and torch-free ONNX decoding:

```bash
georvq-ae-tile encode --model <bundle> src.tiff out_dir/
georvq-ae-tile decode --model <bundle> out_dir/ recon.tiff --quality "p90<=0.02"
```

## Docs

- `docs/OVERVIEW.md` — codebase map, invariants, personas, working knowledge
- `docs/FURTHER_READING.md` — architecture explainer + file-by-file guide
- `docs/AE_RELEASE.md` — the shipped AlphaEarth codec's complete contract
- `docs/ADDING_A_SOURCE.md` — the plug-point for new datasets
- `docs/BITSTREAM.md` — what defines a bitstream; decoder-upgrade contract
- `docs/FINDINGS.md` — the empirical results behind the design choices
- `docs/PARKED.md` — validated-but-deferred ideas, with their evidence

## Tests

```bash
pytest -m "not slow and not gpu"   # fast suite
pytest                             # everything (training smokes included)
```

Tests marked `ringspace` verify numerical parity against the original
research implementation and its trained checkpoints; they self-skip where
that repo isn't present.
