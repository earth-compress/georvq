<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/earth-compress-dark.svg">
    <img src="docs/assets/earth-compress.svg" alt="Earth Compress" width="420">
  </picture>
</p>

# georvq

Pyramidal residual-VQ compression for geospatial datasets.

Prefix-able bitstreams that offer the entire rate–distortion front from a single encode and write: georvq trains
hierarchical (coarse→fine) residual-VQ codecs over gridded geospatial data —
embedding fields (e.g. AlphaEarth, Tessera), multispectral imagery
(e.g. Sentinel-2), elevation, SAR, or any source, though training data is needed.

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
  machinery that keeps codebooks in use, from K=256 (Tessera) to 16,384
  (AlphaEarth).
- **Learned entropy model (exploratory).** An optional per-tier context
  model makes codes cheaper (rate-aware training via soft-rate, gradient
  codebooks) and reports achievable bits.
- **Stats side-channel.** Sources that need per-tile normalization ship
  their stats alongside the indices, and the rate figures charge them
  (S2 sends its 26 values as about 10 bytes of PCA-packed stats at low
  rates and as 53 bytes of float16 at high ones). Sources whose windows
  carry most of their energy in the mean can instead *transmit* it and
  code only what is left — cheap, and worth far more per bit than extra
  pyramid depth where it applies.
- **Additive refinements.** A tier's post-RVQ residual can be scalar-
  quantized and added back at decode, buying back the quantizer gap
  without touching the codebooks. Optional by construction: ignoring one
  still decodes.
- **Location conditioning (exploratory).** Predict-and-subtract cascades
  (frozen location→latent prior + cross-tier predictors) and
  location-gated codebook subsetting, via SatCLIP embeddings.
- **Warm starts and staged training.** Warm-start with shape
  reconciliation and encoder/codec freeze schedules; σ-conditioned
  continuous stage-1 pretraining (exploratory).
- **Decoder-independent bitstream.** Packets are defined by tiers +
  codebooks + stats codec only; decoders can be upgraded forever without
  re-encoding (docs/BITSTREAM.md).

Everything is driven by validated pydantic configs; every checkpoint embeds
its full resolved config (plus git state and config hash) and is rebuilt
exactly by `georvq.from_checkpoint()`. Compare any two runs with
`georvq.config.config_diff`.

## Install

```bash
pip install -e .                                    # decoding
pip install -e ".[train]"                           # + training
pip install -e ".[train,eval,viz,onnx,era5,dev]"   # everything
```

The extras are `train`, `tessera`, `baseline`, `eval`, `viz`, `onnx`,
`era5`, `notebooks` and `dev`. The oldest tested versions are Python
3.10, torch 2.10, numpy 2.0, rasterio 1.4 and, in `[onnx]`, onnxruntime
1.19; `[era5]` needs Python 3.11 for zarr 3.

## Quickstart: the AlphaEarth release codec

The most mature part of the codebase currently is the AlphaEarth release. Working with the AE codec to get real data and decode it
takes five steps. For more detailed information about the release, see `docs/AE_RELEASE.md`.

```bash
# 1. install (add ".[onnx]" for ONNX Runtime decode)
git clone <this repo> && cd georvq
pip install -e .

# 2. use it; the weights a file needs are fetched on first read  [pending: hosting]
python -m georvq.ae.v2 info   tile.georvq.parquet
python -m georvq.ae.v2 decode tile.georvq.parquet out.npy --quality 8-8-2
python -m georvq.ae.v2 encode src.tif out_dir/ --bundle <bundle dir> --device cuda
```

```python
import georvq.ae
ds = georvq.ae.open("<url or path>/annual/2021/44N/<stem>.georvq.parquet")
a = ds.read(window=((0, 1024), (0, 1024)), quality="p99<=0.02")
```

The file names the release that decodes it, so when you first `open` a file that uses the codec, the
bundle will be automatically downloaded from Hugging Face. Decoding runs
on torch, or on ONNX Runtime when asked for (`backend="onnx"`).

## Train

Training of models with georvq, as with any repo, requires a lot of data set up before it can run. If you're interested in training a georvq model for new sources, check out the section below that has some details about the data format used during training.

```bash
# standard 64px 3-tier pyramid on an AlphaEarth chip store
georvq-train -c configs/ae/pyr64.yaml --gpu 0 log.run_tag=my_run

# override anything with dot paths
georvq-train -c configs/ae/pyr64.yaml model.rate='{weight: 0.005}' train.steps=60000

# stage-1 continuous pretrain, then a rate-model fine-tune on top
georvq-train-stage1 -c configs/ae/stage1.yaml --gpu 0
georvq-train -c configs/ae/hybrid_rate.yaml --gpu 0 train.init_from=checkpoints/stage1/model.pt
```

Presets live in `configs/<source>/`: `ae/` (`pyr64` the standard
recipe, plus the exploratory `stage1`, `hybrid_rate`, `cascade` and
`satclip_select`), `tessera/` (the lineage ending at the released
`tessera_vq8_k256_statscond`), `s2/`, `dem/` and `era5/`, with
`smoke.yaml` (CI) at the top level. A preset's `data.store_dir` points at the
group server; override it for your own store.

## Data

Sources are converted once into a standard chip store (sharded memmaps +
parquet index + versioned manifest); training streams it through a
GPU/CPU-resident pool with background refresh (whole-dataset FIFO sweeps
for multi-TB stores). Builders: `scripts/build_store_{ae,tessera,s2,dem,era5}.py`;
S2 and DEM can also be read in place without building a store. Adding a
source = builder + `Source` adapter + YAML preset — see
`docs/ADDING_A_SOURCE.md`, and `docs/SOURCE_ONBOARDING.md` for measuring
the source *before* choosing an architecture for it.

## Evaluate

- `scripts/eval_rate_sweep.py` — downstream probe metrics across the whole
  R-D front (frozen-embedding kNN/linear probes on contamination-checked
  held-out windows, geographic splits), with identity and noise sanity
  harnesses (`--sanity identity|noise`).
- `scripts/baseline_classical.py` — classical codec baselines at actual
  file bytes.

## Ship

Release bundles (`georvq.inference.release`) are self-contained
directories (weights + embedded config + `bitstream_id`) that decode
anywhere georvq is installed:

```python
from georvq.inference.release import load_release
codec = load_release("releases/my_run", device="cuda")
xhat = codec.decode(codec.encode(x), depths=[8, 4, 0])
```

On top of the generic machinery, each shipped source gets its own
**release layer** with source-appropriate containers, side channels and
executables. Two are shipped, and they differ in every place the data
differs.

The AlphaEarth release (`georvq/ae/v2.py`, `docs/AE_RELEASE.md`) codes
each source tile with a model that transmits each window's mean (440 bits per window),
into one GeoParquet file per tile (`docs/GEORVQ_PARQUET.md`). The
reader stands in for the source tile's rasterio dataset, fetches only
the byte ranges a window and rate need, and picks a rate from 23 rungs
(0.11-7.5 b/px) or from a bound on encode-time measured error:

```python
import georvq.ae
ds = georvq.ae.open(url)
a = ds.read(window=((0, 1024), (0, 1024)), quality="8-8-2")
```

The Tessera release (`georvq/tessera/`, `docs/TESSERA_RELEASE.md`) codes
128-band embedding windows across 0.23–10.9 b/px, transmitting the patch
mean as a side channel and offering an opt-in fine-tier refinement:

```python
from georvq.tessera import load
R = load("/data/georvq-releases/tessera-beta-v1/bundle", device="cuda")
recon = R.decode(R.encode(x, valid=v), depths=[20, 20, 5], valid=v)
```

## Docs

- `docs/OVERVIEW.md` — codebase map, invariants, personas, working knowledge
- `docs/AE_RELEASE.md` — the shipped AlphaEarth codec's complete contract
- `docs/TESSERA_RELEASE.md` — the shipped Tessera codec's complete contract
- `docs/ADDING_A_SOURCE.md` — the plug-point for new datasets
- `docs/SOURCE_ONBOARDING.md` — measuring a new source into an architecture
- `docs/BITSTREAM.md` — what defines a bitstream; decoder-upgrade contract
- `docs/DESIGN.md` — how the method works and why it is built that way
- `lab/` — the findings ledger: empirical results with their provenance (`lab/query.py` filters them)

## Tests

```bash
pytest -m "not slow"   # fast suite, training smokes included (about 80 s on 8 CPUs)
pytest                 # adds the resume and real-checkpoint tests
```

The golden decode tests (`tests/test_ae_v2_bundle.py` for the AlphaEarth
release, `tests/test_tessera_bundle.py` for the Tessera release) verify
that released weights still encode and decode bit for bit; they
self-skip where the bundle isn't present (`GEORVQ_AE_V2_BUNDLE`,
`GEORVQ_AE_BUNDLE`, `GEORVQ_TESSERA_BUNDLE`). The recorded hashes hold
on torch 2.10 or later with numpy 2; older versions give different codes.

## Development

georvq is developed by Asterisk Laboratories Co-operative Ltd. (Asterisk
Labs) as part of the Earth Compress project, a partnership between
Asterisk Labs and the National Oceanography Centre.

<p>
  <a href="https://asterisk.coop">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/asterisk-labs-dark.svg">
      <img src="docs/assets/asterisk-labs.svg" alt="Asterisk Labs" width="220">
    </picture>
  </a>
</p>

## Funding

This work was funded by the Advanced Research and Invention Agency
(ARIA).

## Licence

The code is licensed under the Apache License 2.0 (`LICENSE`), as are the
model weights on Hugging Face. A few files are derived from third-party
datasets and carry those datasets' terms instead: test fixtures and
notebook outputs from AlphaEarth (CC-BY 4.0) and TESSERA (CC-BY-SA 4.0),
a PCA basis fitted on Major TOM Core-S2L1C (CC-BY-SA 4.0), and ERA5
statistics (CC-BY 4.0). `NOTICE` lists each file with its attribution;
the licence texts are in `LICENSES/`. Data encoded or decoded with the AE
codec is derived from the AlphaEarth dataset and carries its CC-BY 4.0
licence and attribution.
