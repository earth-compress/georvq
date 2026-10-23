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

Each window is coded by a pyramid of tiers. A tier codes a grid of
latent cells, and each tier has a finer grid than the one before it
(`grid:dim:codes:depth`, e.g. `1:256:2048:8 / 4:192:8192:8 /
16:64:16384:8` over a 64px window), so the finer tiers code residual
detail on top of the coarser ones. The encoder is a ViT tokeniser with
a convolutional patch stem per tier, and an AdaLN attention decoder
mixes all the tiers back into pixels.

Every tier is quantised by residual VQ, so a cell's code is a stack of
levels in which each level corrects the ones before it. Training
samples monotone per-tier depth tuples and conditions the decoder on the
tuple (the matryoshka scheme), so a single checkpoint decodes at any
point on its R-D front. The rate is chosen at decode time by how many
levels of each tier are read.

Left alone, codebook entries fall out of use during training, and an
index into a codebook with dead entries carries fewer bits of
information than it costs. The trainer prevents this with k-means or
data initialisation, EMA decay ramps, dead-code resets and fp32 VQ
state under bf16 training, and has kept codebooks in use at sizes from
K=256 (Tessera) to 16,384 (AlphaEarth).

Sources that need per-tile normalisation have to send their statistics
as well as the indices, and the rate figures charge for them. For S2,
the 26 values are charged as about 10 bytes of PCA-packed stats at low
rates and as 53 bytes of float16 at high ones. Sources whose windows
carry most of their energy in the mean can instead transmit the mean
and code only what is left. On Tessera, where the mean is 85.5% of
per-window energy, the mean channel was about 60 times more
rate-efficient than extra pyramid bits (`docs/SOURCE_ONBOARDING.md`).

A tier's residual after RVQ can also be scalar-quantised and added back
at decode, which recovers some of the quantiser gap without touching
the codebooks. These refinements are optional, and a decoder that
ignores one still produces a valid decode.

Some parts of the codebase are exploratory and are not used by any
release. An optional per-tier context model makes codes cheaper through
rate-aware training (soft-rate, gradient codebooks) and reports the
achievable bits. Location conditioning, via SatCLIP embeddings, comes as
predict-and-subtract cascades (a frozen location-to-latent prior with
cross-tier predictors) and as location-gated codebook subsetting. There
is also a σ-conditioned continuous stage-1 pretraining. Alongside these,
training can warm-start from another checkpoint with shape
reconciliation, and can freeze the encoder or the codec on a schedule.

A packet is defined by the tiers, the codebooks, any conditioning
priors and the release layer's own decode state, such as a mean
channel's tables, and never by the decoder. A newer decoder can
therefore read files written for an older one without re-encoding them
(`docs/BITSTREAM.md`).

Every run is driven by a pydantic config that is validated before
training starts, and every checkpoint embeds its full resolved config,
with the git state and a config hash. `georvq.from_checkpoint()`
rebuilds the model exactly from that config, so any run can be
reproduced, and `georvq.config.config_diff` compares two runs.

## Install

```bash
pip install -e .                                    # decoding
pip install -e ".[train]"                           # + training
pip install -e ".[train,eval,viz,onnx,era5,dev]"   # everything
```

The extras are `train`, `tessera`, `baseline`, `eval`, `viz`, `onnx`,
`era5`, `notebooks` and `dev`. The oldest tested versions are Python
3.12 (zarr-python 3.3 needs it), torch 2.10, numpy 2.0, rasterio 1.4
and, in `[onnx]`, onnxruntime 1.19.

## Quickstart: the AlphaEarth release codec

The most mature part of the codebase currently is the AlphaEarth release. Working with the AE codec to get real data and decode it
takes five steps. For more detailed information about the release, see `docs/AE_RELEASE.md`.

```bash
# 1. install (add ".[onnx]" for ONNX Runtime decode)
git clone <this repo> && cd georvq
pip install -e .

# 2. use it; the weights a store needs are fetched on first read  [pending: hosting]
python -m georvq.ae.v2 info   tile.georvq.zarr
python -m georvq.ae.v2 decode tile.georvq.zarr out.npy --quality 8-8-2
python -m georvq.ae.v2 encode src.tif out_dir/ --bundle <bundle dir> --device cuda
```

```python
import georvq
ds = georvq.open("<url or path>/annual/2021/44N/<stem>.georvq.zarr")
a = ds.read(window=((0, 1024), (0, 1024)), quality="p99<=0.02")
```

The store names the release that decodes it, so when you first `open` a store that uses the codec, the
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

Presets are grouped by source under `configs/<source>/`. `configs/ae/`
holds `pyr64`, the standard recipe, and the exploratory `stage1`,
`hybrid_rate`, `cascade` and `satclip_select`. `configs/tessera/` holds
the lineage ending at the released `tessera_v2`, and
`s2/`, `dem/` and `era5/` hold the presets for those sources.
`smoke.yaml` (CI) is at the top level. A preset's `data.store_dir`
points at the group server, so override it for your own store.

## Data

Each source is converted once into a standard chip store, which holds
the chips in sharded memmaps with a parquet index and a versioned
manifest, so that training reads fixed-geometry chips cheaply instead of
reading the source's own format at every step. Training streams the
store through a GPU- or CPU-resident pool with background refresh, using
whole-dataset FIFO sweeps for multi-TB stores. The builders are
`scripts/build_store_{ae,tessera,s2,dem,era5}.py`, and S2 and DEM can
also be read in place without building a store. To add a source you
write a builder, a `Source` adapter and a YAML preset, as described in
`docs/ADDING_A_SOURCE.md`. `docs/SOURCE_ONBOARDING.md` covers measuring
the source before choosing an architecture for it.

## Evaluate

- `scripts/eval_rate_sweep.py` measures downstream probe metrics across
  the whole R-D front, using frozen-embedding kNN and linear probes on
  contamination-checked held-out windows with geographic splits. It also
  has identity and noise sanity harnesses (`--sanity identity|noise`).
- `scripts/baseline_classical.py` gives classical codec baselines at
  actual file bytes.

## Ship

Release bundles (`georvq.inference.bundle`) are self-contained
directories, holding the weights, the config and the `bitstream_id`,
that decode anywhere georvq is installed:

```python
from georvq.tessera.v2 import load_bundle
codec = load_bundle("georvq-tessera-v2", device="cuda").codec
xhat = codec.decode(codec.encode(x), depths=[20, 13, 0])
```

Each shipped source also has its own release layer on top of this
generic code, which sets the container, the side channels and the
executables for that source. Two are shipped. They share the store
format and the kind of mean channel, and differ in the mean channel's
rate, in the output they return and in the quality statistic, and each
of those choices was measured on its source.

The AlphaEarth release (`georvq/ae/v2.py`, `docs/AE_RELEASE.md`) codes
each source tile with a model that transmits each window's mean (440
bits per window), and writes one Zarr store per tile
(`docs/GEORVQ_ZARR.md`). The reader stands in for the source tile's
rasterio dataset and fetches only the chunks that a window and rate
need. The rate is picked from 23 rungs (0.11-7.5 b/px), or from a bound
on the error measured at encode time:

```python
import georvq
ds = georvq.open(url)
a = ds.read(window=((0, 1024), (0, 1024)), quality="8-8-2")
```

The Tessera release (`georvq/tessera/v2.py`, `docs/TESSERA_RELEASE.md`)
codes each geotessera tile (128 bands) as one Zarr store, readable at 42
rungs across 0.23–9.8 b/px. It transmits each window's mean (896 bits
per window) and returns float32 embeddings:

```python
import georvq
ds = georvq.open("2024/grid_-9.35_30.15.georvq.zarr")
a = ds.read(quality="20-13-0")
```

## Docs

- `docs/OVERVIEW.md` maps the codebase, lists the invariants that
  protect released codecs and collects practical notes, mainly for
  agents working on the code.
- `docs/AE_RELEASE.md` is the full reference for the shipped AlphaEarth
  codec, for data users and archive operators.
- `docs/TESSERA_RELEASE.md` is the same for the shipped Tessera codec.
- `docs/ADDING_A_SOURCE.md` describes the pieces you write to train on a
  new dataset.
- `docs/SOURCE_ONBOARDING.md` describes the measurements of a new source
  that inform its architecture.
- `docs/GEORVQ_ZARR.md` specifies the Zarr store the AE codec
  writes, for anyone reading the stores without the georvq reader.
- `docs/BITSTREAM.md` defines what a bitstream depends on, and why
  decoders can be upgraded without re-encoding.
- `docs/DESIGN.md` explains how the method works and why it is built
  that way.
- `lab/` holds the findings ledger, the empirical results with their
  provenance (`lab/query.py` filters them).

## Tests

```bash
pytest -m "not slow"   # fast suite, training smokes included (about 80 s on 8 CPUs)
pytest                 # adds the resume and real-checkpoint tests
```

The golden decode tests (`tests/test_ae_v2_bundle.py` for the AlphaEarth
release, `tests/test_tessera_v2_bundle.py` for the Tessera release) verify
that released weights still encode and decode bit for bit, and skip
themselves where the bundle is not present (`GEORVQ_AE_V2_BUNDLE`,
`GEORVQ_AE_BUNDLE`, `GEORVQ_TESSERA_V2_BUNDLE`, `GEORVQ_TESSERA_BUNDLE`). The recorded hashes hold
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
notebook outputs from AlphaEarth (CC-BY 4.0), TESSERA (CC-BY-SA 4.0) and
Copernicus Sentinel-2 (the Copernicus Sentinel data terms),
a PCA basis fitted on Major TOM Core-S2L1C (CC-BY-SA 4.0), and ERA5
statistics (CC-BY 4.0). `NOTICE` lists each file with its attribution,
and the licence texts are in `LICENSES/`. Data encoded or decoded with the AE
codec is derived from the AlphaEarth dataset and carries its CC-BY 4.0
licence and attribution.
