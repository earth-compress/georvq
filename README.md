# georvq

Draft for review: the AE v2 quickstart, the AE v2 paragraph under Ship and its Docs entry are new and not yet approved.

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
- **Learned entropy model + honest bytes (exploratory).** An optional per-tier context
  model makes codes cheaper (rate-aware training via soft-rate, gradient
  codebooks) and reports achievable bits; `georvq.inference.archive`
  writes real rANS files with byte-exact verification so claimed ratios
  are realized ones.
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
  location-gated codebook subsetting, via SatCLIP embeddings
  (`georvq[loc]`).
- **Warm starts and staged training.** Warm-start with shape
  reconciliation and encoder/codec freeze schedules; σ-conditioned
  continuous stage-1 pretraining (exploratory).
- **Decoder-independent bitstream.** Packets are defined by tiers +
  codebooks + stats codec only; decoders can be upgraded forever without
  re-encoding (docs/BITSTREAM.md).

Which of these a shipped release depends on, and which are exploratory,
is tracked per feature in `docs/SOURCES.md`.

Everything is driven by validated pydantic configs; every checkpoint embeds
its full resolved config (plus git state and config hash) and is rebuilt
exactly by `georvq.from_checkpoint()`. Compare any two runs with
`georvq.config.config_diff`.

## Install

```bash
pip install -e .                              # core
pip install -e ".[loc,archive,eval,viz,dev]"  # everything
```

## Quickstart: the AlphaEarth release codec

Working with the shipped AE codec (decode archive tiles, run the demos)
takes five steps; full contract in `docs/AE_RELEASE.md`.

```bash
# 1. install (any env with torch; on the group server one is already set up)
git clone <this repo> && cd georvq
pip install -e ".[eval,onnx]"

# 2. get the weights (NOT in git)
#    on the group server they are already at /data/georvq-releases/ae-beta-v1
#    elsewhere: download ae-beta-v1.tar.gz from the Drive link (ask Ali /
#    see the release thread), then verify before use:
sha256sum -c <<< "d535fe3c692327acbafcfd8816be9d776d4596fe02ea18e6963759fbbbf969b5  ae-beta-v1.tar.gz"

# 3. extract ANYWHERE and point the env var at it
tar xzf ae-beta-v1.tar.gz
export GEORVQ_AE_BUNDLE=$PWD/ae-beta-v1/bundle   # add to your shell rc

# 4. use it
georvq-ae-tile info   some_tile_assets/                       # inspect a blob
georvq-ae-tile decode --model "$GEORVQ_AE_BUNDLE" some_tile_assets/ recon.tiff \
    --quality "p90<=0.02"                                     # or --tuple 8,8,2
georvq-ae-tile encode --model "$GEORVQ_AE_BUNDLE" --device cuda src.tiff out/

# 5. demos (pre-executed with figures — readable without any data)
jupyter lab notebooks/demo_ae_codec.ipynb notebooks/demo_downstream_eval.ipynb
```

Notes: every CLI also accepts `--model path/...` directly (the env var
is only the notebooks' default); decode-only use needs no GPU — pass
`--backend onnx --model .../ae-beta-v1/onnx` for the torch-free path;
`bundle/model.pt` is a torch pickle, so always verify the checksum.

## Quickstart: the AlphaEarth v2 release codec

AE v2 codes each source tile as one `{stem}.georvq.parquet` file that
reads like the source tile, at 23 rates; full contract in
`docs/AE_V2_RELEASE.md`. v1 above is unchanged.

```bash
# 1. install
git clone <this repo> && cd georvq
pip install -e ".[eval]"

# 2. get the weights from Hugging Face   [pending: hosting]
huggingface-cli download earth-compress/georvq-ae-v2 --local-dir georvq-ae-v2

# 3. use it (--model/--channel until the bundle loader lands   [pending: bundle])
python -m georvq.ae.v2 info   tile.georvq.parquet
python -m georvq.ae.v2 decode tile.georvq.parquet out.npy \
    --model <checkpoint> --channel <mean channel .npz> --quality 8-8-2
python -m georvq.ae.v2 encode src.tif out_dir/ \
    --model <checkpoint> --channel <mean channel .npz> --device cuda
```

```python
import georvq.ae
ds = georvq.ae.open("<url or path>/annual/2021/44N/<stem>.georvq.parquet",
                    model="<checkpoint>", channel="<mean channel .npz>")
a = ds.read(window=((0, 1024), (0, 1024)), quality="p99<=0.02")
```

Decoding needs torch for now; torch-free decode through ONNX is
planned `[pending: onnx]`.

## Train

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
`smoke.yaml` (CI) at the top level. `docs/SOURCES.md` names the current
preset for each source. A preset's `data.store_dir` points at the
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
xhat = codec.decode(codec.encode(x), depths=[8, 4, 0])
```

On top of the generic machinery, each shipped source gets its own
**release layer** with source-appropriate containers, side channels and
executables. Two are shipped, and they differ in every place the data
differs.

The AlphaEarth release (`georvq/ae/`, `docs/AE_RELEASE.md`) turns a
2.25 GB source tile into ~58 MB of georeferenced COG assets with
progressive rate, quality-controlled decode against encode-time
measured error stats, and torch-free ONNX decoding:

```bash
georvq-ae-tile encode --model <bundle> src.tiff out_dir/
georvq-ae-tile decode --model <bundle> out_dir/ recon.tiff --quality "p90<=0.02"
```

AE v2 (`georvq/ae/v2.py`, `docs/AE_V2_RELEASE.md`) codes the same tiles
with a model that transmits each window's mean (440 bits per window),
into one GeoParquet file per tile (`docs/GEORVQ_PARQUET.md`). The
reader stands in for the source tile's rasterio dataset, fetches only
the byte ranges a window and rate need, and picks a rate from 23 rungs
(0.11-7.5 b/px) or from a bound on encode-time measured error:

```python
import georvq.ae
ds = georvq.ae.open(url, model=ckpt, channel=mean_npz)   # [pending: bundle]
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
- `docs/AE_V2_RELEASE.md` — the AlphaEarth v2 codec's contract (draft)
- `docs/TESSERA_RELEASE.md` — the shipped Tessera codec's complete contract
- `docs/ADDING_A_SOURCE.md` — the plug-point for new datasets
- `docs/SOURCE_ONBOARDING.md` — measuring a new source into an architecture
- `docs/BITSTREAM.md` — what defines a bitstream; decoder-upgrade contract
- `docs/DESIGN.md` — how the method works and why it is built that way
- `docs/SOURCES.md` — where each data source currently stands
- `lab/` — the findings ledger: empirical results with their provenance (`lab/query.py` filters them)

## Tests

```bash
pytest -m "not slow and not gpu"   # fast suite
pytest                             # everything (training smokes included)
```

The golden-checkpoint gate (`tests/test_compat.py`) verifies the released
weights still decode bit-faithfully; it self-skips where the release
bundle isn't present (`GEORVQ_AE_BUNDLE`).
