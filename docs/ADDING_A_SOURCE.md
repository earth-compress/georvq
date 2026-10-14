# Adding a data source

This document is for anyone who wants to train georvq on a new dataset.
Everything specific to a dataset is kept in three pieces, which are a
store builder (or a live reader), a `Source` adapter and a YAML preset.
The pyramid, RVQ, rate model, decoder trunk, training loop, evals and
inference are shared. Sections 0 to 3 describe the three pieces, and
section 4 covers the release layer a source needs if its codec is to be
shipped.

Before writing any of it, measure the source.
`docs/SOURCE_ONBOARDING.md` turns a handful of cheap measurements into
the architecture parameters below, so that window size, tier grids,
latent widths and codebook sizes are derived from the data rather than
inherited from whichever preset you copied. The measurements are
intrinsic dimensionality, autocorrelation range, the lossless floor and
magnitude spread.

## 0. Whether a store is needed

A chip store exists to give the pool fixed-geometry chips it can read
cheaply. If the source is already laid out that way, building one
duplicates the corpus for no gain. Three live readers do without a
store:

| reader | source | why live |
|---|---|---|
| `live_dem` (`data/live_dem.py`) | GLO-30 COG mosaics | chips are windows on a continuous mosaic |
| `live_s2` (`data/live_s2.py`) | Major-TOM Core-S2L1C | already 106k scene dirs on a regular 10 km grid |
| `era5` (`data/era5.py`) | ERA5 cube zarr | hourly variable slices on a fixed global grid |

All three implement the same interface as `ChipStore`, with the same
attributes, the `open_shard(sh, group)[slot]` contract and `split()`, so
the pool, the trainer and the evals use them without changes.
`open_store` picks one, choosing `live_dem` and `era5` from the `kind` in
the store's `manifest.json`, and Major-TOM by its directory layout.
`data.reader` can pin `chipstore` or `live_s2`.

Write a live reader when the source's own layout already has fixed
geometry and a usable index, and a store builder when it does not (chips
must be cut, reprojected, or dequantised once). Before choosing, check
how the source's files are blocked and compressed, because that sets the
smallest read that is actually cheap. Major-TOM's rasters are LZW with
1024 px blocks over a 1068 px image, so a windowed read decompresses the
whole band anyway, and `live_s2` therefore reads whole scenes rather
than windows.

## 1. A store builder

Convert your source once into the standard on-disk format, chip-store v1
(`georvq/data/store.py`). This is a directory of `shard_XXXX.npy`
memmaps `(shard_size, bands, px, px)`, an `index.parquet` with one row
per chip (`chip_id, shard, slot, lat, lon, crs, year`, plus any
provenance columns you want), and a versioned `manifest.json` (geometry,
dtype, nodata, `grid_period`, attribution). Use
`georvq.data.build.ShardWriter`, which handles shard allocation, atomic
index checkpoints, and append/resume with dedup.
`scripts/build_store_ae.py` is the example for dense embedding chips
sampled from cloud COGs. `scripts/build_store_s2.py` builds a
multi-resolution store, with one shard file per band group and a shared
index. It is kept for anyone who wants S2 as shards, although the
shipped S2 preset reads Major-TOM live instead (see section 0).

Two further points apply to every store.

- `grid_period`: if your pixels have periodic structure (e.g. a source
  upsampled 2x onto a finer grid), record the period. Aligned cropping
  (`data.pool.align_crops`) then keeps every training crop at the same
  phase, so the model can reproduce the structure instead of averaging
  it away.
- Store raw integer data with a dequantisation recipe. Dequantisation
  happens on the device in `Source.prep`, which keeps the resident pool
  small.

## 2. A `Source` adapter

Subclass `georvq.sources.Source` and add it to the `SOURCES` table in
`get_source()` (`georvq/sources/__init__.py`), which maps the
`data.source` name to the class. ERA5 shows the other pattern, in which
the name `era5:<var>` carries a parameter the adapter needs. The adapter
implements the following methods.

- `prep(raw, device) -> (x, valid)` converts the store dtype to
  model-space floats and returns a validity mask, so nodata handling
  belongs here.
- `build_stem(cfg)` maps the raw plane(s) to the `(B, C, W, W)` feature
  plane the shared ViT tokeniser expects. This is where multi-resolution
  band groups get their native-GSD conv stems (see `S2NativeStem`).
  Dense single-plane sources use `nn.Identity()`.
- `build_head(cfg, in_dim, n_tiers=None, out_px=None)` maps the
  decoder's fine tokens to pixel output(s). `n_tiers`/`out_px` are set
  when the head belongs to a decoder-bank member that decodes a prefix
  of the tiers at a smaller size, and `Source.head_geom` turns them into
  the token grid and patch size. `DenseHead` (single plane, optional
  per-pixel L2 norm) covers embedding fields, and `S2MultiResHead`
  decodes each native grid.
- `normalizer(cfg)` returns the per-tile stats side channel, or `None`
  if your data is already normalised (unit-norm embedding fields). See
  Stats side channels below.
- `recon_stats(out, x, valid) -> (loss, metrics)` returns the training
  loss and the quality metrics in model space.
- `report(cfg) -> ReportSpec` declares what a run on this source should
  measure, plot and log (`georvq/sources/report.py`). See Reporting
  below.

`georvq/sources/ae.py` is the minimal reference (about 80 lines), and
`georvq/sources/s2.py` is the full multi-resolution example.

### Stats side channels

A source with a normaliser also implements `last_stats()`, which returns
the stats of the batch `prep` just normalised (the figures need them to
plot physical units), and `side_bits(cfg)`, the per-window bits the
stats cost. `side_bits` is what charges the channel, because
`PyramidCodec.side_bits` adds it to every rate figure. Without it, the
rate figures would leave out bits that a real file has to carry, and
those bits are not small. In an earlier system, which did not charge its
stats channel, the channel reached 34% of the bitstream before anyone
noticed. How the stats are encoded is the source's choice. S2 charges
them at the size `georvq.inference.stats_codec.StatsPacker` packs them
to (about 10 bytes at low rates), and DEM charges two float32 values.

A normaliser whose stats are per sample sets `per_sample = True`, so
that `SourcePrep.inverse` takes the batch's own stats and
`SourcePrep.invertible` is False. Constants the source fits once (for
example the size of S2's PCA stats channel) are embedded in the
checkpoint through `prep_state()` and restored by
`load_prep_state(state)`.

A separate option, `model.stats_cond`, transmits the per-window mean and
codes only the remainder. It pays when the mean is a large share of
window energy. On a unit-norm field such as AlphaEarth the decoder
renormalises after the mean is added back. The codec charges that
channel itself.

### Reporting

There is deliberately no default `report`, because the metrics have to
suit the kind of data. Reporting inherited from the previous source is
how a single-band elevation model once ended up logging a cosine, so a
source that declares nothing raises an error. Two ready-made shapes
cover most cases, `embedding_field_spec()` (cosine, centred cosine,
block-mean floor and PCA-RGB preview) and `scalar_field_spec()` (R²,
RMSE, no block-mean, greyscale preview).

A spec names the headline `quality` metric, which labels the val scalars
and the R-D axis and selects `model_best.pt`, together with `extra`
diagnostics, `baselines`, `figures` and a `preview`. Every metric and
preview declares the `min_bands` it is defined for, so cosine on one
band (where it collapses to sign agreement) or a 3-component PCA-RGB
preview of one band is refused at config resolution rather than logged.
A run narrows the surface with
`log.report.{metrics,baselines,figures,preview}`. Those settings may only
take things away, and naming something the source does not declare is a
config error.

### Samplers and fitted preps

Cutting the sample and transforming it to model space can also be done
by two separate pieces, rather than both by `prep`. A sampler
(`georvq/data/sampler.py`, a single `WindowSampler` with no config of its
own) cuts the sample from the pool, and a prep (`data.prep`,
`georvq/sources/prep.py`) is a static, fitted transform to model space
with a declared inverse. Sources are moving to this one at a time, and
ERA5's Box-Cox prep is the first. Leaving `data.prep` unset gives the
behaviour described above.

## 3. A YAML preset

Copy the closest preset from `configs/<source>/`, set `data.source`,
`data.store_dir`, `model.window/bands/tiers`, and train. Presets'
`store_dir` values point at the group server and need overriding
elsewhere.

```bash
georvq-train -c configs/<source>/my_preset.yaml --gpu 0
```

## 4. A release layer, when shipping

The three pieces above are all that training needs. If the codec is to
be shipped, a release layer has to decide the container format, the rate
and quality controls, the side channels, the refinements and the
executables. These decisions are made per source, on top of the generic
`georvq.inference` code.

There are two worked examples, and reading them side by side is the
quickest way to see which decisions are policy rather than architecture:

| | AlphaEarth (`georvq/ae/`) | Tessera (`georvq/tessera/`) |
|---|---|---|
| container | one GeoParquet file per source tile | packets; no container yet, none required |
| stats channel | the window mean, 440 bits/window (RVQ + PCA), charged | the patch mean, 768 bits/window, charged |
| refinement | none | fine tier, 1-bit rotated Lloyd, opt-in (+150% rate) |
| quality control | encode-time measured error ladder | not built, since no user has asked for it |

Each of those differences was measured on the data, and the refinement
row shows how far the right choice can differ between sources. The
coarse-tier refinement that recovered ~98% of the first AlphaEarth
release's quantiser gap recovers 0.0% on Tessera. Before copying either
policy, locate your model's slack by comparing full-depth reconstruction
against the unquantised ceiling, per tier.

## Current integration status

Single-plane sources (one `(bands, px, px)` array per chip) are fully
supported. The streaming `ChipPool` (coverage and cycle modes), the
training loop, validation, figures, eval and inference all work with
them without changes.

Multi-group sources (dict batches, e.g. S2) go through the same path.
`ChipPool` holds one resident tensor per group, all indexed by the same
slot, and `sample_batch` returns a `GroupBatch`, a dict of per-group
crops sharing one crop origin. Residency (coverage/cycle), the shuffled
full-dataset cursor and the background refresh do not depend on the
groups and are not duplicated. Only allocation, the shard read and the
gather divide by each group's ratio.

The geometry imposes two constraints, and the code asserts both rather
than assuming them.

- Crop origins snap to the coarsest ratio. One origin serves every
  group, so on S2 it must be a multiple of 6, otherwise a 20m/60m crop
  would begin part-way into its own pixel and the groups would decode
  out of phase. Multi-group `sample_batch` is therefore always aligned.
- `slot_rows` must divide by that period, so that each group's row-strip
  is a whole number of its own rows.

A source's `prep`, head output and normaliser should return `GroupBatch`
rather than a bare dict, because the shared stack slices batches, reads
`.shape[0]` and calls `.to()`/`.float()` on them, and `GroupBatch`
implements those.
