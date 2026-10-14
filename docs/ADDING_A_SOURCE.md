# Adding a data source

Everything dataset-specific lives in three pluggable pieces; the pyramid,
RVQ, rate model, decoder trunk, training loop, evals and inference are
shared. To onboard a new source you write:

Before writing any of it, measure the source. `docs/SOURCE_ONBOARDING.md`
turns a handful of cheap measurements — intrinsic dimensionality,
autocorrelation range, lossless floor, magnitude spread — into the
architecture parameters below, so that window size, tier grids, latent
widths and codebook sizes are derived rather than inherited from
whichever preset you copied.

## 0. First ask whether you need a store at all

A chip store exists to give the pool fixed-geometry chips it can read
cheaply. If the source is *already* laid out that way, building one
duplicates the corpus to gain nothing. Three live readers do without:

| reader | source | why live |
|---|---|---|
| `live_dem` (`data/live_dem.py`) | GLO-30 COG mosaics | chips are windows on a continuous mosaic |
| `live_s2` (`data/live_s2.py`) | Major-TOM Core-S2L1C | already 106k scene dirs on a regular 10 km grid |
| `era5` (`data/era5.py`) | ERA5 cube zarr | hourly variable slices on a fixed global grid |

All three present the `ChipStore` surface — the same attributes, the
`open_shard(sh, group)[slot]` contract and `split()` — so the pool, the
trainer and the evals cannot tell the difference. `open_store` picks one:
`live_dem` and `era5` from the `kind` in the store's `manifest.json`, and
Major-TOM by its directory layout. `data.reader` can pin `chipstore` or
`live_s2`.

Write a live reader when the source's own layout already has fixed
geometry and a usable index, and a store builder when it does not (chips
must be cut, reprojected, or dequantized once). Note what the read
granularity actually is before choosing: Major-TOM's rasters are LZW with
1024 px blocks over a 1068 px image, so a windowed read decompresses the
whole band anyway and `live_s2` reads whole scenes rather than pretending
windows are cheap.

## 1. A store builder → chip-store v1

Convert your source once into the standard on-disk format
(`georvq/data/store.py`): a directory of `shard_XXXX.npy` memmaps
`(shard_size, bands, px, px)`, an `index.parquet` with one row per chip
(`chip_id, shard, slot, lat, lon, crs, year`, plus any provenance columns
you want), and a versioned `manifest.json` (geometry, dtype, nodata,
`grid_period`, attribution). Use `georvq.data.build.ShardWriter` — it
handles shard allocation, atomic index checkpoints, and append/resume with
dedup. See `scripts/build_store_ae.py` (dense embedding chips sampled from
cloud COGs) and `scripts/build_store_s2.py` (multi-resolution store: one
shard file per band group, shared index — kept for anyone who wants S2 as
shards, though the shipped S2 preset reads Major-TOM live instead; see
section 0).

Notes:
- `grid_period`: if your pixels have periodic structure (e.g. a source
  upsampled 2x onto a finer grid), record the period; aligned cropping
  (`data.pool.align_crops`) keeps every training crop at the same phase so
  the model can reproduce the structure instead of averaging it away.
- Store raw integer data + a dequantization recipe; dequant happens
  on-device in `Source.prep` (keeps the resident pool small).

## 2. A `Source` adapter

Subclass `georvq.sources.Source` and add it to the `SOURCES` table in
`get_source()` (`georvq/sources/__init__.py`), which maps the
`data.source` name to the class. ERA5 shows the other pattern: the name
`era5:<var>` carries a parameter the adapter needs.

- `prep(raw, device) -> (x, valid)` — store dtype to model-space floats
  plus a validity mask (nodata handling lives here).
- `build_stem(cfg)` — raw plane(s) → the `(B, C, W, W)` feature plane the
  shared ViT tokenizer expects. This is where multi-resolution band groups
  get their native-GSD conv stems (see `S2NativeStem`). Dense single-plane
  sources use `nn.Identity()`.
- `build_head(cfg, in_dim, n_tiers=None, out_px=None)` — decoder fine
  tokens → pixel output(s). `n_tiers`/`out_px` are set when the head
  belongs to a decoder-bank member that decodes a prefix of the tiers at
  a smaller size; `Source.head_geom` turns them into the token grid and
  patch size. `DenseHead` (single plane, optional per-pixel L2 norm)
  covers embedding fields; `S2MultiResHead` decodes each native grid.
- `normalizer(cfg)` — per-tile stats side channel, or `None` if your data
  is already normalized (unit-norm embedding fields). A source with one
  also implements `last_stats()`, returning the stats of the batch `prep`
  just normalized (the figures need them to plot physical units), and
  `side_bits(cfg)`, the per-window bits the stats cost. `side_bits` is
  what charges the channel: `PyramidCodec.side_bits` adds it to every
  rate figure, and an uncharged stats channel once reached 34% of a
  bitstream before anyone noticed. How the stats are encoded is the
  source's choice: S2 packs them with
  `georvq.inference.stats_codec.StatsPacker` (about 10 bytes at low
  rates), DEM sends two float32 values.
  A normalizer whose stats are per sample sets `per_sample = True`, so
  that `SourcePrep.inverse` takes the batch's own stats and
  `SourcePrep.invertible` is False. Constants the source fits once (for
  example the size of S2's PCA stats channel) are embedded in the
  checkpoint through `prep_state()` and restored by
  `load_prep_state(state)`.
  A separate option, `model.stats_cond`, *transmits* the per-window mean
  and codes only the remainder; it pays when the mean is a large share of
  window energy. On a unit-norm field such as AlphaEarth the decoder
  renormalises after the mean is added back. The codec charges that
  channel itself.
- `recon_stats(out, x, valid) -> (loss, metrics)` — the training loss and
  quality metrics in model space.
- `report(cfg) -> ReportSpec` — what a run on this source should measure,
  plot and log (`georvq/sources/report.py`). There is deliberately no
  default: reporting inherited from the previous source is how a
  single-band elevation model ended up logging a cosine, so a source that
  declares nothing raises. Two ready-made shapes cover most cases —
  `embedding_field_spec()` (cosine + centred cosine + block-mean floor +
  PCA-RGB preview) and `scalar_field_spec()` (R² + RMSE, no block-mean,
  grayscale preview).

  A spec names the headline `quality` metric — it labels the val scalars
  and the R-D axis, and selects `model_best.pt` — plus `extra`
  diagnostics, `baselines`, `figures` and a `preview`. Every metric and
  preview declares the `min_bands` it is defined for, so cosine on one
  band (it collapses to sign agreement) or a 3-component PCA-RGB preview
  of one band is refused at config resolution rather than logged. A run
  narrows the surface with `log.report.{metrics,baselines,figures,preview}`;
  those may only take things away, and naming something the source does
  not declare is a config error.

`georvq/sources/ae.py` is the minimal reference (about 80 lines);
`georvq/sources/s2.py` is the full multi-resolution example.

A newer layer separates two things `prep` used to do at once: a
sampler (`georvq/data/sampler.py`, a single `WindowSampler` with no
config of its own) cuts the sample from the pool, and a prep
(`data.prep`, `georvq/sources/prep.py`) is a
static, fitted transform to model space with a declared inverse. Sources
are moving to it one at a time; ERA5's Box-Cox prep is the first, and
leaving `data.prep` unset gives the behaviour described above.

## 3. A YAML preset

Copy the closest preset from `configs/<source>/`, set `data.source`,
`data.store_dir`, `model.window/bands/tiers`, and train. Presets'
`store_dir` values point at the group server and need overriding
elsewhere.

```bash
georvq-train -c configs/<source>/my_preset.yaml --gpu 0
```

## 4. (When shipping) a release layer

Training-side, the three pieces above are everything. If the codec is to
be *shipped*, deployment policy — container format, rate/quality
controls, side channels, refinements, executables — is deliberately
per-source, built on the generic `georvq.inference` machinery.

There are two worked examples, and reading them side by side is the
fastest way to see which decisions are policy rather than architecture:

| | AlphaEarth (`georvq/ae/`) | Tessera (`georvq/tessera/`) |
|---|---|---|
| container | one GeoParquet file per source tile | packets; no container yet, none required |
| stats channel | the window mean, 440 bits/window (RVQ + PCA), charged | the patch mean, 768 bits/window, charged |
| refinement | none | fine tier, 1-bit rotated Lloyd, opt-in (+150% rate) |
| quality control | encode-time measured error ladder | not built — nobody has asked |

Every one of those differences is a measured response to the data, and
the refinement row is the cautionary one: the coarse-tier refinement
that recovered ~98% of the first AlphaEarth release's quantizer gap
recovers 0.0% on Tessera.
Locate your model's slack (compare full-depth reconstruction against the
un-quantized ceiling, per tier) before copying either.

## Current integration status

Single-plane sources (one `(bands, px, px)` array per chip) are fully
wired: streaming `ChipPool` (coverage + cycle modes), the training loop,
validation, figures, eval and inference all work out of the box.

Multi-group sources (dict batches, e.g. S2) are wired through the same
path. `ChipPool` holds one resident tensor per group, all indexed by the
same slot, and `sample_batch` returns a `GroupBatch` — a dict of per-group
crops sharing one crop origin. Residency (coverage/cycle), the shuffled
full-dataset cursor and the background refresh are group-agnostic and are
not duplicated; only allocation, the shard read and the gather divide by
each group's ratio.

Two rules the geometry imposes, both asserted rather than assumed:

- **Crop origins snap to the coarsest ratio.** One origin serves every
  group, so on S2 it must be a multiple of 6 or a 20m/60m crop would begin
  part-way into its own pixel and the groups would decode out of phase.
  Multi-group `sample_batch` is therefore always aligned.
- **`slot_rows` must divide by that period**, so each group's row-strip is
  a whole number of its own rows.

A source's `prep`, head output and normalizer should return `GroupBatch`
rather than a bare dict: the shared stack slices batches, reads
`.shape[0]` and calls `.to()`/`.float()` on them, and `GroupBatch`
implements exactly those. `GroupChipLoader` remains for RAM-resident
working sets.
