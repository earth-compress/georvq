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
shard file per band group, shared index).

Notes:
- `grid_period`: if your pixels have periodic structure (e.g. a source
  upsampled 2x onto a finer grid), record the period; aligned cropping
  (`data.pool.align_crops`) keeps every training crop at the same phase so
  the model can reproduce the structure instead of averaging it away.
- Store raw integer data + a dequantization recipe; dequant happens
  on-device in `Source.prep` (keeps the resident pool small).

## 2. A `Source` adapter

Subclass `georvq.sources.Source` and register it in
`georvq/sources/__init__.py`:

- `prep(raw, device) -> (x, valid)` — store dtype to model-space floats
  plus a validity mask (nodata handling lives here).
- `build_stem(cfg)` — raw plane(s) → the `(B, C, W, W)` feature plane the
  shared ViT tokenizer eats. This is where multi-resolution band groups
  get their native-GSD conv stems (see `S2NativeStem`). Dense single-plane
  sources use `nn.Identity()`.
- `build_head(cfg, in_dim)` — decoder fine tokens → pixel output(s).
  `DenseHead` (single plane, optional per-pixel L2 norm) covers embedding
  fields; `S2MultiResHead` decodes each native grid.
- `normalizer(cfg)` — per-tile stats side channel, or `None` if your data
  is already normalized (unit-norm embedding fields). Stats are compressed
  with `georvq.inference.stats_codec.StatsPacker` (~11 bytes/tile).
  A separate option, `model.stats_cond`, *transmits* the per-window mean
  and codes only the remainder; it pays when the mean is a large share of
  window energy and the field is not unit-norm. Charge it — an uncharged
  stats channel once reached 34% of a bitstream unnoticed.
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

`georvq/sources/ae.py` is the minimal reference (60 lines);
`georvq/sources/s2.py` is the full multi-resolution example.

## 3. A YAML preset

Copy the closest `configs/*.yaml`, set `data.source`, `data.store_dir`,
`model.window/bands/tiers`, and train:

```bash
georvq-train -c configs/my_source.yaml --gpu 0
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
| container | COG assets mirroring the source's 1024 px block quantum | packets; no container yet, none required |
| stats channel | none — the field is exactly unit-norm | the patch mean, 768 bits/window, charged |
| refinement | coarse tier, 2-bit uniform, **always on** (+1.5% rate) | fine tier, 1-bit rotated Lloyd, **opt-in** (+150% rate) |
| quality control | encode-time measured error ladder | not built — nobody has asked |

Every one of those differences is a measured response to the data, and
the refinement row is the cautionary one: the coarse-tier sidecar that
recovers ~98% of AlphaEarth's quantizer gap recovers **0.0%** on Tessera.
Locate your model's slack (compare full-depth reconstruction against the
un-quantized ceiling, per tier) before copying either.

## Current integration status

Single-plane sources (one `(bands, px, px)` array per chip) are fully
wired: streaming `ChipPool` (coverage + cycle modes), the training loop,
validation, figures, eval and inference all work out of the box.

Multi-group sources (dict batches, e.g. S2) have model-level support
(stems/heads/normalizer/quantization all tested) and a RAM-resident
`GroupChipLoader`; wiring dict batches through the streaming pool and the
standard train loop/validation is the designated next step — see
`tests/test_s2.py::test_s2_toy_training` for the working training pattern
in the meantime.
