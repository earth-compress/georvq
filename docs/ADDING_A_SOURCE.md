# Adding a data source

Everything dataset-specific lives in three pluggable pieces; the pyramid,
RVQ, rate model, decoder trunk, training loop, evals and inference are
shared. To onboard a new source you write:

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
- `recon_stats(out, x, valid) -> (loss, metrics)` — the training loss and
  quality metrics in model space.

`georvq/sources/ae.py` is the minimal reference (60 lines);
`georvq/sources/s2.py` is the full multi-resolution example.

## 3. A YAML preset

Copy the closest `configs/*.yaml`, set `data.source`, `data.store_dir`,
`model.window/bands/tiers`, and train:

```bash
georvq-train -c configs/my_source.yaml --gpu 0
```

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
