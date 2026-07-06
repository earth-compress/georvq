# georvq

Pyramidal residual-VQ compression for geospatial datasets.

One model, one bitstream, a whole rate–distortion front: georvq trains
hierarchical (coarse→fine) residual-VQ codecs over gridded geospatial data —
embedding fields (e.g. AlphaEarth), multispectral imagery (e.g. Sentinel-2),
or any source you write an adapter for — with:

- **Pyramid of tiers**: each tier codes a grid of cells at increasing spatial
  resolution; finer tiers code the residual detail on top of coarser ones.
- **Matryoshka variable rate**: training samples monotone per-tier depth
  tuples, so a single checkpoint decodes at any point on its R-D front.
- **Learned entropy model**: an optional context model makes codes cheaper
  to store and enables rate-aware training; archives are written with a real
  rANS coder and byte-exact verification.
- **Stats side-channel**: per-tile normalization stats compressed separately
  (adaptive PCA), for sources that need it.
- **Location conditioning** (optional): SatCLIP-based predict-and-subtract
  and codebook subsetting.
- **Staged training**: σ-conditioned continuous pretraining, warm-start with
  shape reconciliation, freeze schedules.

Everything is driven by validated pydantic configs; every checkpoint embeds
the full resolved config and can be rebuilt with `georvq.from_checkpoint()`.

## Install

```bash
pip install -e .            # core
pip install -e .[loc,archive,eval,viz,dev]   # everything
```

## Quick start

```bash
# Train the standard 3-tier pyramid on an AlphaEarth chip store:
georvq-train -c configs/pyr64.yaml data.store_dir=/data/AE-samples/chips_1024 log.run_tag=my_run

# Override anything with key=value dot paths:
georvq-train -c configs/pyr64.yaml rate.weight=0.01 train.steps=50000
```

See `docs/ADDING_A_SOURCE.md` for how to plug in a new data source
(a store builder + a `Source` adapter + a YAML preset), `docs/BITSTREAM.md`
for the packet/archive format, and `docs/LINEAGE.md` for how the checked-in
configs map to the original research runs.

## Status

Ported from the `ring-space` research repo (AlphaEarth compression cycle,
June 2026). This README is expanded as the port progresses.
