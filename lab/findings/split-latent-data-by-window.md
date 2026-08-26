---
id: split-latent-data-by-window
date: 2026-08-19
commit: c964b9a
sources: [any]
context: Codebook fitting datasets built from latent stores.
status: observed-once
keywords: [split, leakage, kmeans, latents]
imported: true
related:
  - supports: geographic-splits-only
---

Split latent datasets by window, never by cell. Adjacent cells are
near-duplicates (mean nearest-neighbour distance ~6% of vector norm), and
a random cell split makes k-means look ~500x better than it is.
