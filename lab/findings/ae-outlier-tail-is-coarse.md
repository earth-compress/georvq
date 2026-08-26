---
id: ae-outlier-tail-is-coarse
date: 2026-07-29
commit: 6e533f0
sources: [ae]
context: 320M-pixel error analysis of full-depth decodes on the AE release model.
status: observed-once
keywords: [outliers, coarse, sidecar, tail]
imported: true
related:
  - supports: refinement-budget-is-the-ceiling-gap
---

On AlphaEarth the worst-0.1% pixels under full-depth decode are
coarse-codebook misses (rare land covers; window-scale bias), correlated
0.74 with the coarse residual norm. A 2-bit scalar-quantized residual
sidecar (66 B/window, per-window scale) recovers ~98% of the gap including
the tail, for +1.5% rate. Mid and fine sidecars on that model are
worthless-to-harmful: those codebooks sit at their ceiling.
