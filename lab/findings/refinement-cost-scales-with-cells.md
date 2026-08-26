---
id: refinement-cost-scales-with-cells
date: 2026-08-19
commit: c964b9a
sources: [ae, tessera]
context: Cost accounting of residual sidecars on the 1/4/16 pyramid.
status: observed-once
keywords: [refinement, sidecar, rate, cells]
imported: true
---

A refinement costs cells x dim x bits, which on a 1/4/16 grid pyramid is
1x, 16x, 256x cells per tier. AlphaEarth's coarse sidecar is +1.5% of rate
and can be left on by default; Tessera's fine refinement more than doubles
the bitstream and has to be an opt-in.
