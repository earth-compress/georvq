---
id: refinement-budget-is-the-ceiling-gap
date: 2026-08-19
commit: c964b9a
sources: [ae, tessera]
context: Scalar-quantized residual sidecars (inference/refine.py) on the AE and Tessera release models.
status: observed-once
keywords: [refinement, sidecar, ceiling, saturation]
imported: true
---

A residual refinement can recover at most the distance between full-depth
reconstruction and the same model decoding un-quantized latents. If that gap
is small the tier is saturation-limited, not rate-limited, and there is
nothing to buy. This is why the AlphaEarth mid and fine sidecars were
worthless: those codebooks sit at their ceiling. Measure the ceiling first.
