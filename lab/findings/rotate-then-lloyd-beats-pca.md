---
id: rotate-then-lloyd-beats-pca
date: 2026-08-19
commit: c964b9a
sources: [tessera]
context: TurboQuant-style analysis of post-RVQ residuals on the Tessera model (analysis/turboquant.py).
status: observed-once
keywords: [refinement, rotation, pca, lloyd, scalar-quantization]
imported: true
---

The post-RVQ residual is close to isotropic quantization noise: a random
rotation and a PCA rotation land within 1% of each other (relative MSE
0.1425 vs 0.1436 at ~1 bit/dim). With no structure left for PCA to exploit,
a Gaussian Lloyd-Max scalar quantizer is near-optimal, and the rotation
costs nothing in the bitstream because it is a seed. Random rotation even
beat PCA on raw latents (0.4062 vs 0.4249): PCA concentrates variance and a
fixed levels-per-dimension coder then quantizes the high-variance directions
too coarsely.
