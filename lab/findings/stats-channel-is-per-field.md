---
id: stats-channel-is-per-field
date: 2026-08-19
commit: 9bc7af5
sources: [ae, tessera, s2]
context: Comparing per-tile normalization across multispectral imagery, AlphaEarth and Tessera; Tessera numbers from the statscond runs.
status: observed-once
keywords: [stats, mean, std, side-channel, normalization]
imported: true
experiment: lab/experiments/stats-channel-is-per-field
related:
  - supports: fixed-side-channels-dominate-coarse-rungs
  - supports: s2-stats-pca-packing
  - clashes: ae-transmitted-mean
---

Whether a stats side channel pays is a property of the field, not the
modality. Per-tile mean/std normalization was very effective on
multispectral imagery. AlphaEarth's unit-norm field needs none. Tessera
(magnitude spread 1.57x) wants the mean transmitted but not the std: the
mean is 85.5% of window energy and buys +0.0069 cosine for 0.188 b/px,
while forcing the std hurts (0.9700 -> 0.9678) because the decoder is
already smoother than the truth, so rescaling amplifies its error.
