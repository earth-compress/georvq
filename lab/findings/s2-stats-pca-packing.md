---
id: s2-stats-pca-packing
date: 2026-08-21
commit: 6c769f0
sources: [s2]
context: 26 per-band mean/std floats per S2 window, PCA basis fitted on
  training windows and measured on held-out windows
  (`inference/stats_codec.py`), tolerance 0.01.
status: observed-once
keywords: [stats, pca, rate, side-channel, quantization]
related:
  - supports: fixed-side-channels-dominate-coarse-rungs
---

PCA with tolerance-coded uint8 components packs the S2 stats vector to
9.73 bytes per window, against 52 bytes for 26 x f16. Five components
carry 99% of the variance in the stats vector.

The basis appears not to be delicate: this reproduces the roughly 11
bytes per tile that the ring-space S2 betas shipped, on a basis refitted
for a different window size.

Measured for one window size on one source. Whether the ratio holds for
sources with fewer bands, or with bands whose statistics are less
correlated, is untested.
