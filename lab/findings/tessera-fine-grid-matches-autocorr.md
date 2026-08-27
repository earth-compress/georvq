---
id: tessera-fine-grid-matches-autocorr
date: 2026-08-10
commit: c420e9c
sources: [tessera]
context: Autocorrelation-versus-lag measurement (analysis/data_probe.py) against the deployed Tessera geometry.
status: observed-once
keywords: [autocorrelation, grids, window, tiers]
imported: true
---

Tessera's autocorrelation falls below 0.5 at a lag of 4 px, consistent
with the deployed fine tier's 4 px cells (16x16 grid on a 64 px window).
The first case where a grid choice was retrospectively justified by a
measurement rather than inherited.
