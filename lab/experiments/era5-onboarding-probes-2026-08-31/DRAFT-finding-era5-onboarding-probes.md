---
id: era5-onboarding-probes
date: 2026-08-31
commit: 9f28f94
sources: [era5]
context: >
  First onboarding measurements for the ERA5 cube store. data_probe and
  pyramid_sizing (W=64, grids 1/4/16, 40 chips) on the completed val
  split (320 cubes, 2019+2024); one chip is one hourly slice of one
  variable with the 37 pressure levels as bands, time flattened into the
  chip draw. Temporal autocorrelation from 12 val cubes per variable.
  Standardization moments from 256 slices of the train split (committed
  windows only, 55/200 at measurement time).
status: observed-once
keywords: [era5, onboarding, autocorrelation, pyramid-sizing, lossless-floor]
experiment:
related:
---

Spatial structure. All five pressure-level variables hold
autocorrelation at or above 0.5 out to the 32 px lag (t, z, u above
0.5 at 32 px; v and q at 0.50). ERA5 at 0.25 deg is the smoothest
source measured so far: energy below 4 px is 0.15% (t), 0.08% (z),
0.64% (u), 0.97% (v), 1.30% (q) of per-level-standardized window
energy, against 46% for Tessera and 1.7% for DEM.

Pyramid decomposition, per-level standardized, W=64 grids 1/4/16
(energy % tier0/1/2/left, then d95 per tier):

  t  88.1 / 10.4 / 1.4 / 0.15    d95  5 / 12 / 21
  z  86.1 / 12.8 / 1.0 / 0.08    d95  3 /  3 /  3
  u  68.0 / 26.5 / 4.8 / 0.64    d95  4 /  6 / 12
  v  61.5 / 30.7 / 6.8 / 0.97    d95  6 / 11 / 20
  q  74.0 / 18.1 / 6.6 / 1.30    d95  7 / 15 / 21

The d95 floors are far below 37, so the levels of one variable are
strongly redundant at every scale, most extremely for z (rank ~3 at
every tier).

Lossless floor, unit = the stored int16 (per-cube affine, matching
native GRIB 16-bit packing): left-neighbour prediction gives
t 361, z 298, u 375, v 382, q 377 b/px over 37 levels (roughly
8-10 b/px per level); the left predictor beats (left+up)/2 on all
five, saving 19-33% against raw entropy. Order-0 per band, so the
floor is conservative, and it is a floor for reproducing the store's
per-cube quantization, not the underlying field.

Magnitude spread of the 37-level profile (p90/p10, physical units):
t 1.07x, z 1.04x, u 3.57x, v 4.18x, q 8.10x. For t and z an equal
per-level weighting after standardization is uncontroversial; u, v and
especially q carry real per-profile magnitude variation, which the
cycle-1 standardized-MSE objective flattens by design.

Temporal decorrelation of within-cube anomalies (per-pixel 96 h time
mean removed): below 0.5 by 12 h for t, u, v, q and msl, by 24 h for
z; t2m rebounds to 0.62 at lag 24 h (diurnal cycle). A 6 h index
stride keeps neighbouring kept slices at correlation ~0.58-0.76.

Caveats: probes ran on the val years (2019+2024); the standardization
moments came from the train years. 40 chips (160 windows) per variable;
d95 numbers moved by <=1 between a 6-chip trial and the 40-chip run.
