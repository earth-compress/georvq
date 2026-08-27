---
id: patch-conditioning-headroom
date: 2026-08-10
commit: c420e9c
sources: [ae, tessera, s2]
context: analysis/patch_intrinsic_dim.py, pooled versus local complexity per source.
status: observed-once
keywords: [conditioning, intrinsic-dimension, patches]
imported: true
---

The ratio of pooled to within-patch complexity bounds what per-patch
conditioning can ever buy: 0.92x on Sentinel-2 (nothing to gain), 1.44x
on AlphaEarth, 2.90x on Tessera.
