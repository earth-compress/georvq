---
id: bigger-k-beats-more-levels
date: 2026-08-19
commit: c964b9a
sources: [ae]
context: Matched rate (~112 bits/cell) on a healthy AE tier.
status: observed-once
keywords: [codebook, K, levels, rate-distortion]
imported: true
related:
  - clashes: codebook-sample-budget
---

At matched rate on a healthy tier, bigger K beats more levels: K=16384x8
gives 0.2252, 4096x9 gives 0.2354, 1024x11 gives 0.2411, 256x14 gives
0.2539 (residual, lower better). The opposite conclusion had been drawn
from a collapsed tier and was an artefact. This is a statement about
rate-distortion at fixed rate on a well-sampled tier; the sample-budget
finding governs whether a K that large can be trained at all on a given
source.
