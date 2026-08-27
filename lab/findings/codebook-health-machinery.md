---
id: codebook-health-machinery
date: 2026-07-06
commit: 988ca86
sources: [ae]
context: Large codebooks (16k to 65k entries) in the AlphaEarth research cycle.
status: observed-once
keywords: [codebook, kmeans, reset, ema, collapse]
imported: true
---

Large codebooks stay alive only with the full set of health measures:
k-means or data init, an EMA decay ramp, dead-code resets with
traffic-scaled thresholds, and a commit loss. Without them, deep RVQ levels
collapse.
