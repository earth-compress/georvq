---
id: matryoshka-min-depths
date: 2026-07-06
commit: 988ca86
sources: [ae]
context: AlphaEarth research cycle; recorded 2026-07-06, runs earlier.
status: observed-once
keywords: [matryoshka, depth, training, rate]
imported: true
---

Training with per-tier minimum depths (for example 6-0-0) concentrates
model capacity on the operating points that get used. Depth tuples that were
never sampled during training give misleading evaluation numbers.
