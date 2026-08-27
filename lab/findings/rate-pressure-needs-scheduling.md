---
id: rate-pressure-needs-scheduling
date: 2026-07-06
commit: 988ca86
sources: [ae]
context: Rate-aware VQ training in the AlphaEarth research cycle.
status: observed-once
keywords: [rate, entropy, codebook, schedule]
imported: true
---

A gentle rate weight recovers a few percent of rate for free, then
plateaus once the coarse codebook starves. Scheduling the rate pressure
works; increasing it does not.
