---
id: rate-ladder-asymmetry
date: 2026-07-29
commit: 6e533f0
sources: [ae]
context: Measured Pareto fronts on the AE release geometry.
status: observed-once
keywords: [rate, ladder, pareto, tiers]
imported: true
related:
  - supports: cells-dominate-rate
---

The rate ladder is extremely asymmetric (coarse 11 / mid 208 / fine 3,584
bits per level). The measured Pareto front always saturates coarse, then
mid, then adds fine levels. Hand-picked balanced tuples like 6-4-4 are
strictly dominated.
