---
id: match-budget-before-crossovers
date: 2026-08-10
commit: f9fa700
sources: [ae]
context: AE-versus-linear-baseline comparisons during the AlphaEarth cycle.
status: observed-once
keywords: [metrics, crossover, training-budget, baselines]
imported: true
---

The AE-versus-linear crossover moved from 200 to 300 to 500 to 600 bits
as data and steps were added, and every move favoured whichever arm had
been under-resourced. Match training budgets before believing a crossover.
