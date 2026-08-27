---
id: warmstart-needs-maturing
date: 2026-07-06
commit: 988ca86
sources: [ae]
context: Warm-starting runs from trained codebooks.
status: observed-once
keywords: [codebook, warmstart, reset, ema]
imported: true
---

Warm-started codebooks must be matured: resets off and slow EMA.
Otherwise dead-code resets destroy the loaded structure.
