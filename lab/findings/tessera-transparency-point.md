---
id: tessera-transparency-point
date: 2026-08-10
commit: f9fa700
sources: [tessera]
context: Same benchmark suite; all five benchmarks are coarse thematic point labels.
status: observed-once
keywords: [downstream, transparency, rate, benchmarks]
imported: true
related:
  - supports: probe-noise-floors
---

Tessera is downstream-transparent above ~1 b/px, at cos ~0.93, on
point-thematic tasks; AlphaEarth needs cos ~0.99 for the same verdict.
Below ~0.5 b/px Tessera degrades about twice as fast (LUCAS -0.108 vs
-0.052), established on LUCAS only. Confound: the suite has no
fine-spatial or temporal task, so the high-rate end of the ladder is
unmeasured, not shown to be wasted.
