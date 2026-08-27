---
id: cosine-not-transferable
date: 2026-08-10
commit: f9fa700
sources: [ae, tessera]
context: AE-point-matched benchmark suite (LUCAS, LCMAP, ASTER GED); tile-blocked splits.
status: observed-once
keywords: [downstream, cosine, quality-target, lucas]
imported: true
related:
  - supports: tessera-transparency-point
---

At matched reconstruction cosine the Tessera codec costs less downstream
accuracy than the AlphaEarth codec. LUCAS land cover: AE at cos 0.9532
loses 0.043 balanced accuracy; Tessera at cos 0.9475 loses 0.005. Tessera
carries ~2.5x more independent variation per pixel (per-tier PCA d95 of
51/91/71 against tier dims 512/512/256, versus AlphaEarth's 19/34/28), much
of it weakly correlated with land cover. A quality target must come from
the task, never from another source's cosine.
