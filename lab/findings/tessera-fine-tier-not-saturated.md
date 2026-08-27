---
id: tessera-fine-tier-not-saturated
date: 2026-08-19
commit: 9bc7af5
sources: [tessera]
context: Per-level residual decay on the Tessera release geometry (K=256, 20 levels).
status: observed-once
keywords: [depth, levels, saturation, rate]
imported: true
related:
  - supports: refinement-tier-differs-per-model
---

Tessera's fine tier at level 20 still contributes 17% of what level 1
contributes, so the tier is not saturated and 20 levels are too few.
Each further level costs 0.5 b/px. Depth N should follow per-level
residual decay: levels keep paying while their contribution is a
material fraction of level 1's.
