---
id: refinement-tier-differs-per-model
date: 2026-08-19
commit: c964b9a
sources: [tessera, ae]
context: Tessera release checkpoint, 768 held-out windows, base rate 10.85 b/px; share of the centred-cosine ceiling gap recovered.
status: observed-once
keywords: [refinement, sidecar, tier, slack]
imported: true
related:
  - supports: refinement-budget-is-the-ceiling-gap
  - supports: ae-outlier-tail-is-coarse
---

Which tier holds recoverable slack differs by model, and the wrong choice
returns exactly zero. On the Tessera release checkpoint:

| refined tier | rate | gap closed |
|---|---:|---:|
| 0 (coarse), 1-2 bits | +0.13-0.25 b/px | 0.0% |
| 1 (mid), 1-2 bits | +2.0-4.0 b/px | 1.9% |
| 2 (fine), 1 bit | +16 b/px | 56.2% |
| 2 (fine), 2 bits | +32 b/px | 81.2% |

AlphaEarth refines tier 0 and recovers ~98% of its gap there; the same
policy on Tessera recovers nothing. Tessera's coarse book is healthy and its
fine tier is still rate-limited (residual still decaying at level 20).
