---
id: geographic-splits-only
date: 2026-07-06
commit: 988ca86
sources: [any]
context: Evaluation protocol adopted after random splits leaked.
status: observed-once
keywords: [evaluation, split, leakage, geography]
imported: true
---

Split train/test geographically (hashed lat/lon cells), never randomly.
Chips from one region leak into both sides of a random split.
