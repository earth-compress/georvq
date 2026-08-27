---
id: relative-gap-closure
date: 2026-08-10
commit: f9fa700
sources: [any]
context: Convention adopted after a depth-graded improvement was misread; numbers from Tessera training.
status: observed-once
keywords: [metrics, gap-closure, bounded-metric]
imported: true
---

For any bounded metric, use relative gap closure d(1-m)/(1-m), not
absolute deltas. It exposed a depth-graded improvement (coarse 20%, mid 9%,
full 0.9% per 1k steps) that absolute deltas flattened into noise.
