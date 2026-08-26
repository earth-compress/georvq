---
id: tier-redundancy-collapse
date: 2026-08-19
commit: c964b9a
sources: [tessera]
context: The 4-tier uf model (grids 1/4/16/32 over 64 px) against 3-tier models s1_4x and s1_2x.
status: observed-once
keywords: [tiers, collapse, variance, rate-waste]
imported: true
---

In the 4-tier uf model the fine tier carries only 0.2% of its variance
within a window; its 256 cells are near-copies of one window-level vector.
The same tier is healthy in every 3-tier model (96.8% in s1_4x, 97.9% in
s1_2x); it collapses only once the 2 px ultrafine tier is added and takes
over its job. The collapsed tier costs ~62% of the rate budget and is
invisible in the ceiling, because a duplicated tier wastes rate, not
quality. Check within-window latent variance per tier before trusting any
tier.
