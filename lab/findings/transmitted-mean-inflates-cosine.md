---
id: transmitted-mean-inflates-cosine
date: 2026-08-10
commit: f9fa700
sources: [tessera]
context: Tessera stats-conditioned models; coarse-only decode readings.
status: observed-once
keywords: [metrics, ccos, cosine, stats, mean]
imported: true
---

A transmitted mean inflates plain cosine. A coarse-only decode read
0.9154 plain cosine while contributing no spatial structure at all, and a
block-mean reconstruction scores 0.92. Track centred cosine (ccos) on any
model with a stats channel.
