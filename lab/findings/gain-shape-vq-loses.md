---
id: gain-shape-vq-loses
date: 2026-08-19
commit: c964b9a
sources: [ae]
context: Implemented and measured at matched rate; since removed from the codebase.
status: observed-once
keywords: [codebook, gain-shape, vq]
imported: true
---

Gain-shape VQ lost at matched rate: plain VQ 0.2251 versus 0.2537
(K=4096 plus 2 gain bits) and 0.3087 (K=1024 plus 4). The premise that
books waste capacity on magnitude does not hold: the shape-codebook
capacity the split costs exceeds the magnitude precision it buys.
