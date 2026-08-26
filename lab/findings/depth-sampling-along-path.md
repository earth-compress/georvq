---
id: depth-sampling-along-path
date: 2026-08-19
commit: 6055dd0
sources: [tessera]
context: Matryoshka depth-tuple sampling during training; measured on the k256 geometry.
status: observed-once
keywords: [matryoshka, depth, sampling, path-tuples]
imported: true
---

Drawing the coarse depth uniformly and each finer tier conditionally makes
deep fine-tier levels vanishingly rare: measured P(active)=0.006 for tier-2
level 14, so those codebooks saw 0.6% of rows. Spreading probability
uniformly along the monotone truncation path (one step per codebook) raises
the same level to 0.197. Implemented as path_tuples in
georvq/model/codec.py.
