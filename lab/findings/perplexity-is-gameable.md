---
id: perplexity-is-gameable
date: 2026-08-19
commit: c964b9a
sources: [tessera]
context: An isolated agent given only vectors and a fixed perplexity scorer; verified fp64 with independent re-scoring.
status: observed-once
keywords: [codebook, perplexity, metrics, utilization]
imported: true
---

Given only vectors and a fixed scorer, an agent reached 11.94-11.97 bits
of 12 with all 4096 codes used at every level, at 0.014% of energy
explained; decoded cosine 0.45 versus 0.98 for k-means. Utilization was
bought by quantizing directions where the data has no structure (8.24 bits
on the top-12 PCA directions versus 11.90 deep in the tail). Codebook
perplexity is not a health metric. Judge codebooks on rate-distortion.
