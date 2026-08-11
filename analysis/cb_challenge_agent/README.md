# Codebook utilization challenge

## Data
`vectors_train.npy` — (786432, 256) float32. Real vectors from a
production system. This is your training data; use it however you like.

A **private holdout set** of vectors from the same source (disjoint
groups, so no near-duplicates across the split) is used for scoring. You
cannot see it.

## Task
Produce **3 codebooks**, each of shape **(4096, 256)**, that act as a
3-level residual quantizer on these vectors:

    level 0 quantizes v        -> residual r0 = v - C0[a0]
    level 1 quantizes r0       -> residual r1 = r0 - C1[a1]
    level 2 quantizes r1       -> residual r2 = r1 - C2[a2]

Assignment at every level is nearest-neighbour in Euclidean distance.

## Objective
**Maximise the sum of per-level perplexity** (perplexity = 2^H, where H
is the entropy in bits of the index distribution at that level).

- Maximum possible: 3 x 4096 = 12288
- **Stop target**: mean per-level entropy >= log2(4096) - 1 = **11.0 bits**

The residual energy is also reported. It is NOT the objective, but it is
recorded for every submission.

## Rules
- **K = 4096 and DIM = 256 are fixed.** You may not change them.
- Write all your own code from scratch, in this directory. Do not import
  or read from any other codebase on this machine, and do not go looking
  for one — the point of the exercise is an independent attempt.
- Do not read, open, or search for the holdout file, and do not modify
  `challenge_eval.py`. It is hash-checked after the run.
- Any method is allowed: one-shot constructions, iterative fitting,
  optimisation, randomised methods, or anything else you invent.

## Scoring API (fixed — do not edit)
```python
from challenge_eval import spec, score_local, score_holdout

spec()                              # problem parameters
score_local([C0, C1, C2], my_vecs)  # free, unlimited, your own vectors
score_holdout([C0, C1, C2], note="what I tried")   # SUBMISSION
```
`score_holdout` is limited to **10 submissions total**. Use them
deliberately: iterate with `score_local` on your own held-out split of
the training data, and submit only when you expect an improvement.

Report at the end: what you tried, what the holdout scores were, and
what you concluded about why the achievable utilization is what it is.
