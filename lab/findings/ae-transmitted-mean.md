---
id: ae-transmitted-mean
date: 2026-10-13
commit: 4b57fc4
sources: [ae]
context: The AE v2 release checkpoint (ownk_sc4_step80000, a stats_cond model with the mean transmitted) against the ae-beta-v1 release, scored on the training run's validation tokens with the fill-order ladder; mean-coder fits on 9.5M training-window means and 420k held-out validation means.
status: observed-once
keywords: [stats, mean, side-channel, stats_cond, alphaearth, mean-coder, rvq, pca]
related:
  - clashes: stats-channel-is-per-field
  - supports: fixed-side-channels-dominate-coarse-rungs
---

The AlphaEarth v2 release subtracts each 64 px window's mean before the
encoder and adds it back after the decoder, so the mean is sent with
every window at every rung. Scored at matched bits, with the mean charged
at 6 bits per band (384 bits per window), the v2 checkpoint has 7-36%
less error (1 - cosine) than the ae-beta-v1 release, which sends no mean,
above about 0.2 b/px, and more error below it:

| rung | b/px | v2 (sc4) | ae-beta-v1 |
|---|---|---|---|
| 2-0-0 | 0.099 | 0.9717 | 0.9810 |
| 8-1-0 | 0.166 | 0.9852 | 0.9855 |
| 8-5-0 | 0.369 | 0.9914 | 0.9895 |
| 8-8-0 | 0.521 | 0.9927 | 0.9905 |
| 8-8-8 | 7.52 | 0.9976 | 0.9963 |

With the exact mean, v2 reaches 0.99815 at full depth against a ceiling
of 0.99859. This comparison does not isolate the mean. The two models
also differ in codebook sizes (v2 uses its own K of 2048/8192/16384 per
tier), initialisation, depth sampling, int8 books and training length,
and part of the v2 run used two GPUs with live codebooks. What it shows
is that an AlphaEarth model built around a transmitted mean is better
than the mean-free release at the rates above 0.2 b/px, which is where
the release is used, whilst the fixed cost of the mean makes it worse
at the lowest rungs.

The model is sensitive to the accuracy of the mean, because it was
trained on exact means and passes any mean error straight through to
the output. It reaches its exact-mean quality only when the median
relative L2 error of the coded mean is near 0.006. With a 96-bit mean,
full-depth cosine falls to 0.95, and with no mean at all to 0.43. To
reach 0.006, a two-level RVQ (2 x 4096 codes) followed by a PCA coder
of the residual needs about 440 bits per window, against about 512 bits
for PCA alone; uniform per-band quantisation needs 512 bits (8 bits per
band) to reach 0.0085. Greedy, distortion-ranked split initialisation
of the RVQ books beat a balanced PCA tree by 1-2% at every level. A
SatCLIP-plus-year location prior explained R^2 0.59 of the mean on
held-out chips at zero bits, but subtracting it helped only PCA-only
coding and hurt the RVQ coder at low budgets, so the release does not
use it. The release's mean coder (RVQPCAMeanChannel, 440 bits, val
relative L2 0.0060) costs about 0.0001 cosine at every rung against the
exact mean.

The scoring and fitting scripts were not kept in the repository; the
outputs are in the maintainer's scratch directory
(scratch/ae-v2/health/sc4_fill_*.json, scratch/ae-v2/meanrvq/). The
date is approximate: the measurements were made on 2026-10-12 and
2026-10-13.
