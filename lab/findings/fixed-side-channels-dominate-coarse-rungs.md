---
id: fixed-side-channels-dominate-coarse-rungs
date: 2026-08-21
commit: 1ffbc52
sources: [s2]
context: S2 pyramid, 96 px windows, Major-TOM L1C read live. Rate
  composition of the per-window normalizer channel (13 band means and 13
  stds) measured across the depth ladder.
status: observed-once
keywords: [rate, side-channel, normalizer, stats, coarse, ladder]
related:
  - supports: cells-dominate-rate
  - supports: stats-channel-is-per-field
  - supports: ae-transmitted-mean
---

A side channel costs the same number of bits at every depth, while the
coded payload shrinks toward zero as depth is removed, so the channel's
share of the rate is a strong function of where you sit on the ladder.
On the S2 pyramid the per-window normalizer statistics are 1.6% of the
full-depth rate and 3.31x the coarse-only rung.

The consequence is that a rate curve which omits a side channel looks
correct where it is easiest to check and is wrong by more than a factor
of three where the codec is most attractive. Rate comparisons between
codecs are only meaningful at the coarse end if both sides account for
their side channels.

The ratio depends on the size of the channel and on how small the coarse
payload is, so 3.31x is specific to 26 statistics on a 96 px S2 window.
The direction of the effect follows from the arithmetic and is not
specific to S2.
