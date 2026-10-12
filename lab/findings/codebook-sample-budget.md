---
id: codebook-sample-budget
date: 2026-08-19
commit: 6055dd0
sources: [tessera]
context: Tessera runs; the crossover constant comes from comparing K=16384x8 against K=256x20 on that source.
status: observed-once
keywords: [codebook, sizing, K, sample-budget, reset]
imported: true
related:
  - clashes: bigger-k-beats-more-levels
  - supports: batch-past-the-knee-buys-codebook-health
---

A codebook needs roughly 450 vectors per codeword inside its EMA window,
or fitting it is worse than not fitting it:

    vectors_per_code = batch x cells_in_tier x ema_window / K >= ~450

K=16384 at 8 levels sits far below this (18 to 54 vectors per code) and lost
90% of its codebook. K=256 at 20 levels gives thousands of vectors per code;
dead-code resets fell from 6,500-8,200 per event to 1-5, and the codebook
download is 12x smaller at equal rate-distortion. Because cell count rises
with tier, K should grade upward with tier rather than being fixed globally.
