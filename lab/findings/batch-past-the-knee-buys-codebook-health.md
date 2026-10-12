---
id: batch-past-the-knee-buys-codebook-health
date: 2026-08-21
commit: 6c769f0
sources: [s2]
context: S2 pyramid, 96 px windows, A100 80GB with gradient
  checkpointing. Throughput swept over batch size, then repeated after
  doubling RVQ depth per tier and widening the latents by 1.5x.
status: observed-once
keywords: [batch, throughput, codebook, sample-budget, scaling]
related:
  - supports: codebook-sample-budget
---

Windows per second saturates from batch 768 upward: 445 / 628 / 631 /
634 / 570 at batches of 384 / 768 / 1024 / 1536 / 2048. Past that knee a
larger batch costs proportional wall-clock per step and returns better
gradients and more vectors per codeword, not speed.

Doubling RVQ depth and widening the latents by 1.5x moved neither the
knee nor the plateau by more than a few percent. With `skip_masked`, the
levels that the sampled depth mask switches off cost nothing, and the
tokenizer and decoder trunks dominate the step regardless.

This gives an order for the two choices: pick the batch from the
sample-budget rule, then pick the step count from the wall-clock
available.

Measured on one source and one GPU. The knee is a property of that
combination, not a general figure.
