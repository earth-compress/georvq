---
id: vq-state-stays-fp32
date: 2026-07-06
commit: 988ca86
sources: [any]
context: Observed during bf16 training of large-codebook RVQ models.
status: observed-once
keywords: [vq, fp32, bf16, ema, numerics]
imported: true
---

Under bf16 autocast, VQ residuals, EMA buffers and codebooks must stay
fp32 or codebooks degrade. Assignment scores may run bf16; that only flips
near-tie assignments.
