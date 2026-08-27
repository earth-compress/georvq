---
id: bf16-decoder-only
date: 2026-07-29
commit: 6e533f0
sources: [ae]
context: Index stability tests across runs and hardware on the AE release model.
status: observed-once
keywords: [bf16, fp32, indices, bitstream-stability]
imported: true
---

bf16 encoder latents flip ~24% of near-tie RVQ indices. That is
quality-neutral but makes bitstreams unstable across runs and hardware.
fp32 encoder with bf16 decoder keeps indices bit-stable at ~95% of the
speed win.
