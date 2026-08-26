---
id: clip-point-depends-on-bit-depth
date: 2026-08-19
commit: c964b9a
sources: [tessera]
context: Uniform scalar quantizers in residual refinements.
status: observed-once
keywords: [refinement, scalar-quantization, clip, lloyd]
imported: true
---

A uniform quantizer's clip point must be retuned with its bit depth.
Placing the outermost reconstruction level at sigma x std works at 2 bits
with sigma=2, but at 1 bit it puts the two levels at +/-2 std where the
optimum is +/-0.8, and the refinement adds error instead of removing it.
One bit per dimension belongs to the Lloyd family.
