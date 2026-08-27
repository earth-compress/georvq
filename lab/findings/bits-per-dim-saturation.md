---
id: bits-per-dim-saturation
date: 2026-08-10
commit: c420e9c
sources: [any]
context: Single-pixel autoencoder rate-distortion sweeps (analysis/pixel_rd.py), dims 16/32/64/128; the source measured was not recorded at import.
status: observed-once
keywords: [latent, bits-per-dim, saturation, rate]
imported: true
---

A learned bottleneck saturates at ~4 bits per latent dimension: beyond
that, extra rate on the same dimension count buys nothing. Confirmed
across dims 16, 32, 64 and 128. Imported from docs/SOURCE_ONBOARDING.md,
which did not record which source the sweep ran on.
