---
id: flat-tail-means-decoder-floor
date: 2026-08-10
commit: f9fa700
sources: [any]
context: Low-rate ends of measured rate-distortion curves.
status: observed-once
keywords: [rate-distortion, decoder, entropy]
imported: true
---

A flat low-rate tail in a rate-distortion curve usually means a decoder
floor, not an entropy gap. Fix decode capacity before blaming the coder.
