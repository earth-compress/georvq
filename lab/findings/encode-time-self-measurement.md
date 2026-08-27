---
id: encode-time-self-measurement
date: 2026-08-20
commit: ae1f95b
sources: [ae]
context: The AE release encoder's fused rate-ladder pass.
status: observed-once
keywords: [encode, error-stats, quality-control]
imported: true
---

Decoding the rate ladder during encode (fused pass: shared encode plus
cumsums, ~2x the base encode cost) yields per-blob and per-window true
error statistics. These make quality-controlled delivery possible with no
predictor network.
