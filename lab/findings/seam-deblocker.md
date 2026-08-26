---
id: seam-deblocker
date: 2026-07-29
commit: 6e533f0
sources: [ae]
context: Windowed decoding of AE tiles; the shipped deblock decoder (georvq/ae/deblock.py).
status: observed-once
keywords: [seams, deblocking, windows, decoder]
imported: true
---

Windowed codecs produce seam artifacts between neighbouring windows, with
inter-window error well above intra-window. A small post-decode deblocker
conditioned on the boundary removes most of it at zero rate cost.
