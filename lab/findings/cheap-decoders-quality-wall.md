---
id: cheap-decoders-quality-wall
date: 2026-08-20
commit: ae1f95b
sources: [ae]
context: Tier-subset decoders with low-res heads and bilinear upsampling, 0.76M and 4.25M params, against the 31M trunk.
status: observed-once
keywords: [decoder, speed, distillation, release-bar]
imported: true
experiment: lab/experiments/cheap-decoders-quality-wall
---

Cheap decoders run 17-28x faster but land 0.006-0.1 cosine below the 31M
trunk at matched rate. Speed is available; quality at speed is not, on this
architecture. Parked below the release bar.
