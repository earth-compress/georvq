---
id: location-explains-coarse-variance
date: 2026-07-06
commit: 988ca86
sources: [ae]
context: SatCLIP conditioning experiments on the flagship AlphaEarth dataset.
status: observed-once
keywords: [location, satclip, conditioning, coarse]
imported: true
---

Location explains about 47% of coarse-embedding variance on the flagship
dataset. A pretrained location encoder beat raw spherical harmonics in the
deployment (interpolation) regime. The marginal codec value of location
conditioning concentrates on the coarse tier.
