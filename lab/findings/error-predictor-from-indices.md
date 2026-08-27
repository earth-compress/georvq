---
id: error-predictor-from-indices
date: 2026-07-28
commit: 3d2c9f9
sources: [ae]
context: Val split of ae_pyr64_divfull; parked 2026-07-28 with feasibility evidence.
status: observed-once
keywords: [error-prediction, indices, deferred]
imported: true
---

The 24 per-(tier, level) RVQ increment norms, pure bitstream features the
decoder already sums, predict log window-mean error at r=0.94 with a linear
probe, better than the encode-side coarse-residual norm (0.898). 42% of
predicted worst-5% windows are truly in the worst 5%. Which pixels are
likely bad, and how much deeper levels would help, are answerable from
indices alone. Parked, not shipped.
