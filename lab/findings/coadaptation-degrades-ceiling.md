---
id: coadaptation-degrades-ceiling
date: 2026-07-06
commit: 988ca86
sources: [ae]
context: Cascade conditioning runs with a frozen location prior.
status: observed-once
keywords: [conditioning, cascade, encoder-freeze, ceiling]
imported: true
---

Co-adapting an encoder against a frozen location prior degrades it late in
training: the pre-quantization ceiling falls. Freezing the encoder while the
conditioning trains avoids this. The ceiling, not the codebook gap, is the
right diagnostic.
