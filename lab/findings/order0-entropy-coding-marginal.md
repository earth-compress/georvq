---
id: order0-entropy-coding-marginal
date: 2026-07-29
commit: 6e533f0
sources: [ae]
context: rANS archives of real AE tiles versus npz-deflated raw indices.
status: observed-once
keywords: [entropy-coding, rans, context-model, rate]
imported: true
---

Order-0 rANS gained ~0.6% over npz-deflated raw indices on real tiles
(the fine tier runs at ~95% of codebook entropy; deflate already strips the
uint16 padding to ~13.4 effective bits of 14) while dominating decode time.
The remaining gap is cross-tier context (~-22% measured by probes), which
needs the learned context model, not a better static coder.
