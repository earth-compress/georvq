---
id: blocked-halo-attention
date: 2026-07-29
commit: 6e533f0
sources: [any]
context: Kernel experiment for narrow local attention; the stream it served was abandoned.
status: observed-once
keywords: [attention, kernel, performance, deferred]
imported: true
---

Computing a fine tier's neighbourhood attention in blocks with a halo,
rather than gathering per query, measured 7x faster. The narrow-suffix
stream it was built for was abandoned with its architectures; the kernel
idea stands on its own. Parked.
