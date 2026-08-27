---
id: mse-books-are-decoder-optimal
date: 2026-08-19
commit: c964b9a
sources: [ae]
context: AE encoder and decoder frozen; codewords optimised against the decoded loss.
status: observed-once
keywords: [codebook, decoded-loss, optimization]
imported: true
---

Freezing AE's encoder and decoder and optimising only the codewords
against the decoded loss gained +0.00007 cosine (0.99585 to 0.99592,
monotone, held-out). A real but negligible degree of freedom; not worth a
pipeline stage.
