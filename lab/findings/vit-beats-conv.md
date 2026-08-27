---
id: vit-beats-conv
date: 2026-07-06
commit: 988ca86
sources: [ae]
context: AlphaEarth research cycle; date and commit are when this was recorded, the runs predate them.
status: observed-once
keywords: [architecture, vit, conv, tokenizer, decoder]
imported: true
---

A ViT tokenizer with an attention decoder beat convolutional trunks and
FPN decoders decisively on dense embedding fields. Convolutional patch-embed
stems were kept as the per-source plug point; the part that lost was
convolutional feature mixing.
