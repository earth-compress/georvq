---
id: decoder-is-the-bottleneck
date: 2026-07-06
commit: 988ca86
sources: [ae]
context: AlphaEarth research cycle; recorded 2026-07-06, runs earlier.
status: observed-once
keywords: [decoder, bitstream, capacity]
imported: true
---

With the encoder and bitstream frozen, swapping in a larger transformer
decoder gained +1 to +3 dB at identical rate. The reconstruction quality of a
stored bitstream was limited by the decoder, not the latent. This motivated
the decoder-independent bitstream contract (docs/BITSTREAM.md).
