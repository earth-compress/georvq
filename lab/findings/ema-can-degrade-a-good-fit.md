---
id: ema-can-degrade-a-good-fit
date: 2026-08-19
commit: c964b9a
sources: [tessera, ae]
context: Frozen encoder, static latents, no resets firing; end-to-end comparisons on Tessera and AE.
status: observed-once
keywords: [codebook, ema, kmeans, drift]
imported: true
---

With everything frozen, ema_decay 0.99 took residual 0.033 to 0.047 by
2400 steps, damage concentrated in deep levels (cascade drift); 0.9995 was
flat. On Tessera a fresh 8-iteration k-means beat 60k EMA steps end-to-end
(0.9766 vs 0.9744). On AE the same test is a tie (0.9963 vs 0.9964); AE's
latents are much lower-rank (d95 19/34/28 vs 51/91/71), so its books are
better sampled and stable. Fit-once-and-freeze is never worse.
