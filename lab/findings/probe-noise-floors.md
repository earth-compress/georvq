---
id: probe-noise-floors
date: 2026-08-10
commit: f9fa700
sources: [ae, tessera]
context: Re-probing every rate point under repeated split seeds.
status: observed-once
keywords: [downstream, noise-floor, lcmap, lucas]
imported: true
---

Probe noise floors: +/-0.017 on LUCAS and +/-0.023 on LCMAP under
repeated split seeds. LCMAP cannot resolve anything below ~0.05, and its
readings of compression beating the original are artefacts of that noise.
