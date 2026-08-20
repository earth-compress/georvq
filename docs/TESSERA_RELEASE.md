# The Tessera release codec

The shippable codec for [Tessera](https://github.com/ucam-eo/geotessera)
embedding tiles. Everything documented here is **release policy for this
source** (`georvq/tessera/`), built on the source-agnostic machinery in
`georvq/inference/`. The AlphaEarth release (`docs/AE_RELEASE.md`) is the
other worked example, and the two differ in instructive places.

## Who is this for

| you are | you touch |
|---|---|
| **Tessera data user** — decode compressed windows | `georvq.tessera.load`, the *Format* and *Rate control* sections; never the encoder or training code |
| **codec researcher** — train or improve models | `georvq/{model,train,...}`, `docs/OVERVIEW.md`; this package is a worked example of a release layer |

## The release

- Model: `tessera_vq8_k256_statscond`, step 52,000 (cumulative 71,000 --
  the run completed),
  window 64, tiers `1:512:256:20 / 4:512:256:20 / 16:256:256:20`.
- Bundle: `/data/georvq-releases/tessera-beta-v1/bundle/` — `model.pt`,
  `mean_channel.npz`, `refine_table.npz`, `bundle.json` (carries
  `bitstream_id` and `decoder_id`).
- Status: **beta.** Cut from the completed run. The last 20,000 steps
  bought +0.0002 cosine and +0.0018 centred cosine over the mid-run
  checkpoint, so this is where the recipe lands rather than where it was
  going. A replacement checkpoint can still be dropped in without
  changing any of the policy below, and will change `bitstream_id`.

## Getting it

Weights are not in git.

- *On the group server*: `/data/georvq-releases/tessera-beta-v1/bundle`.
- *Elsewhere*: copy `tessera-beta-v1.tar.gz` (425 MB) and **verify it
  before loading**.

```bash
sha256sum -c <<< "090b9d628c0d9b5520810d8f5eb62718a5caf310373e70def43241d73b523dc6  tessera-beta-v1.tar.gz"
tar xzf tessera-beta-v1.tar.gz
pip install -e .                  # decoding needs no extras at all
```

The tarball is **data, not an executable**: four files, no code. The
decoder is the `georvq` package, which is what lets a better decoder
ship later against the same `bitstream_id` without re-encoding
anything.

    bundle/model.pt            weights + the embedded resolved config
    bundle/mean_channel.npz    the mean channel's per-band constants
    bundle/refine_table.npz    the fine-tier Lloyd table
    bundle/bundle.json         bitstream_id, decoder_id, tiers, refinement spec

The checksum is not ceremony. `model.pt` is a torch pickle and the
config rides inside it, so it is loaded with `weights_only=False` --
which means **loading it executes code**. Treat an unverified bundle
the way you would treat an unverified script.

## Weight precision

The reference bundle stores fp32 weights -- exactly what was trained and
validated. `scripts/build_release_tessera.py --fp16` produces an
otherwise identical bundle at **half the size**:

| | model.pt |
|---|---:|
| fp32 (default) | 469 MiB |
| `--fp16` | 235 MiB |

It costs nothing to decode. Rounding all 116M parameters to fp16 moved
full-depth cosine by **-8.8e-07** and centred cosine by -8.0e-07 over
1,024 held-out windows -- about 260x smaller than the difference between
two consecutive release checkpoints.

The reason it is free is worth being precise about, because it is easy
to claim more than is true. This changes the **stored** precision, not
the precision anything computes in: `load_state_dict` copies fp16
tensors into the fp32 model and upcasts, so the arithmetic afterwards is
bit-identical to an fp32 release, and index determinism across hardware
is whatever it already was.

**The two variants carry different `bitstream_id`s, and should.** The id
hashes the weights that actually ship, so rehashing a bundle reproduces
its stated id -- which is the whole point, since the tile backends
recompute it from the weights they loaded and refuse a mismatch. An
fp16 bundle's codebooks really do hold different values, so it really is
a different bitstream: the same indices decode to a very slightly
different field. Pick one variant per archive and stay on it.

Computing in fp16 would be a different proposal. A narrower accumulate
makes near-tie RVQ assignments hardware-dependent, and the encoder's
assignments *are* the bitstream -- rounding the encoder's weights to
fp16 already moves 1.8% of indices relative to fp32 (at no cost to
quality, since they are near-ties by definition). That is harmless when
one released file is the only encoder in existence, and a reproducibility
problem the moment two coexist, which is why fp32 stays the default and
`--fp16` is opt-in.

## Quickstart

Feed it what `geotessera` gives you — the int8 embedding array and its
per-pixel scale plane. You do not need this project's chip store, and
you should not have to know anything about its layout.

```python
from georvq.tessera import load, encode_tile, decode_tile, requantize

R = load("tessera-beta-v1/bundle", device="cuda")

packets, grid = encode_tile(R, embedding, scale)   # (128, H, W) int8, (H, W)
recon = decode_tile(R, packets, grid, depths=[20, 20, 5])   # 3.35 b/px
original_units = requantize(recon)                 # undo the model's rescale

print(R.bpp([20, 20, 5]), "b/px")
for tup in R.ladder():
    print(tup, round(R.bpp(list(tup)), 3))
```

One encode serves every rate: `packets` are full depth, and `depths`
truncates at decode time. A tile is cropped to whole 64 px windows
rather than padded — pad upstream if you need the last partial window.

Already-dequantized float embeddings work too: pass them without a
`scale`. `georvq.tessera.prep` is the single-window form of the same
door, and produces tensors bit-identical to the chip-store path (there
is a test pinning that).

## Format: what a packet carries

Per 64x64 window:

- **indices** — `(cells, depth)` per tier, 8-bit codes: 1, 16 and 256
  cells at grids 1/4/16. Truncating levels at decode time is the rate
  control (the matryoshka property), so one encode serves every rate.
- **the patch mean** — 128 bands x 6 bits = 768 bits, quantized against
  shipped per-band constants (`georvq/inference/mean_codec.py`).
- **an optional fine-tier refinement** — off by default, see below.

**The mean is not optional and not free.** Tessera vectors are not
unit-norm, and the model is trained on a *mean-removed* field with both
encoder and decoder conditioned on the mean, so the mean is part of the
reconstruction rather than a rescaling of it. It is charged at every
rate point: 0.1875 b/px. That matters most at the bottom of the ladder,
where it is most of the bitstream — the AlphaEarth precedent for *not*
charging a stats channel had it silently reach 34% of the rate.

Encoding rounds the mean through the channel and conditions on the
**rounded** vector, so both ends see the same thing. Measured cost of
that rounding at 6 bits/band: **-0.00037 cosine** against the exact float
mean. Below 4 bits/band the channel is worse than not having it.

## Rate control

The efficient ladder saturates coarse, then mid, then adds fine levels
one at a time. Level costs are wildly asymmetric — 8 / 128 / 2,048 bits
for one level of the coarse / mid / fine tier — so hand-picked
"balanced" tuples are strictly dominated.

Measured on 512 held-out windows, decoded through the reloaded bundle:

| tuple | b/px | cos | ccos |
|---|---:|---:|---:|
| 20-0-0 | 0.227 | 0.93844 | 0.40354 |
| 20-20-0 | 0.852 | 0.96076 | 0.63904 |
| 20-20-1 | 1.352 | 0.96388 | 0.66758 |
| 20-20-5 | 3.352 | 0.97206 | 0.74050 |
| 20-20-13 | 7.352 | 0.97754 | 0.78948 |
| 20-20-20 | 10.852 | 0.97971 | 0.80870 |
| 20-20-20 + refinement | 26.852 | 0.98293 | 0.83777 |
| *un-quantized ceiling* | — | 0.98583 | 0.85980 |

Track **`ccos`** (centred cosine), not plain `cos`: with the mean
transmitted, plain cosine is inflated by a channel that is already paid
for. A block-mean reconstruction scores 0.92 plain cosine while carrying
no spatial structure at all.

Pick a rate from the task, not from the cosine. On the five downstream
benchmarks in `scripts/eval_rate_sweep.py` (ASTER GED, LUCAS LC/LU,
LCMAP LC/LU) Tessera is transparent — indistinguishable from the
uncompressed embedding — from about **1 b/px**, i.e. around `20-20-1`.
The rest of the ladder exists for users whose task is not on that list,
and for fine-scale or temporal work that these point-thematic benchmarks
cannot see.

## The fine-tier refinement (opt-in)

After all 20 levels a tier still has a residual. Coding it with ~1 bit
per dimension in a fixed random rotation and adding it back at decode
recovers part of the gap to the un-quantized ceiling, with **no extra
trained codebooks** — the whole "codebook" is a seed plus a Lloyd-Max
table fitted once on held-out residuals.

```python
packets = R.encode(x, valid=valid, refine=True)   # 26.85 b/px
```

**Which tier is refined is a measurement, not a convention.** On this
model (768 held-out windows, share of the ceiling's `ccos` gap
recovered):

| refined tier | added rate | gap closed |
|---|---:|---:|
| 0 (coarse) | +0.13–0.25 b/px | **0.0%** |
| 1 (mid) | +2.0–4.0 b/px | 1.9% |
| 2 (fine), 1 bit | +16 b/px | 56.2% |
| 2 (fine), 2 bits | +32 b/px | 81.2% |

The AlphaEarth release refines tier **0** and recovers ~98% of its gap
there; the identical policy applied here recovers **nothing**. AE's
coarse book carries its error tail while its finer books sit at their
ceiling; Tessera's coarse book is healthy and its fine tier is still
rate-limited, with its residual measurably decaying at level 20. Slack
has to be located per model.

That asymmetry is also why the default differs. A refinement costs
`cells x dim x bits`, so on a 1/4/16 pyramid the same idea is +1.5% of
rate on the coarse tier and **+150%** on the fine one. AE leaves its
sidecar on always; here it is off unless asked for, and it is a
maximum-fidelity option rather than a free win — it more than doubles
the bitstream to recover part of a gap that no benchmark in the suite
can currently resolve.

## Deliberate non-features

- **No COG asset container.** The AlphaEarth release stores packets as
  georeferenced COGs because AEF ships as 1024-px-blocked COG tiles and
  ranged reads fall out for free. Tessera is distributed as `geotessera`
  tiles with a separate per-pixel scale sidecar; the equivalent container
  is worth building when there is a delivery requirement to shape it,
  and packets are container-agnostic until then.
- **No entropy coding.** Order-0 rANS recovered ~0.6% on the AlphaEarth
  release and is not expected to do better here; the remaining gap is
  cross-tier context, which needs the learned context model rather than
  a better static coder.
- **No quality-controlled delivery.** The encode-side self-measurement
  that gives AE its `--quality p90<=0.02` control is source-agnostic and
  could be turned on here; it has not been, because no consumer has
  asked for it.
- **`use_std` is blocked, not merely unused.** Matching the reconstructed
  standard deviation to the source's was measured to *hurt*: the decoder is
  already smoother than the truth, so rescaling amplifies its error instead
  of restoring detail. The config validator refuses it; the numbers are in
  docs/FINDINGS.md.

## Rebuilding

```bash
python scripts/build_release_tessera.py \
    --ckpt <checkpoint.pt> --out <bundle-dir> --gpu 1
```

Fits the mean channel's constants and the refinement table on held-out
windows encoded with the shipped weights, writes the bundle, then
reloads it from disk and measures the ladder — the numbers in this
document are that script's output. Refit both whenever the weights
change: a table fitted against a different checkpoint's residuals is
fitted to the wrong distribution.
