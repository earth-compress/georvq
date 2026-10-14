# The Tessera release codec

This document describes the released georvq codec for
[Tessera](https://github.com/ucam-eo/geotessera) embedding tiles. With
it, a Tessera user can encode the int8 embeddings and per-pixel scale
that `geotessera` provides, and decode them at any rate from 0.23 to
10.9 b/px. Everything documented here is release policy for this source
(`georvq/tessera/`), built on the source-agnostic code in
`georvq/inference/`. The AlphaEarth release (`docs/AE_RELEASE.md`) is
the other worked example, and the two differ in how the mean is coded,
in the container and in whether a refinement is offered.

## Who is this for

| you are | you touch |
|---|---|
| Tessera data user, decoding compressed windows | `georvq.tessera.load`, the Packet format and Rate control sections; never the encoder or training code |
| codec researcher, training or improving models | `georvq/{model,train,...}`, `docs/OVERVIEW.md`; this package is a worked example of a release layer |

## The release

The released model is `tessera_vq8_k256_statscond` at step 52,000
(cumulative 71,000, at which point the run completed), with window 64
and tiers `1:512:256:20 / 4:512:256:20 / 16:256:256:20`. The bundle is
at `/data/georvq-releases/tessera-beta-v1/bundle/`. It holds fp16
weights (235 MiB) in `model.pt`, together with `mean_channel.npz`,
`refine_table.npz` and `bundle.json`, which carries `bitstream_id` and
`decoder_id`.

The release is a beta, cut from the completed run. The last 20,000
steps bought +0.0002 cosine and +0.0018 centred cosine over the mid-run
checkpoint, so further training was no longer changing the result. A
replacement checkpoint can still be dropped in without changing any of
the policy below, and will change `bitstream_id`.

## Getting it

The weights are not in git. On the group server the bundle is at
`/data/georvq-releases/tessera-beta-v1/bundle`, and elsewhere you should
copy `tessera-beta-v1.tar.gz` (216 MB) and verify it before loading.

```bash
sha256sum -c <<< "ca776bbe0d679640d20193ded63863e9db7aa385cd5473fa61c2acc3ce38edb9  tessera-beta-v1.tar.gz"
tar xzf tessera-beta-v1.tar.gz
pip install -e .                  # decoding needs no extras at all
```

The tarball only contains data, in four files, and all the executable
code is in the `georvq` package. Keeping the decoder out of the bundle
is what lets a better decoder ship later against the same
`bitstream_id` without re-encoding anything.

    bundle/model.pt            weights + the embedded resolved config
    bundle/mean_channel.npz    the mean channel's per-band constants
    bundle/refine_table.npz    the fine-tier Lloyd table
    bundle/bundle.json         bitstream_id, decoder_id, tiers, refinement spec

Always verify the checksum. `model.pt` is a torch pickle with the
config stored inside it. It is loaded through
`georvq.repro.load_checkpoint` with `weights_only=True`, which refuses
pickled code, and `georvq.tessera.load` checks the `bitstream_id`, but
only the checksum shows that the bundle is the one released.

## Weight precision

The fp16 build is what ships, with `bitstream_id d5051eb0ac4f61a8`. The
fp32 reference, which is exactly what was trained and validated (id
`3db03c9b246d9c94`), stays on the group server at
`tessera-beta-v1/bundle-fp32/` for anyone who wants to reproduce the
validation numbers.

| | model.pt | tarball |
|---|---:|---:|
| fp32 reference | 469 MiB | -- |
| **fp16, shipped** | **235 MiB** | **216 MB** |

The builder defaults to fp32 (`--fp16` opts in), so the reference
artifact is always what was validated and the smaller build has to be
requested explicitly.

The weights are stored in fp16 but computed in fp32, and decode quality
is unaffected. Rounding all 123M parameters to fp16 moved full-depth
cosine by -8.8e-07 and centred cosine by -8.0e-07 over 1,024 held-out
windows, about 260x smaller than the difference between two consecutive
release checkpoints. `load_state_dict` copies fp16 tensors into the fp32
model and upcasts, so the arithmetic afterwards is bit-identical to an
fp32 release, and index determinism across hardware is whatever it
already was.

The two variants deliberately carry different `bitstream_id`s. The id
hashes the weights that ship, so rehashing a bundle reproduces its
stated id, and `georvq.tessera.load` recomputes it from the files it
loaded and refuses a mismatch. An fp16 bundle's codebooks hold different
values, so the same indices decode to a very slightly different field
and it is treated as a different bitstream. Pick one variant per
archive and stay on it.

Computing in fp16 would be a different change. A narrower accumulate
makes near-tie RVQ assignments depend on the hardware, and because the
encoder's assignments are the bitstream, that changes the files
themselves. Rounding the encoder's weights to fp16 already moves 1.8% of
indices relative to fp32 (at no cost to quality, since they are
near-ties by definition). This does no harm as long as only one encoder is in
use. If an fp16 and an fp32 encoder were both in use, however, the same
tile would be written as two different files depending on which encoder
ran, which is why fp32 stays the default and `--fp16` is opt-in.

## Quickstart

The codec takes what `geotessera` gives you, which is the int8 embedding
array and its per-pixel scale plane. You do not need this project's chip store, and
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
rather than padded, so pad upstream if you need the last partial window.
`grid` carries the tile's validity, and `decode_tile` returns nodata
pixels as zero vectors.

Already-dequantised float embeddings work too: pass them without a
`scale`. `georvq.tessera.prep` is the single-window form of the same
interface, and produces tensors bit-identical to the chip-store path
(there is a test pinning that).

## Packet format

A packet carries the following for each 64x64 window.

- The indices, `(cells, depth)` per tier with 8-bit codes, for 1, 16
  and 256 cells at grids 1/4/16. Truncating levels at decode time sets
  the rate (the matryoshka property), so one encode serves every rate.
- The patch mean, 128 bands x 6 bits = 768 bits, quantised against
  shipped per-band constants (`georvq/inference/mean_codec.py`).
- An optional fine-tier refinement, which is off by default (see below).

The mean is always transmitted and always charged. Tessera vectors
are not unit-norm, and the model is trained on a mean-removed field
with both encoder and decoder conditioned on the mean, so the mean is
part of the reconstruction rather than a rescaling of it. It costs
0.1875 b/px at every rate point, which matters most at the bottom of
the ladder, where it is most of the bitstream. The reason every stats
channel is charged in the rate figures is given in
`docs/ADDING_A_SOURCE.md`.

Encoding rounds the mean through the channel and conditions on the
rounded vector, so both ends see the same thing. At 6 bits/band that
rounding was measured to cost -0.00037 cosine against the exact float
mean, and below 4 bits/band the channel is worse than not having it.

## Rate control

The efficient ladder saturates coarse, then mid, then adds fine levels
one at a time. Level costs are very asymmetric (8 / 128 / 2,048 bits
for one level of the coarse / mid / fine tier), so hand-picked
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

Track `ccos` (centred cosine) rather than plain `cos`: with the mean
transmitted, plain cosine is inflated by a channel that is already paid
for. A block-mean reconstruction scores 0.92 plain cosine whilst carrying
no spatial structure at all.

Pick a rate from the task rather than from the cosine. On the five downstream
benchmarks in `scripts/eval_rate_sweep.py` (ASTER GED, LUCAS LC/LU,
LCMAP LC/LU), Tessera is transparent from about 1 b/px, i.e. around
`20-20-1`, which means that above that rate the probes cannot tell the
decoded embedding from the uncompressed one.
The rest of the ladder exists for users whose task is not on that list,
and for fine-scale or temporal work that these point-thematic benchmarks
cannot see.

## The fine-tier refinement (opt-in)

After all 20 levels a tier still has a residual. Coding it with ~1 bit
per dimension in a fixed random rotation and adding it back at decode
recovers part of the gap to the unquantised ceiling with no extra
trained codebooks: the refinement needs only a rotation seed and a
Lloyd-Max table fitted once on held-out residuals.

```python
packets = R.encode(x, valid=valid, refine=True)   # 26.85 b/px
```

Which tier to refine is measured per model. On this
model (768 held-out windows, share of the ceiling's `ccos` gap
recovered):

| refined tier | added rate | gap closed |
|---|---:|---:|
| 0 (coarse) | +0.13–0.25 b/px | **0.0%** |
| 1 (mid) | +2.0–4.0 b/px | 1.9% |
| 2 (fine), 1 bit | +16 b/px | 56.2% |
| 2 (fine), 2 bits | +32 b/px | 81.2% |

The first AlphaEarth release (v1, since replaced) refined tier 0 and
recovered ~98% of its gap there, whereas the same policy recovers 0.0%
on this model. That AE model's coarse book carried its error tail whilst
its finer books sat at their ceiling, but Tessera's coarse book is
healthy and its fine tier is still rate-limited, with its residual
measurably decaying at level 20. The tier to refine therefore has to be
measured for each model.

That asymmetry is also why the default differs. A refinement costs
`cells x dim x bits`, so on a 1/4/16 pyramid a 1-bit refinement adds about 1.2%
of rate on the coarse tier and about 150% on the fine one. The first AE
release left its coarse refinement on always (the current AE release
has none), whereas here it is off unless asked for, and it is a
maximum-fidelity option rather than a default, because it more than
doubles the bitstream to recover part of a gap that no benchmark in
the suite can currently resolve.

## What the release does not include

- There is no container. The AlphaEarth release stores packets in one
  GeoParquet file per source tile (`docs/GEORVQ_PARQUET.md`), mirroring
  the source's tiles so that a ranged read fetches only what a window
  and rate need. Tessera is distributed as `geotessera`
  tiles with a separate per-pixel scale sidecar. A container for it is
  worth building when there is a delivery requirement to shape it, and
  packets are container-agnostic until then.
- There is no entropy coding. Order-0 rANS recovered ~0.6% on the first
  AlphaEarth release and is not expected to do better here, because the
  remaining gap is cross-tier context, which needs the learned context
  model rather than a better static coder.
- There is no quality-controlled delivery. The encode-side
  self-measurement behind AE's quality bounds (`quality="p99<=0.02"`, on
  the `mean` or `p99` of per-pixel `1 - cos`) could be applied here, and
  the GeoParquet container has a column for it, but it has not been
  applied because no user has asked for it.
- The release does not use `use_std`, and a model trained with it
  should not be shipped, because matching the reconstructed standard deviation
  to the source's was measured to hurt: the decoder is already smoother
  than the truth, so rescaling amplifies its error instead of restoring
  detail. The numbers are in `lab/findings/stats-channel-is-per-field.md`.

## Rebuilding

```bash
python scripts/build_release_tessera.py \
    --ckpt <checkpoint.pt> --out <bundle-dir> --device cuda
```

The script fits the mean channel's constants and the refinement table on
held-out windows encoded with the shipped weights, writes the bundle,
then reloads it from disk and measures the ladder, and the numbers in
this document are its output. Refit both whenever the weights change,
because a table fitted against a different checkpoint's residuals is
fitted to the wrong distribution.
