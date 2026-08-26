<!-- DRAFT: wording not yet reviewed by the maintainer. -->

# How georvq works, and why it is built that way

The first half of this document explains the method, using the shipped
AlphaEarth release (`ae_pyr64_divfull`) as the running example: 64x64
pixel windows, each pixel a 64-band unit-length embedding vector. The
second half gives the reasoning behind the design decisions, citing the
findings ledger (`lab/findings/`, searchable with `lab/query.py`),
which holds the numbers, the context they were measured in, and the
results that did not make it into the design.

## Part 1 -- How it works

### The idea

georvq is a learned compressor for geospatial data. A classical codec
like JPEG applies one fixed mathematical recipe to everything; georvq
trains a neural network on the kind of data it will compress, so the
network learns what that data tends to look like and can describe it
in fewer bits. Each map tile is cut into small windows. An encoder
network reduces each window to a handful of vectors; those vectors are
snapped to entries in learned lookup tables (codebooks), so that only
small integer IDs need to be stored; a decoder network turns the IDs
back into pixels. The IDs are ordered most-important-first, so one
stored file can be decoded at many quality levels: read a small prefix
for a rough preview, or all of it for a near-exact reconstruction.

### The pipeline

```
             one 64x64-pixel window (64 bands per pixel)
                               |
                               v
                 +---------------------------+
                 | [optional] normalization  |  source-dependent, e.g.
                 |       (stats_cond)        |  remove the per-window mean
                 +-------------+-------------+  and send it as a side channel
                               |
                               v
                 +---------------------------+
                 |        ViT encoder        |  a vision transformer that
                 |      (ViTTokenizer)       |  reads the whole window at
                 +-------------+-------------+  once (9 blocks, width 256)
                               |
                               |  summarizes it at three scales:
            +------------------+------------------+
            v                  v                  v
      +-----------+      +-----------+      +-----------+
      |  COARSE   |      |    MID    |      |   FINE    |
      | 1x1 cell  |      | 4x4 cells |      |16x16 cells|  <- grid over the window
      |   256-d   |      |   192-d   |      |   64-d    |  <- numbers per cell (AE)
      +-----+-----+      +-----+-----+      +-----+-----+
            |  "the whole      |  "this           |  "this 4x4-px
            |   window is      |   quadrant-ish   |   patch has this
            |   farmland"      |   area is..."    |   texture"
            v                  v                  v
      residual VQ        residual VQ        residual VQ      8 codebooks
      2,048 entries      8,192 entries      16,384 entries   per tier (AE)
            |                  |                  |
            +------------------+------------------+
                               v
             +=======================================+
             |  the bitstream: one integer ID per    |
             |  cell per codebook -- this is all     |
             |  that ever gets written to disk       |
             +=======================================+
                               |
                               v
                 +---------------------------+
                 |    attention decoder      |  all 1+16+256 cells attend to
                 |   (PyramidAttnDecoder)    |  each other (12 blocks, width
                 +-------------+-------------+  384); it is also told how
                               |                many codebook levels of each
                               v                tier it actually received
                 +---------------------------+
                 |        DenseHead          |  each fine cell -> a 4x4-px
                 +-------------+-------------+  patch; a small conv smooths
                               |                the seams
                               v
                 +---------------------------+
                 | [optional] denormalization|  undo the source's scheme:
                 +-------------+-------------+  add the mean back, rescale
                               |                (AE: to unit length)
                               v
              reconstructed 64x64-pixel window
```

The structure is shared; the sizes are not. Window size, tier grids,
latent widths, codebook sizes and the number of codebooks are chosen
per source, derived from measurements of the data
(`docs/SOURCE_ONBOARDING.md`); the figures shown are the AlphaEarth
release's.

Two design choices do most of the work here.

Three scales, because information in imagery is not spread evenly
across scales. "This window is mostly farmland" is one fact shared by
4,096 pixels, and describing it once at the coarse tier is almost
free. Fine texture is the opposite: local, unpredictable and
expensive. Giving each scale its own grid and its own codebooks lets
the bit budget go to where the information is.

A decoder that sees everything at once, because attention spanning all
three tiers means the fine cells are always interpreted in context. A
fine code does not have to spell out "dark green, vegetation-like"
from scratch when the coarse tier has already established it, so the
same fine codebook entry can mean different things in different
surroundings. This sharing of context is a large part of why so few
bits are enough.

### Codebooks, and quantizing by successive approximation

A codebook is a learned lookup table: a list of, say, 2,048 prototype
vectors that between them cover the kinds of cell summaries the
encoder produces. To store a cell you do not store its numbers; you
find the closest prototype and store its index. 2,048 entries means 11
bits, however many dimensions the vector has.

One lookup alone is too crude, so georvq quantizes in rounds, in the
way a number is written down digit by digit. Each round has its own
codebook, trained on whatever the previous rounds failed to capture:

```
   target vector z
        |
        v
   book 1: closest of its 2,048 entries -> q1      a rough first guess
        |
        v   what is still missing:  z - q1
   book 2: closest entry to that  -> q2            a correction
        |
        v   still missing:  z - q1 - q2
   book 3: closest entry to that  -> q3            a finer correction
        |
       ...                                          (8 rounds in total)

   stored on disk:   8 small integers per cell
   reconstruction:   q1 + q2 + ... + qL  -- and you may stop at any L
```

This is residual vector quantization (residual VQ, or RVQ). Each round
costs one integer per cell: 11, 13 or 14 bits for the three tiers. The
corrections shrink quickly round over round on typical content, and
the size of the late corrections is a useful health signal: if round 7
is still making large moves, the quantizer is struggling with this
window, and exactly that is used as an error predictor.

The codebooks are learned during training by exponential moving
averages, with the machinery that keeps every entry in use: k-means
initialization, resets for entries that fall out of use. That is most
of `model/vq.py`, and its numerics must never drift, because the value
of the repository rests on trained checkpoints decoding bit-identically
forever.

### One model, every compression level

Most codecs require a quality setting at encode time. georvq is
trained so the choice can wait until decode time, with qualities
nested inside one bitstream the way matryoshka dolls nest.

The mechanism is in training. For every window the decoder is handed a
randomly chosen number of quantization rounds per tier -- perhaps all
8 coarse rounds, 4 mid and no fine at all -- and is told what it got
(the depth tuple conditions the network). Having practised at every
budget, it degrades gracefully instead of failing when codes are
missing. At encode time everything is stored once at full depth;
readers then decode whatever prefix suits them:

```
  reconstruction quality (cosine similarity, held-out data)

  0.998 +                                              * 8-8-8 + sidecar
  0.994 +                    * 8-8-1
  0.990 +        * 8-8-0
  0.977 + * 8-0-0
        +--+---------+---------+-------------------------+--
           88      1,752     5,336                    30,952
                                                bits per window

  "8-8-1" = all 8 coarse rounds, all 8 mid, 1 fine.
```

The tiers have very different costs: one round is 11 bits at the
coarse tier, 208 at mid (13 bits x 16 cells), 3,584 at fine (14 bits x
256 cells). The efficient path up the quality ladder is therefore to
saturate coarse first, then fill mid, then add fine rounds one at a
time. That ordering is measured, not assumed: a full sweep of depth
combinations on the validation set confirms it, and hand-picked
"balanced" tuples such as 6-4-4 lose to same-cost alternatives
([`rate-ladder-asymmetry`](../lab/findings/rate-ladder-asymmetry.md)).

### The 8* sidecar

After all 8 rounds the coarse tier's approximation is still not exact,
and the remaining error is concentrated in unusual windows: rare land
covers the 2,048-entry codebook never learned well, and whole-window
effects the finer tiers cannot make up for. Since the average windows
were already fine and only the outliers hurt, the release ships the
final coarse leftover directly, rounded to 2 bits per number (66 bytes
per window). It costs +1.5% file size and cuts the worst pixel errors
(99.9th percentile) by about 3x
([`ae-outlier-tail-is-coarse`](../lab/findings/ae-outlier-tail-is-coarse.md)).

The same idea was tested on the mid and fine tiers and found
worthless: those codebooks already capture what they are going to, and
do it about 4.6x cheaper than the rounding approach at equal quality
([`refinement-budget-is-the-ceiling-gap`](../lab/findings/refinement-budget-is-the-ceiling-gap.md)).

### Optional submodules

Model variants are expressed as the presence of submodules, never as
runtime flags: `TierPredictor` (predicts one tier from another and
codes only the surprise; the cascade), `CodebookSelector` (restricts
codebook entries by geographic location), `PyramidRateModel` (a
learned model of which codes are likely, for entropy coding). All
exist and all were trained; none made the AlphaEarth release. Part 2
and the ledger record what each one bought and why it was left out.

### What lands on disk (AE release)

The unit of deployment is a satellite image tile, and the compressed
form is itself a set of ordinary georeferenced rasters. The code IDs
are pixel values, so standard geospatial tooling (GDAL) can read any
region of any tier without touching the rest:

```
   source COG tile (8192 px, 2.25 GB)
        |
        v  encode (GPU, ~76 s -- includes measuring its own quality)
        |
        +- coarse.tif    128x128, 8 bands  -+   the code IDs, one band
        +- mid.tif       512x512, 8 bands   +-  per quantization round,
        +- fine.tif    2048x2048, 8 bands  -+   as georeferenced COGs
        +- star.tif      the 8* sidecar (66 B/window)
        +- mask.tif      validity mask (nodata restored exactly)
        +- quality/<decoder_id>.tif   measured error per window,
        |                             per quality level
        +- metadata.json              stats, identity hashes, profile

   total: ~58 MB  (vs 2.25 GB source)
```

Because bands are quantization rounds, rate control at read time is
band selection; and because the internal block layout mirrors the
source's 1024 px blocks, ranged partial reads work with plain GDAL.
Two identity hashes guard correctness: `bitstream_id` (the codebooks;
a mismatched model refuses to decode rather than decode wrongly) and
`decoder_id` (the decoder weights, which key the quality rasters).

## Part 2 -- Why it is built that way

Each claim cites a record in the findings ledger, which holds the
numbers and their measurement context.

## Architecture

The encoder is a ViT tokenizer and the decoder an attention trunk
because both beat convolutional alternatives on dense embedding fields
([`vit-beats-conv`](../lab/findings/vit-beats-conv.md)). Convolutional patch stems remain as the per-source
plug point.

The bitstream excludes the decoder because reconstruction quality was
found to be limited by decoder capacity rather than by the stored
latents. A stored file can therefore benefit from every future decoder
improvement without re-encoding ([`decoder-is-the-bottleneck`](../lab/findings/decoder-is-the-bottleneck.md),
docs/BITSTREAM.md).

One checkpoint serves every rate because training samples monotone
per-tier depth tuples and conditions the decoder on the tuple. Minimum
depths concentrate capacity on the operating points that get used
([`matryoshka-min-depths`](../lab/findings/matryoshka-min-depths.md)), and depth is sampled uniformly along the
truncation path so that deep fine-tier levels see enough training
traffic ([`depth-sampling-along-path`](../lab/findings/depth-sampling-along-path.md)).

## Quantization

VQ state is kept in fp32 under bf16 training, since lower precision in
the residuals, EMA buffers or codebooks degrades them
([`vq-state-stays-fp32`](../lab/findings/vq-state-stays-fp32.md)). Large codebooks survive only with the full set
of health measures, from k-means init to traffic-scaled dead-code
resets ([`codebook-health-machinery`](../lab/findings/codebook-health-machinery.md)).

Codebook size is derived, not chosen. A codebook needs a minimum number
of training vectors per codeword inside its EMA window, which ties K to
batch size, cell count and reset cadence, and means K should grow with
tier rather than being fixed globally ([`codebook-sample-budget`](../lab/findings/codebook-sample-budget.md)). Where a
tier is well sampled, larger codebooks beat deeper stacks at matched
rate ([`bigger-k-beats-more-levels`](../lab/findings/bigger-k-beats-more-levels.md)); the two records state the contexts
in which each applies. Once fitted, codebooks are best frozen, because
continued EMA training can degrade a good fit ([`ema-can-degrade-a-good-fit`](../lab/findings/ema-can-degrade-a-good-fit.md)). Codebook quality is judged on rate-distortion, never on usage or
perplexity, which can be gamed to perfection while explaining almost
none of the energy ([`perplexity-is-gameable`](../lab/findings/perplexity-is-gameable.md)).

## Rate

Cell count dominates rate on a pyramid, so the efficient ladder
saturates the coarse tier first and adds fine levels last
([`cells-dominate-rate`](../lab/findings/cells-dominate-rate.md), [`rate-ladder-asymmetry`](../lab/findings/rate-ladder-asymmetry.md)). Static entropy coding of
the indices is not worth its decode cost on healthy codebooks; the
achievable gain sits in cross-tier context and needs the learned
context model ([`order0-entropy-coding-marginal`](../lab/findings/order0-entropy-coding-marginal.md)).

## Side channels and refinements

Whether a stats side channel pays is a property of the data, measured
per source. Some fields want the window mean transmitted, none measured
so far wants the standard deviation forced, and a unit-norm field wants
nothing ([`stats-channel-is-per-field`](../lab/findings/stats-channel-is-per-field.md)).

A tier's post-quantization residual can be scalar-quantized and added
back at decode without touching the bitstream. The available gain is
bounded by the tier's ceiling gap, which must be measured first
([`refinement-budget-is-the-ceiling-gap`](../lab/findings/refinement-budget-is-the-ceiling-gap.md)); which tier holds the slack is a
fact about the model, not the architecture
([`refinement-tier-differs-per-model`](../lab/findings/refinement-tier-differs-per-model.md)); and cost scales with cell count,
which is why a coarse refinement can ship on by default while a fine
one is an opt-in ([`refinement-cost-scales-with-cells`](../lab/findings/refinement-cost-scales-with-cells.md)). The residual
itself is best coded with a seeded random rotation and a Lloyd scalar
quantizer ([`rotate-then-lloyd-beats-pca`](../lab/findings/rotate-then-lloyd-beats-pca.md)).

## Conditioning

Location conditioning targets the coarse tier, where location explains
most of the variance it can reach ([`location-explains-coarse-variance`](../lab/findings/location-explains-coarse-variance.md)).
The encoder is frozen while conditioning trains, because co-adaptation
against a frozen prior degrades the model late in training
([`coadaptation-degrades-ceiling`](../lab/findings/coadaptation-degrades-ceiling.md)).

## Evaluation

Splits are geographic, never random, at every level from chips to
latent datasets ([`geographic-splits-only`](../lab/findings/geographic-splits-only.md), [`split-latent-data-by-window`](../lab/findings/split-latent-data-by-window.md)).
Quality targets come from downstream tasks, because reconstruction
cosine does not transfer between sources ([`cosine-not-transferable`](../lab/findings/cosine-not-transferable.md)) and
compression can be nearly free downstream long before cosine looks
perfect ([`compression-nearly-free-downstream`](../lab/findings/compression-nearly-free-downstream.md),
[`tessera-transparency-point`](../lab/findings/tessera-transparency-point.md)). A probe's noise floor is established before reading it
([`probe-noise-floors`](../lab/findings/probe-noise-floors.md)), models with a transmitted mean are tracked on
centred cosine ([`transmitted-mean-inflates-cosine`](../lab/findings/transmitted-mean-inflates-cosine.md)), and classical
baselines are compared at actual file bytes ([`baselines-at-actual-bytes`](../lab/findings/baselines-at-actual-bytes.md)).

## Release engineering

Encoders run fp32 and decoders bf16, which keeps bitstreams bit-stable
across hardware at most of the speed win ([`bf16-decoder-only`](../lab/findings/bf16-decoder-only.md)). Encoding
measures its own error by decoding the rate ladder as it goes, which is
what makes quality-controlled delivery possible without a predictor
network ([`encode-time-self-measurement`](../lab/findings/encode-time-self-measurement.md)). Window seams are removed by a
boundary-conditioned deblocker at decode, at no rate cost
([`seam-deblocker`](../lab/findings/seam-deblocker.md)).
