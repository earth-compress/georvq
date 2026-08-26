<!-- DRAFT: wording not yet reviewed by the maintainer. -->

# How georvq works, and why it is built that way

The first half of this document explains the method; the second half
gives the reasoning behind the design decisions, citing the findings
ledger (`lab/findings/`, searchable with `lab/query.py`), which holds
the numbers, the context they were measured in, and the results that
did not make it into the design. The method is shared across sources;
every concrete size in it is chosen per source, and the values for a
shipped source are in its release contract (`docs/AE_RELEASE.md`,
`docs/TESSERA_RELEASE.md`).

## Part 1 -- How it works

### The idea

georvq is a learned compressor for geospatial data. A classical codec
like JPEG applies one fixed mathematical recipe to everything; georvq
trains a neural network on the kind of data it will compress, so the
network learns what that data tends to look like and can describe it
in fewer bits. Each map tile is cut into small windows of W x W pixels
with B bands per pixel. An encoder network reduces each window to a
handful of vectors; those vectors are snapped to entries in learned
lookup tables (codebooks), so that only small integer IDs need to be
stored; a decoder network turns the IDs back into pixels. The IDs are
ordered most-important-first, so one stored file can be decoded at
many quality levels: read a small prefix for a rough preview, or all
of it for a near-exact reconstruction.

### The pipeline

Three tiers are shown; the tier count, like every size in the diagram,
is per-source.

```
              one W x W-pixel window (B bands per pixel)
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
                 |      (ViTTokenizer)       |  reads the whole window
                 +-------------+-------------+  at once
                               |
                               |  summarizes it at several scales:
            +------------------+------------------+
            v                  v                  v
      +-----------+      +-----------+      +-----------+
      |  COARSE   |      |    MID    |      |   FINE    |
      | g0 x g0   |      | g1 x g1   |      | g2 x g2   |  <- grid of cells
      |  d0 dims  |      |  d1 dims  |      |  d2 dims  |  <- numbers per cell
      +-----+-----+      +-----+-----+      +-----+-----+
            |  "the whole      |  "this           |  "this small
            |   window is      |   region         |   patch has this
            |   farmland"      |   is..."         |   texture"
            v                  v                  v
      residual VQ        residual VQ        residual VQ      N codebooks
      K0 entries         K1 entries         K2 entries       per tier
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
                 |    attention decoder      |  every cell of every tier
                 |   (PyramidAttnDecoder)    |  attends to every other; it
                 +-------------+-------------+  is also told how many
                               |                codebook levels of each
                               v                tier it actually received
                 +---------------------------+
                 |        DenseHead          |  each fine cell -> its patch
                 +-------------+-------------+  of pixels; a small conv
                               |                smooths the seams
                               v
                 +---------------------------+
                 | [optional] denormalization|  undo the source's scheme,
                 +-------------+-------------+  e.g. add the mean back
                               |
                               v
              reconstructed W x W-pixel window
```

Window size W, the tier grids g, the latent widths d, the codebook
sizes K and the depth N are all derived from measurements of the
source (`docs/SOURCE_ONBOARDING.md`), not copied between sources.

Two design choices do most of the work here.

Several scales, because information in imagery is not spread evenly
across scales. "This window is mostly farmland" is one fact shared by
every pixel, and describing it once at the coarse tier is almost free.
Fine texture is the opposite: local, unpredictable and expensive.
Giving each scale its own grid and its own codebooks lets the bit
budget go to where the information is.

A decoder that sees everything at once, because attention spanning all
tiers means the fine cells are always interpreted in context. A fine
code does not have to spell out "dark green, vegetation-like" from
scratch when the coarse tier has already established it, so the same
fine codebook entry can mean different things in different
surroundings. This sharing of context is a large part of why so few
bits are enough.

### Codebooks, and quantizing by successive approximation

A codebook is a learned lookup table: a list of K prototype vectors
that between them cover the kinds of cell summaries the encoder
produces. To store a cell you do not store its numbers; you find the
closest prototype and store its index, which costs log2(K) bits
however many dimensions the vector has.

One lookup alone is too crude, so georvq quantizes in rounds, in the
way a number is written down digit by digit. Each round has its own
codebook, trained on whatever the previous rounds failed to capture:

```
   target vector z
        |
        v
   book 1: closest of its K entries -> q1          a rough first guess
        |
        v   what is still missing:  z - q1
   book 2: closest entry to that  -> q2            a correction
        |
        v   still missing:  z - q1 - q2
   book 3: closest entry to that  -> q3            a finer correction
        |
       ...                                          (N rounds in total)

   stored on disk:   N small integers per cell
   reconstruction:   q1 + q2 + ... + qL  -- and you may stop at any L
```

This is residual vector quantization (residual VQ, or RVQ). Each round
costs one integer per cell, log2(K) bits for that tier's codebook. The
corrections shrink quickly round over round on typical content, and
the size of the late corrections is a useful health signal: if the
last rounds are still making large moves, the quantizer is struggling
with this window, and exactly that is used as an error predictor.

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
randomly chosen number of quantization rounds per tier -- perhaps
every coarse round, half the mid rounds and no fine at all -- and is
told what it got (the depth tuple conditions the network). Having
practised at every budget, it degrades gracefully instead of failing
when codes are missing. At encode time everything is stored once at
full depth; readers then decode whatever prefix suits them.

The tiers have very different costs. A round is log2(K) bits per cell,
and the fine grid has (g2/g0)^2 times as many cells as the coarse one,
so a fine round can cost orders of magnitude more than a coarse round.
The efficient path up the quality ladder is therefore to saturate
coarse first, then mid, then add fine rounds one at a time. That
ordering is measured, not assumed: depth-tuple sweeps on validation
data show balanced tuples losing to same-cost alternatives
([`rate-ladder-asymmetry`](../lab/findings/rate-ladder-asymmetry.md),
[`cells-dominate-rate`](../lab/findings/cells-dominate-rate.md)).

### Residual sidecars

After all N rounds a tier still has a leftover, and that leftover can
be shipped directly: scalar-quantized at a bit or two per number and
added back at decode, without touching the codebooks or the index
stream. Whether this pays, and on which tier, is a fact about the
trained model rather than the architecture. The available gain is
bounded by the tier's ceiling gap, which is measured first
([`refinement-budget-is-the-ceiling-gap`](../lab/findings/refinement-budget-is-the-ceiling-gap.md)),
and the slack sits on different tiers in different models
([`refinement-tier-differs-per-model`](../lab/findings/refinement-tier-differs-per-model.md)).
On AlphaEarth the coarse tier holds it, where the leftover is
concentrated in unusual windows the coarse codebook never learned well
([`ae-outlier-tail-is-coarse`](../lab/findings/ae-outlier-tail-is-coarse.md));
on Tessera it is the fine tier. Because a sidecar's cost scales with
the tier's cell count, a coarse sidecar can ship on by default while a
fine one has to be an opt-in
([`refinement-cost-scales-with-cells`](../lab/findings/refinement-cost-scales-with-cells.md)).

### Optional submodules

Model variants are expressed as the presence of submodules, never as
runtime flags: `TierPredictor` (predicts one tier from another and
codes only the surprise; the cascade), `CodebookSelector` (restricts
codebook entries by geographic location), `PyramidRateModel` (a
learned model of which codes are likely, for entropy coding). All
exist and all were trained; none is in a shipped release. Part 2 and
the ledger record what each one bought and why it was left out.

### What lands on disk

The bitstream is the indices plus what is needed to interpret them:
the tiers, the codebooks, any conditioning priors and any side
channels. It deliberately excludes the decoder, so decoders can be
upgraded forever without re-encoding (`docs/BITSTREAM.md`). Two
identity hashes guard correctness: `bitstream_id` names everything
that defines the bitstream, so a mismatched model refuses to decode
rather than decode wrongly, and `decoder_id` names the decoder
weights, keying decoder-specific artifacts such as measured quality
maps.

How the bitstream is packaged is release-layer policy, chosen per
source. The AlphaEarth release stores the indices as georeferenced
rasters in which bands are quantization rounds, so standard geospatial
tooling can read any region at any rate by band selection
(`docs/AE_RELEASE.md`); the Tessera release uses a different container
with a transmitted mean and an opt-in fine-tier sidecar
(`docs/TESSERA_RELEASE.md`).

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
