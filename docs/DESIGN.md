<!-- DRAFT: wording not yet reviewed by the maintainer. -->

# How georvq works, and why it is built that way

The first half of this document explains the method, including details
about the model architecture and quantisation approach. The second half
gives the reasoning which drove some of those design decisions, citing the findings
ledger (`lab/findings/`, searchable with `lab/query.py`), which gives more detail about specific 
experiments and what was run. Whilst the specific configurations and parameters
differ between data sources, the main method doesn't. For specific solutions that are
now released, see the other docs (`docs/AE_RELEASE.md`,
`docs/TESSERA_RELEASE.md`).

## How it works

### The idea

georvq is a learned compressor for geospatial data. Unlike a classical codec
like JPEG, which applies one fixed mathematical recipe to everything, georvq
trains a neural network on the kind of data it will compress. By doing so, the
network learns what that data tends to look like and can describe it
in fewer bits. Each map tile is cut into small windows, which in the
example used throughout are 64 x 64 pixels with B bands per pixel. An encoder network reduces each window to a
handful of vectors, and those vectors are snapped to entries in learned
lookup tables (codebooks), where each successive codebook approximates the residual from the last, 
getting us closer and closer to the real vector. Only small integer IDs need to be
stored. At decode time, the IDs are turned back into reconstructions of the original latent vectors, then fed through
a model to retrieve estimates of the input pixel values. The IDs are
ordered most-important-first, so one stored file can be decoded at
many quality levels: read a small prefix for a rough preview, or all
of it for a near-exact reconstruction.

### The pipeline

The example geometry here is three tiers on grids of 1, 4 and 16 over
a 64-pixel window; the tier count and every size are per-source.

```
              one 64x64-pixel window (B bands per pixel)
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
      | 1x1 cell  |      | 4x4 cells |      |16x16 cells|  <- grid over the window
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
                 |    attention decoder      |  all 1+16+256 cells attend
                 |   (PyramidAttnDecoder)    |  to each other; the decoder
                 +-------------+-------------+  is also told how many
                               |                codebook levels of each
                               v                tier it actually received
                 +---------------------------+
                 |        DenseHead          |  each fine cell -> a 4x4-px
                 +-------------+-------------+  patch; a small conv
                               |                smooths the seams
                               v
                 +---------------------------+
                 | [optional] denormalization|  undo the source's scheme,
                 +-------------+-------------+  e.g. add the mean back
                               |
                               v
              reconstructed 64x64-pixel window
```



Behind this design are two key principles:

1) Encoding and decoding should occur in a single pass. We do not want
stateful, sequential operations with multiple model passes, unless 
those models are very cheap, vanishingly small parts of the larger DL
model. This is to keep the method somewhat competitive with computationally
cheaper methods like JPEG and zstd.

2) Information in imagery is not spread evenly across scales. "This 
window is mostly farmland" is one fact shared by 4,096 pixels, and 
describing it once at the coarse tier is almost free. Meanwhile, fine texture is 
the opposite: local, unpredictable and expensive. Giving each scale its own 
grid and its own codebooks lets the bit budget go to where the information is.

These two ideas lead to an obvious design decision, that the decoder should see 
everything (all scales) at once, because attention spanning all
tiers means the fine cells are always interpreted in context. A fine
code does not have to spell out "dark green, vegetation-like" from
scratch when the coarse tier has already established it, so the same
fine codebook entry can mean different things in different
surroundings. This sharing of context is a large part of why so few
bits are enough.

Regarding hyperparameters, the window size, tier grids, latent widths d, 
codebook sizes K and depth N are all dependent on the source.
`docs/SOURCE_ONBOARDING.md` is an attempt to help with some empirical analysis
of the data to understand a sensible configuration. However, whilst informed by these
arguments, the design is ultimately arrived at through trial-and-error 
for each source.

### Codebooks, and quantising by successive approximation

A codebook is a learned lookup table, a list of K prototype vectors
that between them cover the kinds of cell summaries the encoder
produces. A cell is stored as the index of its closest prototype, which
costs log2(K) bits however many dimensions the vector has.

One lookup alone is too crude, so georvq quantises in rounds, in the
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

This is residual vector quantisation (residual VQ, or RVQ). Each round
costs one integer per cell, log2(K) bits for that tier's codebook. The
corrections shrink quickly round over round on typical content, and
the size of the late corrections is a useful health signal. Large
corrections in the last rounds mean that the codebooks fit this window
poorly, and their size is used to predict the window's reconstruction
error.

The codebooks are learned during training by exponential moving
averages, with k-means initialisation, and with resets for entries that
fall out of use so that every entry keeps being used. That is most of
`model/vq.py`, and its numerics must never change, because files
encoded with a trained checkpoint have to keep decoding bit-identically
(see the invariants in `docs/OVERVIEW.md`).

### Variable compression from a single model

Many codecs require a quality setting at encode time. georvq is
trained so the choice can wait until decode time, because the encoded
data is structured in an ordered list of importance.

During training, the decoder receives a randomly chosen number of
quantisation rounds per tier for every window, for example every coarse
round, half the mid rounds and no fine rounds at all, and the depth
tuple is passed to the network as conditioning. Because it has been
trained at every budget, the decoder's output becomes gradually worse as
codes are removed, rather than failing. At encode time everything is
stored once at full depth, and readers then decode whatever prefix suits
them.

The tiers have very different costs. A round is log2(K) bits per cell,
and on the 1/4/16 example the fine tier has 256 cells to the coarse
tier's one, so a fine round costs more than two orders of magnitude
more than a coarse round.
The efficient path up the quality ladder is therefore to saturate
coarse first, then mid, then add fine rounds one at a time.
Depth-tuple sweeps on validation data confirm this ordering, with
balanced tuples losing to same-cost alternatives
([`rate-ladder-asymmetry`](../lab/findings/rate-ladder-asymmetry.md),
[`cells-dominate-rate`](../lab/findings/cells-dominate-rate.md)).

### Residual sidecars

Usually, an encoder-decoder's performance will still be significantly
higher when using unquantised vectors, versus the full stream of codebooks.
This is because in high dimensional space, after all N codebooks, a vector 
will still have a significant residual, and that leftover can
be shipped directly: scalar-quantised at a bit or two per number and
added back at decode, without touching the codebooks or the index
stream. This usually works because after many codebooks, the residual
is white noise, and so optimal quantisation is a relatively simple task
and doesn't require more codebooks to get the best performance.
Whether this pays, and on which tier, is a fact about the
trained model rather than the architecture. The available gain is
bounded by the tier's ceiling gap (the improvement in performance when a completely unquantised
vector is used, against using all the codebooks) which is measured first
([`refinement-budget-is-the-ceiling-gap`](../lab/findings/refinement-budget-is-the-ceiling-gap.md)),
and the slack sits on different tiers in different models
([`refinement-tier-differs-per-model`](../lab/findings/refinement-tier-differs-per-model.md)).
On Tessera it is the fine tier. Because a sidecar's cost scales with
the tier's cell count, a coarse sidecar could ship on by default while a
fine one, as on Tessera, has to be an opt-in
([`refinement-cost-scales-with-cells`](../lab/findings/refinement-cost-scales-with-cells.md)).

### Stats normalisation

The stats normalisation module is a key design choice that must be
decided per source. Currently, three modes are supported and used in different
sources. No normalisation, mean-only normalisation (AE and TESSERA) and 
mean+std normalisation (Sentinel-2). If normalisation is used, then the input is
first debiased by the per-band mean of the patch, and divided by the per-band standard deviation,
making each band zero-centred and with a roughly equal spread of values. Whilst removed from the input,
those vectors _are_ seen by the encoder and decoder, through conditioning channels. At the output of the
decoder, the mean (and std) are then reapplied to the data, meaning they need to be sent as part of the 
bitstream if used. The motivation of this approach is to prevent the model from spending lots of
its budget encoding these basic statistics about the data, which can be trivially
done for it, saving its expressive capacity for spatial features and variations within the image.
However, especially for sources with a large number of channels, the number of bits used to encode the 
mean and standard deviation is large compared to the coarse tier's tokens, and so this can significantly 
lift the bitrate of the most extreme compression rates, whilst having very little impact on the bitrate 
of the full bitstream. Therefore, quantisation and compression of those statistics, and the effect of
quality on the decoder, should always be an area of active research for sources.

### Experimental submodules

Other optional features exist: `TierPredictor` (predicts one tier from another and
codes only the surprise; the cascade), `CodebookSelector` (restricts
codebook entries by geographic location), `PyramidRateModel` (a
learned model of which codes are likely, for entropy coding). Most of
these are still experimental and are not yet part of releases.


### The bitstream

The bitstream comprises the IDs of each successive codebook, for each
tier. The codebooks and model are downloaded by the user before any
data can be decoded, which is a real bottleneck in use-cases where the 
amount of data needed is similar in size to the model (on the order of
hundreds of MB for most sources), therefore, no model weights or codebooks
are part of any bitstream; see `docs/BITSTREAM.md` for more information.

## Why it is built that way

Each claim cites a record in the findings ledger, which holds the
numbers and their measurement context.

### Architecture

The encoder is a ViT tokeniser and the decoder an attention trunk
because both beat convolutional alternatives on dense embedding fields
([`vit-beats-conv`](../lab/findings/vit-beats-conv.md)). Convolutional
patch stems remain as the part of the encoder that each source sets for
itself.

One checkpoint serves every rate because training samples monotone
per-tier depth tuples and conditions the decoder on the tuple. Minimum
depths concentrate capacity on the operating points that get used
([`matryoshka-min-depths`](../lab/findings/matryoshka-min-depths.md)). The current recipe (`depth_sampling: path`,
used by the AE and Tessera releases and the S2, DEM and ERA5 presets) samples
depth uniformly along the truncation path so that deep fine-tier levels
see enough training traffic ([`depth-sampling-along-path`](../lab/findings/depth-sampling-along-path.md)). The schema
default is still `nested`, which the older AE presets, including the one
behind the first AE release, were trained with.

### Quantisation

VQ state is kept in fp32 under bf16 training, since lower precision in
the residuals, EMA buffers or codebooks degrades them
([`vq-state-stays-fp32`](../lab/findings/vq-state-stays-fp32.md)). Large codebooks stay in use only with the full set
of health measures, from k-means initialisation to traffic-scaled
dead-code resets ([`codebook-health-machinery`](../lab/findings/codebook-health-machinery.md)).

The codebook size K follows from how many training vectors each
codeword sees. A codebook needs a minimum number of training vectors per codeword inside its EMA window, which ties K to
batch size, cell count and reset cadence, and means K should grow with
tier rather than being fixed globally ([`codebook-sample-budget`](../lab/findings/codebook-sample-budget.md)). Where a
tier is well sampled, larger codebooks seem to beat deeper stacks at matched
rate ([`bigger-k-beats-more-levels`](../lab/findings/bigger-k-beats-more-levels.md)), and the two records state the contexts
in which each applies. Once fitted, codebooks are best frozen, because
continued EMA training can degrade a good fit ([`ema-can-degrade-a-good-fit`](../lab/findings/ema-can-degrade-a-good-fit.md)). 
Codebook quality is judged on rate-distortion, never on usage or
perplexity, because a codebook can reach perfect usage and perplexity
whilst explaining almost none of the energy ([`perplexity-is-gameable`](../lab/findings/perplexity-is-gameable.md)).

### Rate

Cell count quadratically increases in the pyramid, so the efficient ladder
saturates the coarse tier first and adds fine levels last
([`cells-dominate-rate`](../lab/findings/cells-dominate-rate.md), [`rate-ladder-asymmetry`](../lab/findings/rate-ladder-asymmetry.md)). Static entropy coding of
the indices is not worth its decode cost on healthy codebooks. The
achievable gain is in cross-tier context, which needs the learned
context model ([`order0-entropy-coding-marginal`](../lab/findings/order0-entropy-coding-marginal.md)).

### Side channels and refinements

Whether a stats side channel pays is a property of the data, measured
per source. Transmitting the window mean has paid on some fields,
including the unit-norm AlphaEarth field in its current release, whilst
forcing the standard deviation has not paid on any source measured so
far ([`stats-channel-is-per-field`](../lab/findings/stats-channel-is-per-field.md)).

The reasoning behind refinements, with the records it rests on, is
given under Residual sidecars above. One further result is that the
residual itself is best coded with a seeded random rotation and a Lloyd
scalar quantiser ([`rotate-then-lloyd-beats-pca`](../lab/findings/rotate-then-lloyd-beats-pca.md)).

### Conditioning

Location conditioning targets the coarse tier, where location explains
most of the variance it can reach ([`location-explains-coarse-variance`](../lab/findings/location-explains-coarse-variance.md)).
The encoder is frozen while conditioning trains, because co-adaptation
against a frozen prior degrades the model late in training
([`coadaptation-degrades-ceiling`](../lab/findings/coadaptation-degrades-ceiling.md)).

### Evaluation

Splits are geographic, never random, at every level from chips to
latent datasets, because spatial autocorrelation lets a random split
leak near-identical neighbours between training and test data
([`geographic-splits-only`](../lab/findings/geographic-splits-only.md), [`split-latent-data-by-window`](../lab/findings/split-latent-data-by-window.md)).
Quality targets come from downstream tasks, because reconstruction
cosine does not transfer between sources ([`cosine-not-transferable`](../lab/findings/cosine-not-transferable.md)) and
compression can be nearly free downstream long before cosine looks
perfect ([`compression-nearly-free-downstream`](../lab/findings/compression-nearly-free-downstream.md),
[`tessera-transparency-point`](../lab/findings/tessera-transparency-point.md)). A probe's noise floor is established before reading it,
so that a difference smaller than the probe's own variation across split
seeds is not read as a result ([`probe-noise-floors`](../lab/findings/probe-noise-floors.md)). Models with a transmitted
mean are tracked on centred cosine, because plain cosine is inflated by
a mean channel that is already paid for ([`transmitted-mean-inflates-cosine`](../lab/findings/transmitted-mean-inflates-cosine.md)). Classical
baselines are compared at their actual file bytes rather than at a
nominal rate ([`baselines-at-actual-bytes`](../lab/findings/baselines-at-actual-bytes.md)).

### Release engineering

Encoders run fp32 and decoders bf16, which keeps bitstreams bit-stable
across hardware whilst keeping most of the speed gain of bf16 ([`bf16-decoder-only`](../lab/findings/bf16-decoder-only.md)). Encoding
measures its own error by decoding the rate ladder as it goes, which is
what makes quality-controlled delivery possible without a predictor
network ([`encode-time-self-measurement`](../lab/findings/encode-time-self-measurement.md)). Window seams can be removed by a
boundary-conditioned deblocker at decode, at no rate cost
([`seam-deblocker`](../lab/findings/seam-deblocker.md)). No release applies one now, and the
network is kept in `georvq/model/deblock.py` as an exploratory tool.
