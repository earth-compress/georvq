<!-- DRAFT: wording not yet reviewed by the maintainer. -->

# Why the design is the way it is

This is the short account of the decisions embodied in the code and the
evidence behind them. Each claim cites a record in the findings ledger
(`lab/findings/`, searchable with `lab/query.py`), which holds the
numbers, the context they were measured in, and the results that did not
make it into the design.

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
