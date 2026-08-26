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
(vit-beats-conv). Convolutional patch stems remain as the per-source
plug point.

The bitstream excludes the decoder because reconstruction quality was
found to be limited by decoder capacity rather than by the stored
latents. A stored file can therefore benefit from every future decoder
improvement without re-encoding (decoder-is-the-bottleneck,
docs/BITSTREAM.md).

One checkpoint serves every rate because training samples monotone
per-tier depth tuples and conditions the decoder on the tuple. Minimum
depths concentrate capacity on the operating points that get used
(matryoshka-min-depths), and depth is sampled uniformly along the
truncation path so that deep fine-tier levels see enough training
traffic (depth-sampling-along-path).

## Quantization

VQ state is kept in fp32 under bf16 training, since lower precision in
the residuals, EMA buffers or codebooks degrades them
(vq-state-stays-fp32). Large codebooks survive only with the full set
of health measures, from k-means init to traffic-scaled dead-code
resets (codebook-health-machinery).

Codebook size is derived, not chosen. A codebook needs a minimum number
of training vectors per codeword inside its EMA window, which ties K to
batch size, cell count and reset cadence, and means K should grow with
tier rather than being fixed globally (codebook-sample-budget). Where a
tier is well sampled, larger codebooks beat deeper stacks at matched
rate (bigger-k-beats-more-levels); the two records state the contexts
in which each applies. Once fitted, codebooks are best frozen, because
continued EMA training can degrade a good fit (ema-can-degrade-a-good-fit). Codebook quality is judged on rate-distortion, never on usage or
perplexity, which can be gamed to perfection while explaining almost
none of the energy (perplexity-is-gameable).

## Rate

Cell count dominates rate on a pyramid, so the efficient ladder
saturates the coarse tier first and adds fine levels last
(cells-dominate-rate, rate-ladder-asymmetry). Static entropy coding of
the indices is not worth its decode cost on healthy codebooks; the
achievable gain sits in cross-tier context and needs the learned
context model (order0-entropy-coding-marginal).

## Side channels and refinements

Whether a stats side channel pays is a property of the data, measured
per source. Some fields want the window mean transmitted, none measured
so far wants the standard deviation forced, and a unit-norm field wants
nothing (stats-channel-is-per-field).

A tier's post-quantization residual can be scalar-quantized and added
back at decode without touching the bitstream. The available gain is
bounded by the tier's ceiling gap, which must be measured first
(refinement-budget-is-the-ceiling-gap); which tier holds the slack is a
fact about the model, not the architecture
(refinement-tier-differs-per-model); and cost scales with cell count,
which is why a coarse refinement can ship on by default while a fine
one is an opt-in (refinement-cost-scales-with-cells). The residual
itself is best coded with a seeded random rotation and a Lloyd scalar
quantizer (rotate-then-lloyd-beats-pca).

## Conditioning

Location conditioning targets the coarse tier, where location explains
most of the variance it can reach (location-explains-coarse-variance).
The encoder is frozen while conditioning trains, because co-adaptation
against a frozen prior degrades the model late in training
(coadaptation-degrades-ceiling).

## Evaluation

Splits are geographic, never random, at every level from chips to
latent datasets (geographic-splits-only, split-latent-data-by-window).
Quality targets come from downstream tasks, because reconstruction
cosine does not transfer between sources (cosine-not-transferable) and
compression can be nearly free downstream long before cosine looks
perfect (compression-nearly-free-downstream,
tessera-transparency-point). A probe's noise floor is established before reading it
(probe-noise-floors), models with a transmitted mean are tracked on
centred cosine (transmitted-mean-inflates-cosine), and classical
baselines are compared at actual file bytes (baselines-at-actual-bytes).

## Release engineering

Encoders run fp32 and decoders bf16, which keeps bitstreams bit-stable
across hardware at most of the speed win (bf16-decoder-only). Encoding
measures its own error by decoding the rate ladder as it goes, which is
what makes quality-controlled delivery possible without a predictor
network (encode-time-self-measurement). Window seams are removed by a
boundary-conditioned deblocker at decode, at no rate cost
(seam-deblocker).
