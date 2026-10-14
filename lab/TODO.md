# TODO

Testable hypotheses, queued experiments and future work, posted by the
maintainer. Agents: read this before proposing new experimental
directions, and take an item only when asked. When an item is done, its
result goes through the normal findings process and the entry here is
marked done with a link to the record.

Format per entry: date, status (open | in-progress | done), the idea,
why it matters, and how to test it, with links to related findings and
tools.

---

## Cross-tier conditional entropy on Tessera

*2026-08-26, open.*

Hypothesis: a healthy model should show little predictability between
tiers, because correlation between a tier's codes and the coarser
tiers' vectors means information is being spent twice. Conditional
entropy of each tier's indices given the coarser tiers is therefore
both a health metric and a measure of how much rate an entropy coder
could recover.

We measured this on AlphaEarth, beta_v2 and DEM but never on Tessera
(see [`order0-entropy-coding-marginal`](findings/order0-entropy-coding-marginal.md),
where the cross-tier context model recovered meaningful rate that
order-0 coding could not).

Test: run
`lab/experiments/order0-entropy-coding-marginal/probe_index_entropy.py`
with the Tessera release checkpoint against
`/data/tessera-samples/chips_512`. Needs GPU; posted while only CPU is
available. Compare the per-tier conditional gaps with the AE numbers,
and read the result against
[`tier-redundancy-collapse`](findings/tier-redundancy-collapse.md),
which is the extreme case of the same waste.


## Use-case driven compression

*2026-08-26, open.*

Hypothesis: low-loss compression is generically useful across many
use-cases. However, as compression rate increases, and information is
lost, we should be able to engineer compressed representations that 
maintain performance in some specific downstream task. This is most
easily done by a new dedicated model, but feasibly, the same encoder 
and decoder could be conditioned on what metric to optimize for. For
example, the loss could be the divergence of results for a downstream
model between the uncompressed and decoded versions. 

This requires a library of downstream metrics/models that are used to
empirically guide the model towards certain use-cases, against others.


## Spatial window size versus performance

*2026-08-26, open.*

Hypothesis: comparing well-trained models of different patch sizes, 
assuming the same bits per pixel, there will be a very small gain 
at high bit rates when using a larger window. However, at extreme
compression rates, the relative performance gain will be much higher,
because the informational overlap between the coarse information 
across a larger image is much greater than the overlap of the information
held in the fine-scale, high bitrate tiers. 

Given the difficulties in optimize larger and larger models, it is 
likely that the best high bitrate models will be trained for relatively 
small patches.

Existing tools: no new tool is needed for the direct test. Window size
is a config field, so this is two or more training runs at different W
with `scripts/eval_rate_sweep.py` comparing the fronts at matched bits
per pixel. On the predictive side, `analysis/data_probe.py` measures
autocorrelation range and `analysis/pyramid_sizing.py` proposes
candidate windows and grids, but the power-spectrum patch-size sweep
that `docs/SOURCE_ONBOARDING.md` names as the principled route to W
was never built; W is currently inherited by habit.

This entry and the latent-width one below are linked; see the note
there on using small windows to run the width sweep cheaply.

## Models should be wide enough to make quantization the only bottleneck

*2026-08-26, open.*

Theoretically, non-linear layers can transform a small number of
dimensions into a much larger representation. However, in practice,
autoencoders reach much closer to the identity function when there is
no representational bottleneck. For this reason, RVQ in a larger
latent dimension has been better than FSQ in fewer dimensions (a
recollection from the research cycle; not yet recorded as a finding,
so treat as unverified here).

Hypothesis: for each source there is a critical value of effective
dimensions per pixel at the latent bottleneck. Below it, the
pre-quantization ceiling suffers; above it, the ceiling is stable and
performance is limited only by RVQ bitrate. Above the threshold is
where we want to operate, and the threshold itself should be
predictable from the source's intrinsic dimensionality. A per-pixel
PCA count understates that quantity for spatially redundant sources
(DEM has one band but near-total spatial predictability), so a
window-level intrinsic dimension is probably the right predictor.

Test: sweep latent width per tier on one source, read the ceiling from
validation, and locate the knee; then check whether the knee lands
where the source's measured intrinsic dimension predicts, across
sources.

Existing tools: the ceiling is already tracked in validation.
`analysis/pixel_rd.py` sweeps bottleneck dimension against rate with
the quantizer in the loop, `analysis/patch_intrinsic_dim.py` measures
window-level intrinsic structure, and `analysis/data_probe.py` gives
the per-tier PCA d95 used in `docs/SOURCE_ONBOARDING.md`, whose
derivation chain this would upgrade from rule of thumb to measured
threshold. Latent width is a config field, so the sweep is training
runs plus existing tooling.

Link to the window-size entry above: if window enlargement only
recovers coarse-scale redundancy, then per-pixel intrinsic
dimensionality is dominated by fine-scale content and barely depends
on window size. The critical threshold can therefore be measured on
very small windows, which train quickly, and should transfer to larger
windows as a slight overestimate. That makes the width sweep cheap,
and the size of the small-to-large discrepancy is itself a test of the
window-size hypothesis.

## Is there more information in the codebooks than we use?

*2026-08-27, open.*

The decoder currently gets the reconstructed latent vectors plus a depth thermometer
conditioning. This means that the internals of each codebook are invisible to the
decoder, only the final vector that comes out of N of them. What if the model also saw
the stream of codebook entries that were used to build that vector? The motivation for this
is that maybe the model can learn to understand more about the scene if it sees the path it
took through latent space to get to the point its at. These sequences would be large compared
to the current data at the bottleneck, is there a way to efficiently think about them in the
decoder? This would basically be taking the equivalent information used by a sequential decoding operation,
but doing it all-at-once in a transformer architecture.

## Per-band bounded Box-Cox as a general normalization stage

*2026-09-08, open. Posted for the maintainer after the q onboarding
analysis; approved in conversation, entry drafted by the agent.*

For ERA5 specific humidity, per-level standardization leaves heavy
tails (|z| to ~25, excess kurtosis 2.6) because the field is close to
log-normal through the mid-troposphere and close to linear at the
surface and in the stratosphere. Fitting one Box-Cox exponent per
level, bounded to [0,1] (log at 0, identity at 1) with minimum-|skew|
as the criterion, removes the skew at every tropospheric level and
cuts |z|max from 25 to ~2.3-5.3; levels that are already symmetric fit
lambda=1 and reduce to plain standardization. Analysis figures:
/data/ERA5-samples/analysis_2026-09-08/ (q_hist_meancorr, q_hist_log,
q_hist_boxcox).

The idea: make this a general per-band normalization stage available
to any source, not an ERA5 special. The fit degrades gracefully — a
symmetric band comes back with lambda 1 and nothing changes — so it
could sit in the standard onboarding path, with the fitted exponents
stored beside the mean/std moments as part of the model-space
definition and inverted exactly for decode and physical-unit metrics.
Candidate beneficiaries beyond q: any bounded-below or multiplicative
field (precipitation and w when they land in the store; S2 radiance
over dark scenes; DEM roughness derivatives).

The caveat to test, not assume: a transform chosen to make the
distribution Gaussian shrinks exactly the values that used to sit far
out in the tail, so the loss stops over-weighting them. Whether that
is a feature (spikes in training gradients disappear along with the
tail; ERA5 q grad-norm spikes of ~20 correlate with tail batches) or a
loss (the model stops caring about the physically extreme pixels —
which for humidity are the moist columns a user may care most about)
is a per-source judgement about what error metric should be uniform.
The transformed and untransformed q runs give the first measurement:
compare physical-unit RMSE on the top percentile of q pixels, not just
the pooled score, before generalizing.

Test: (1) finish the q A/B (era5_q64_ladder untransformed baseline vs
the transformed arm) and read both pooled and tail-conditional
physical error; (2) if the tail cost is acceptable or favourable,
lift the fit into the onboarding tooling (data_probe already computes
per-band moments) and re-derive one existing source's normalization
with it to check nothing regresses when lambda comes back ~1.

*Update 2026-09-08: the q A/B ran to ~15k steps on the ladder-w320
recipe (era5_q64_ladder vs era5_q64bc_ladder, jobs 243/244). Pooled
physical RMSE tracked within ~1% throughout, full-depth 1.599e-4 kg/kg
for both at 14-15k, and the untransformed arm showed none of the
gradient spikes seen under the attn recipe. The maintainer stopped the
Box-Cox arm: the extra moving parts are not worth the accuracy at this
recipe. Tail-conditional error was not measured; the entry stays open
only for the generalization question (w, tp, other bounded fields),
not for q.*

## Can we makes chains of compression?

*2026-08-29, open.*

Larger windows, in space, time, spectra and modality, mean a larger field over which redundancies and structure can be leveraged to compress data. However, two practical design choices make us resist the urge to make windows as large as possible. First, computationally, it is increasingly demanding to process the information, and train efficient models to do so. Second, users want chunks of data that can be decoded to meet their needs, and the larger the windows are, the more likely it is that data not relevant to the user is included, lowering the effective compression rate and decode speed for that user. To get around this, we should explore chains of compression. In a chain of compression, each successive step attacks a new dimension. This means that a user who only cares about single snapshots of a single band, can take the embeddings from the first step - spatial only. Then, a user who wants multitemporal windows might take the embeddings from a second stage, which encode small time series of the latents from spatial windows. Eventually, multimodal, multitemporal stacks could be compressed, leveraging the all available dimensions to find redundancy and compression. ERA5 is a great resource to test this, as we have 5 effective dimensions: 2 horizontal spatial, one height, one time, and one for the many variables. Theoretically a single blob could encode a chunk covering a volume in that full 5D space. Moreover, different users may benefit from different paths through the chain, e.g. one chain may start with spatial windows, for users who want snapshots, whilst another chain could start with time series of a single variable at a single location, for users who are interested in chunks of data like that. Finally, decoders could be built that either decode the full chunk, or effectively query the blob for a specific subset of values, which may lead to efficient decode for small subsections of a blob. This could be useful for users who mirror a full archive of blobs and need quick random access.
