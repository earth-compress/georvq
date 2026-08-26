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
