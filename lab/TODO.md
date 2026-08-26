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
