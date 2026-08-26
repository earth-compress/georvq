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
