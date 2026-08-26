# Lab notebook

Ali's scratchpad for ideas, hunches, and things worth trying. Nothing here
is a commitment and nothing here has been verified — that is the point.
When an entry graduates into a measured result it moves to
`docs/FINDINGS.md` (with the number) or into code; when it turns out to be
wrong, say so and keep it, since a recorded dead end is worth more than a
silent one.

Loose conventions, ignore them when they get in the way:

- A hypothesis is more useful with a **falsifier** — what would you expect
  to see, and what would prove you wrong?
- Date entries so a stale idea is recognisable as stale.
- Link to the run, checkpoint, or file if one exists.

---

## Hypotheses

<!-- Testable claims. What you expect, and what would refute it. -->

### 2026-08-24 — the grown Tessera codec dies at ~27.5k steps

Both runs of the grown encoder+decoder collapsed at ~27–27.5k cumulative
steps: every rung of the ladder converges to the same value, so the output
stops depending on the codes. Halving the LR (5e-5 → 2.5e-5) delayed it by
nothing, which is what makes it interesting — an optimisation accident
should move when the step size moves. Gradients stayed finite, clipping
was active at 1.0, GPU was clean.

Guess: the identity-initialised added blocks drift into a degenerate
solution. Falsifier: per-block activation and gradient norms resumed from
`checkpoints/tessera_grown_full/best_verified_27k.pt` should show the new
blocks diverging from the pretrained ones before the metrics move. If all
blocks look alike right up to the cliff, the cause is elsewhere.

See the `tessera-grown-collapse` memory for the full account.

### 2026-08-24 — capacity only pays above a rate floor

Growing the codec improved every rung that engages tier 1 (half 97.5%,
full 97.4% of release ccos distance) and did nothing for the tier-0-only
rungs (`min 4-0-0`, `coarse 20-0-0` sat at ~100% across 30+ validations in
both runs). Reading: the coarse rungs are rate-limited, not
capacity-limited — there isn't enough information in 20 tier-0 codes for a
bigger network to exploit.

Falsifier: sweep tier-1 depth (20-20-1 … 20-20-20) on
`best_verified_27k.pt`. If it's rate limiting, the gain should switch on
with the first tier-1 code and grow with depth. If the gain appears fully
formed at 20-20-1, something about tier-0 training dynamics is the cause
instead. The R-D scatter already sweeps all 57 operating points, so this
is a checkpoint eval, not a training run.

If this holds it matters commercially: the cheap operating points that
make bulk download attractive are exactly the ones a bigger model does not
help.

---

## Experiments to run

<!-- Concrete, scoped, someone could pick it up cold. -->

---

## Features / tooling

<!-- Things the repo should be able to do but can't yet. -->

### 2026-08-24 — collapse detector in the training loop

Two runs died and burned 2000 and 500 steps respectively before anyone
noticed. The signature is cheap to test for: if the spread across rungs at
a validation collapses toward zero, the codes have stopped mattering.
Worth an automatic abort (or at least a loud warning) rather than reading
it off a log hours later.

Related: the val/save interval trade-off. `val_interval: 1000` +
`save_interval: 2500` meant the newest periodic checkpoint was itself
already inside the decay. 500/1000 on long warm-start runs.

---

## Questions / unknowns

<!-- Things we don't understand yet. Not necessarily actionable. -->

### 2026-08-24 — why is ceiling cosine so unstable?

Ceiling `1-cos` is ~0.0037, the smallest denominator in the report, so its
relative figure swung 24%–3540% across validations while ceiling `ccos`
moved smoothly and monotonically. Fine as an explanation of the variance,
but it doesn't explain why the *absolute* ceiling cosine wanders by ±0.002
between validations on a fixed val set with a deterministic model. Batch
composition in the ceiling path? Worth a look before quoting ceiling
numbers anywhere.

---

## Done / graduated

<!-- Moved to FINDINGS, shipped, or refuted. Keep the refutations. -->

### 2026-08-24 — REFUTED: codebook resets caused the first collapse

Blamed the reset at step 27750 for the first collapse, since it landed 250
steps before the bad validation. Scoring the step-27500 checkpoint showed
it was *already* degraded (full ccos 0.8173 vs 0.8195 at step 27000) —
the decay started before the reset. Resets are exonerated.
