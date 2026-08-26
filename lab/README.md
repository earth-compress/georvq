# lab

This directory is the machine-facing record of experimental work. It
exists so that agents working at different times can find what has
already been measured, in what context, and with what result. The
human-facing documentation in `docs/` states the conclusions that shaped
the design; this directory holds the evidence, including results that
led nowhere.

## Layout

- `TODO.md` holds the maintainer's queued hypotheses and experiments.
  Read it before proposing new experimental directions; take an item
  only when asked.
- `findings/` holds one file per empirical result.
- `experiments/` holds one directory per experiment worth keeping: the
  configs, the driver script, and anything else needed to re-run it. A
  finding links to its experiment when the experiment was kept.

## The findings format

```
---
id: short-kebab-slug          # matches the filename
date: 2026-08-19              # when the result was measured; if
                              #   approximate, say so in the body
commit: 988ca86               # the code state the experiment ran at
sources: [tessera]            # which data sources the result concerns;
                              #   [any] if source-independent
context: one or two sentences naming the checkpoint, dataset and scale
status: observed-once         # observed-once | established | superseded
superseded_by:                # finding id, when status is superseded
keywords: [codebook, ema]     # for retrieval by grep
experiment:                   # lab/experiments/<id>, if one was kept
related:
  - supports: some-other-id
  - clashes: another-id
---

Body: what was run, what was measured, and the numbers. State the
result no more generally than the experiment supports. A result from
one checkpoint on one source is a fact about that checkpoint and that
source. Confounds you know about go in the body; a finding that omits
a known confound is worse than no finding.
```

## Rules

- Nothing is filed here without the maintainer's approval. Prepare the
  record, present it, and file it once approved.
- Agents write `status: observed-once` only. Promotion to
  `established`, marking a record `superseded`, and deletion are the
  maintainer's calls.
- Do not edit another record's claims. When a new result bears on an
  old finding, add a `related` entry in both records (supports or
  clashes) and leave the old body alone.
- Numbers quoted in `docs/` must cite the finding id they come from.

## Searching

Records are plain text; retrieval is grep over keywords and ids, e.g.
`grep -rl codebook lab/findings/`.

## Imported records

The initial batch was imported from the old `docs/FINDINGS.md` in
August 2026. Those records carry `imported: true`, and their `date`
and `commit` are when the result was written down there, not when the
experiment ran. Most have no kept experiment.
