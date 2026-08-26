# georvq

georvq trains pyramidal residual vector quantization codecs for gridded
geospatial data. It is the first prototype in a wider project on the
compression of earth observation data, which will in time cover classical
and neural methods, lossy and lossless. The main output of this
repository is the method and what has been learned while developing it.
Trained codecs are released with fixed versions and their contracts are
kept, but the code around them keeps moving.

## Getting oriented

Read `docs/OVERVIEW.md` before making changes. It has the module map,
the rules that protect released codecs, and practical notes on running
training and evaluation.

The documentation is split in two:

- `docs/` is written for people. It is short, and its wording is
  controlled by the maintainer. If a change needs new text in `docs/`,
  `README.md` or the notebooks, write a draft and ask for review; do not
  settle the final wording yourself.
- `lab/` is the working record for agents. Empirical results go there as
  findings records, each carrying its date, git commit, source, and the
  context of the experiment, with links to the experiment that produced
  it and to related findings. The format is described in `lab/README.md`.
  State a finding no more generally than the experiment supports: a
  result from one checkpoint on one source is a fact about that
  checkpoint and that source.

## Rules that protect released codecs

- Never edit the numerics in `georvq/model/vq.py`. Trained checkpoints
  must decode bit-for-bit forever, and `tests/test_compat.py` checks
  this against released weights. It must stay exact, not approximately
  equal.
- State dict names are part of the checkpoint format. Renaming modules
  or parameters breaks every existing checkpoint. If a rename is
  unavoidable, extend the rename table in `georvq/compat.py` and show
  that a legacy checkpoint still loads strictly.
- The bytes of shipped payloads are frozen. The AlphaEarth `star.tif`
  bytes are public and pinned by `tests/test_refine.py`. Generalising
  the code behind a shipped format is fine; changing its output bytes is
  not.
- `georvq/ae/` and `georvq/tessera/` hold source-specific release
  policy (`docs/AE_RELEASE.md`, `docs/TESSERA_RELEASE.md`);
  `georvq/inference/` stays source-agnostic. Keep that boundary.

## Verifying changes

Run `pytest -m "not slow and not gpu"` (about 20 seconds) on every
change. For anything touching training, prefer `configs/smoke.yaml`, or
a few hundred steps with overrides, over a long run. Configs are
validated by pydantic with `extra="forbid"`; anything that affects a run
belongs in the `Config` schema, not in an environment variable or an
edit to the source.

## Sources

Sources are at different stages of maturity, tracked in
`docs/SOURCES.md`. The repository does not rank them. Keep shared code
source-agnostic and put source-specific behaviour in the adapter
(`georvq/sources/`) or in a release layer. Before designing for a new
source, read `docs/SOURCE_ONBOARDING.md`: window size, tier grids,
latent widths and codebook sizes are derived from measurements of the
source, not copied from an existing preset.

## What to commit

Everything committed has an upkeep cost, so most exploratory work should
not be committed.

Commit:

- changes to the library, with tests;
- release layers, and anything a test depends on;
- analysis tools that answer a question likely to recur across sources;
- experiment records under `lab/` (config, driver script, finding) when
  the result is worth keeping.

Do not commit:

- one-off probe scripts, which do not go in `scripts/` or `analysis/`.
  If a probe measured something about a source, a checkpoint or the
  training process, file the script with its config and result under
  `lab/`, marked as observed once; if it measured nothing, leave it in
  an untracked scratch directory. The judgement of importance is not
  yours to make, in either direction: promoting a result into `docs/`
  and deleting a recorded finding or its script are both the
  maintainer's calls, made during periodic clean-ups using the ledger's
  dates, statuses and cross references.
- figures, PDFs, archives and other outputs: they belong in run
  directories or untracked folders, never at the repository root;
- checkpoints and datasets;
- sweep configs with no finding attached. `configs/` holds only the
  presets that reproduce a release, presets for sources in active
  development, and `smoke.yaml`; configs that exist to back a finding
  live with their experiment under `lab/`.

At the end of a piece of work, list what you produced, say what you
propose to commit and what to discard, and ask if the split is not
obvious. The default for exploratory code is untracked.
