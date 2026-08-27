# georvq

georvq trains pyramidal residual vector quantization codecs for gridded
geospatial data. The codebase uses different sources of earth data to 
stress-test the developed methodology, and new techniques are explored
to improve performance on those sources. When something works for one
source, that doesn't mean it will transfer well to another. The longer
term goal for this project is to understand and predict what kind of
architecture would best suit a new source, based on its statistics and
structure. As we go, we make release versions for different sources, 
though these may end up getting amended or improved as new features are
developed. Bear in mind during your work that you are likely one of many
agents who have each been given a specific task or focus, and so the 
results and findings of your work may not be that important in the grand
scheme of things. Unless you are explicitly asked, you should not alter
central parts of the repo which affect the whole project. Rather, if you
believe you have found something that is of general importance, you should
use the findings system, described later on, to log it. If you think
this finding is of such importance that it fundamentally challenges or
changes something about the project, that is to be discussed with the
maintainer.

## Style guide

The maintainer of this repo is not a fan of many of the classic claude-isms
which agents typically use a crutch in their prose, code comments, docs,
and thinking. For example, some of the cliches the maintainer hates:

 - Using irrelevant adverbs like "quietly" and "genuinely" to add an
   unnecessary air of importance of emotion to a sentence.
 - Breaking up thoughts into small sentences for dramatic effect.
 - Sounding like a salesman or start-up founder pitching their big idea.
 - Any phrase or word like "load-bearing".
 - Pithy few-word simplifications for concepts that can be explained more 
 clearly e.g. "The machinery remains for codecs where it pays" is better
 phrased as "The machinery remains for codecs which benefit from using it"
 - Though not banned, over-use of colons, semicolons and em-dashes is strongly
 discouraged.
 - Framing a statement as the negation of a claim nobody made, e.g. "It's
 not X, it's Y" or "This isn't just X". Say what the thing is.
 - Inflating the stakes with definite articles and superlatives, e.g. "the
 key insight", "the fundamental tension", "the whole point". Most results
 are not the anything, they are a result.
 - Stock metaphors such as "plumbing", "wiring", "spine", "north star",
 "landmine", "graveyard", "the story". Describe the situation directly.
 - Quality adjectives that carry no information, e.g. "robust", "clean",
 "elegant", "comprehensive", "principled", "seamless". If something is
 good, say what it does well.
 - Filler sentence openers, e.g. "Note that", "It's worth noting",
 "Crucially", "Interestingly". If the point matters the sentence should
 show it, and if it doesn't the sentence should go.
 - Restating a point at the end of a paragraph with more conviction, e.g.
 "In short, ..." or "The upshot is ...". Make the statement once.
 - Forcing material into groups of three (three examples, three clauses,
 three adjectives (notice the irony here?)) whether or not there are 
 naturally three of anything.
 - Describing code as if it had feelings or intentions, e.g. "the decoder
 wants", "this tier doesn't care". Occasionally useful in teaching,
 but agents use it far too often.
 - Bolding phrases mid-sentence to tell the reader what to feel, and
 splitting short documents into many headed sections.
 - Closing a document with a sentence whose only job is to sound finished,
 e.g. "This sets the stage for ..." or "The path forward is clear".
 - In code comments: narrating the change rather than the code ("now
 correctly handles the empty case"), advertising ("fast path"), or
 prefixing with "Note:". A comment states a constraint the code cannot
 show, in the same flat voice as the comments around it.

Some of the worst offenders can be caught mechanically. These words and
phrases are banned outright in docs and prose: sacred, load-bearing,
spine, quietly, genuinely, north star, landmine, graveyard, plumbing,
wiring, crown jewel, battle-tested, first-class citizen, "the story",
"in short", "the upshot", "worth noting", "crucially", "interestingly".
Sweep for them before presenting a draft:

    grep -rniE 'sacred|load-bearing|spine|quietly|genuinely|north star|landmine|graveyard|plumbing|wiring|crown jewel|battle-tested|first-class citizen|the story|in short|the upshot|worth noting|crucially|interestingly' docs/ README.md

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
  Nothing enters the ledger without the maintainer's approval: prepare
  the record, present it, and file it only once approved. State a
  finding no more generally than the experiment supports: a result from
  one checkpoint on one source is a fact about that checkpoint and that
  source.

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
  training process, prepare a findings record and present it to the
  maintainer; only if it is approved do the record, the script and its
  config get filed under `lab/`. Until then, and if the answer is no,
  the probe stays in an untracked scratch directory. Promoting a result
  into `docs/` and deleting a recorded finding or its script are
  likewise the maintainer's calls.
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
