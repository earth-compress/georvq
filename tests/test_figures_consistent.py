"""Measured figures live in one place, so they cannot drift apart.

A result quoted in nine files is a result that will eventually disagree
with itself: someone refines the measurement, updates the file in front
of them, and the other eight quietly go stale. The convention is:

    a measured figure appears in docs/FINDINGS.md (canonical), plus at
    most ONE mechanism home -- the module or table whose subject it is --
    and everywhere else states the qualitative claim and cites.

This test pins that convention by location rather than by parsing prose:
each figure lists the files allowed to contain its digits. A new
quotation fails here, which is the moment to ask whether the reader
really needs the number or just the claim and a pointer.

Excluded from the search: docs/MERGE_TRIAGE.md, a transient triage of
this branch that is deleted at merge, and this file.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SEARCH = ["georvq", "docs", "configs", "scripts", "README.md", "CLAUDE.md"]
EXCLUDE = {"MERGE_TRIAGE.md", Path(__file__).name}

#: literal figure -> the files permitted to state it, and why each is a home
FIGURES = {
    # forcing the std hurts: 0.9700 -> 0.9678
    "0.9678": {
        "docs/FINDINGS.md",                 # canonical
        "docs/SOURCE_ONBOARDING.md",        # the five-source measurement table
        "georvq/model/stats_cond.py",       # the mechanism's own rationale
    },
    # what the transmitted mean buys
    "0.0069": {
        "docs/FINDINGS.md",
        "docs/SOURCE_ONBOARDING.md",
    },
    # the deepest fine levels train on a fraction of a percent of rows
    "P(active)=0.006": {
        "docs/FINDINGS.md",                 # canonical
        "georvq/model/codec.py",            # `path_tuples`, the mechanism
        "configs/tessera_vq8_k256.yaml",    # that run's own provenance
    },
    # the sample-budget crossover
    "450 vectors per codeword": {
        "docs/FINDINGS.md",
        "docs/SOURCE_ONBOARDING.md",
    },
    # PCA-packed S2 stats channel: mean bytes/window at tolerance 0.01
    "9.73": {
        "docs/FINDINGS.md",                 # canonical
        "configs/s2_pyr96.yaml",            # the run that charges it
    },
    # how badly an uncharged normalizer channel understates the coarse rung
    "3.31": {
        "docs/FINDINGS.md",
    },
    # the batch/throughput knee on the S2 pyramid (windows/s)
    "628": {
        "docs/FINDINGS.md",
        "configs/s2_pyr96.yaml",            # why that batch was chosen
    },
    "634": {
        "docs/FINDINGS.md",
        "configs/s2_pyr96.yaml",
    },
}

# Deliberately NOT pinned: short decimals such as 4.7, 12.1 and 5.3, which
# occur incidentally as unrelated constants elsewhere in the tree. This
# test matches literal substrings, so a figure is only a usable key when
# its digits are distinctive enough not to collide.


def _files():
    for name in SEARCH:
        p = ROOT / name
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in p.rglob("*"):
                if (f.is_file() and "__pycache__" not in f.parts
                        and f.name not in EXCLUDE
                        and f.suffix in {".py", ".md", ".yaml", ".yml"}):
                    yield f


@pytest.mark.parametrize("figure", sorted(FIGURES))
def test_figure_stays_in_its_home(figure):
    homes = FIGURES[figure]
    found = set()
    for f in _files():
        try:
            if figure in f.read_text():
                found.add(str(f.relative_to(ROOT)))
        except (UnicodeDecodeError, OSError):
            continue
    strays = sorted(found - homes)
    missing = sorted(homes - found)
    assert not strays, (
        f"{figure!r} is quoted outside its home(s) {sorted(homes)}:\n  "
        + "\n  ".join(strays) +
        "\n\nState the claim and cite docs/FINDINGS.md instead, or add the "
        "file to this figure's homes if it genuinely needs the number."
    )
    assert not missing, (
        f"{figure!r} has vanished from {missing}. If the measurement was "
        "superseded, update every home and this registry together; if it "
        "merely moved, update the registry."
    )
