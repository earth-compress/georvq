"""Measured figures live in one place, so they cannot drift apart.

A result quoted in nine files will eventually disagree with itself:
someone refines the measurement, updates the file in front of them, and
the other eight go stale. The convention is that a measured figure
appears in its findings record (`lab/findings/`, canonical), plus at
most one mechanism home, the module or table whose subject it is.
Everywhere else states the claim and links the record, the way
docs/DESIGN.md does.

This test pins the convention by location rather than by parsing prose:
each figure lists the files allowed to contain its digits. A new
quotation fails here, which is the moment to ask whether the reader
needs the number or just the claim and a link.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SEARCH = ["georvq", "docs", "lab", "configs", "scripts", "README.md", "CLAUDE.md"]
EXCLUDE = {Path(__file__).name}

#: literal figure -> the files permitted to state it
FIGURES = {
    # forcing the std hurts: 0.9700 -> 0.9678
    "0.9678": {
        "lab/findings/stats-channel-is-per-field.md",   # canonical
        "docs/SOURCE_ONBOARDING.md",        # the five-source measurement table
        "georvq/model/stats_cond.py",       # the mechanism's own rationale
    },
    # what the transmitted mean buys
    "0.0069": {
        "lab/findings/stats-channel-is-per-field.md",
        "docs/SOURCE_ONBOARDING.md",
    },
    # the deepest fine levels train on a fraction of a percent of rows
    "P(active)=0.006": {
        "lab/findings/depth-sampling-along-path.md",    # canonical
        "georvq/model/codec.py",            # `path_tuples`, the mechanism
        "configs/tessera/tessera_vq8_k256.yaml",  # that run's own provenance
    },
    # the sample-budget crossover
    "450 vectors per codeword": {
        "lab/findings/codebook-sample-budget.md",
        "docs/SOURCE_ONBOARDING.md",
    },
}


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
        + "\n  ".join(strays))
    assert not missing, (
        f"{figure!r} has vanished from {missing}; update FIGURES if that is deliberate")
