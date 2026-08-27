"""The findings ledger stays parseable and internally consistent.

Every record needs the fields the query tool and the clean-up process
rely on, and every cross-reference must point at a record that exists.
"""
from pathlib import Path

import pytest
import yaml

FINDINGS = Path(__file__).resolve().parent.parent / "lab" / "findings"
RECORDS = sorted(FINDINGS.glob("*.md"))
IDS = {p.stem for p in RECORDS}

REQUIRED = ["id", "date", "commit", "sources", "context", "status", "keywords"]
STATUSES = {"observed-once", "established", "superseded"}
RELATION_KINDS = {"supports", "clashes"}


def frontmatter(path):
    parts = path.read_text().split("---", 2)
    assert len(parts) == 3, f"{path.name}: no frontmatter block"
    return yaml.safe_load(parts[1])


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_is_valid(path):
    rec = frontmatter(path)
    for field in REQUIRED:
        assert field in rec and rec[field] not in (None, "", []), (
            f"{path.name}: missing {field}")
    assert rec["id"] == path.stem, f"{path.name}: id does not match filename"
    assert rec["status"] in STATUSES
    assert isinstance(rec["sources"], list)
    assert isinstance(rec["keywords"], list)
    if rec["status"] == "superseded":
        assert rec.get("superseded_by") in IDS, (
            f"{path.name}: superseded without a valid superseded_by")
    for entry in rec.get("related") or []:
        (kind, target), = entry.items()
        assert kind in RELATION_KINDS, f"{path.name}: bad relation kind {kind!r}"
        assert target in IDS, f"{path.name}: related id {target!r} does not exist"


def test_ledger_not_empty():
    assert RECORDS
