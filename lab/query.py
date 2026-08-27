"""List and filter the findings ledger.

Examples:
    python lab/query.py                                # everything, one line each
    python lab/query.py --source tessera --keyword codebook
    python lab/query.py --status established --since 2026-08-01
    python lab/query.py --show codebook-sample-budget  # full record
"""
import argparse
from pathlib import Path

import yaml

FINDINGS = Path(__file__).parent / "findings"


def load(path):
    text = path.read_text()
    _, frontmatter, body = text.split("---", 2)
    rec = yaml.safe_load(frontmatter)
    rec["body"] = body.strip()
    rec["path"] = path
    return rec


def load_all():
    return [load(p) for p in sorted(FINDINGS.glob("*.md"))]


def matches(rec, args):
    if args.source and args.source not in rec["sources"] and "any" not in rec["sources"]:
        return False
    if args.status and rec["status"] != args.status:
        return False
    if args.keyword and not any(args.keyword in k for k in rec["keywords"]):
        return False
    date = str(rec["date"])
    if args.since and date < args.since:
        return False
    if args.until and date > args.until:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="records about this source (includes 'any')")
    ap.add_argument("--status", choices=["observed-once", "established", "superseded"])
    ap.add_argument("--keyword", help="substring match against keywords")
    ap.add_argument("--since", metavar="YYYY-MM-DD")
    ap.add_argument("--until", metavar="YYYY-MM-DD")
    ap.add_argument("--show", metavar="ID", help="print one full record")
    args = ap.parse_args()

    if args.show:
        print((FINDINGS / f"{args.show}.md").read_text())
        return

    recs = [r for r in load_all() if matches(r, args)]
    for r in recs:
        srcs = ",".join(r["sources"])
        print(f"{r['id']:40s} {r['date']} {r['status']:14s} [{srcs}]")
    print(f"({len(recs)} records)")


if __name__ == "__main__":
    main()
