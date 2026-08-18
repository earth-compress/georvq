"""Recover the ad-hoc analyses that only ever existed as shell one-liners.

A long investigation leaves most of its diagnostics in shell history: the
grad-norm outlier check, the loss-spike attribution, the cross-run wandb
comparison. Each caught a real defect; none was ever committed. This pulls
them out of a session transcript so they can be triaged and promoted into
`analysis/`.

Deliberately mechanical: it classifies and DEDUPES, but does not try to
judge relevance -- that is left to a reader (human or agent) working from
the compact file this writes, so the 27 MB transcript never has to be read
directly.

Classification:
  patch    edits a source file (open(...,'w'), s.replace) -- not a tool
  trivial  ls / git / kill / nvidia-smi / plain tail|grep -- not a tool
  analysis everything else: computes, measures, compares

Dedup is on a NORMALISED body (env prefixes, paths and step numbers
stripped), so the twenty variants of "tail the training log" collapse to
one entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

PREFIXES = [
    r"source /home/ali/miniconda3/etc/profile\.d/conda\.sh\s*&&\s*",
    r"conda activate \S+\s*&&\s*",
    r"PYTHONPATH=\$PWD\s*",
    r"WANDB_MODE=disabled\s*",
    r"cd /home/ali/\S+\s*&&\s*",
    r"timeout \d+\s+",
]
TRIVIAL = re.compile(
    r"^\s*(ls|ll|git (log|status|add|commit|push|diff)|kill|pkill|pgrep|"
    r"nvidia-smi|free|df|mkdir|cp|mv|rm|cat|echo|which|ps|sleep|chmod|"
    r"stat|date|dmesg|journalctl|tail|head|wc)\b")
PATCHY = ("open(p,'w').write", 'open(p, "w").write', "s.replace(",
          "ast.parse(s)")


def strip(cmd: str) -> str:
    out = cmd
    for p in PREFIXES:
        out = re.sub(p, "", out)
    return out.strip()


def normalise(cmd: str) -> str:
    """Collapse variants that differ only in paths, steps or thresholds."""
    n = strip(cmd)
    n = re.sub(r"\d+", "N", n)
    n = re.sub(r"/\S+", "PATH", n)
    n = re.sub(r"\s+", " ", n)
    return n


def classify(cmd: str) -> str:
    s = strip(cmd)
    if any(p in s for p in PATCHY):
        return "patch"
    body = s.split("&&")[-1].strip()
    if TRIVIAL.match(body) and "python" not in s:
        return "trivial"
    if "python" in s or "wandb" in s or "awk" in s or "torch" in s:
        return "analysis"
    return "trivial"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heredoc-only", action="store_true",
                    help="keep only multi-line heredoc analyses. One-line "
                         "`python -c` checks are introspection, not tools, "
                         "and they are 2/3 of the volume.")
    ap.add_argument("--out-lines", type=int, default=8,
                    help="lines of each command's OUTPUT to keep, so a "
                         "reader can tell what the analysis produced")
    args = ap.parse_args()

    calls, results = [], {}
    for line in open(args.transcript):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        msg = rec.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "tool_use" and c.get("name") == "Bash":
                calls.append((c.get("id"), (c.get("input") or {})))
            elif c.get("type") == "tool_result":
                r = c.get("content")
                if isinstance(r, list):
                    r = " ".join(x.get("text", "") for x in r
                                 if isinstance(x, dict))
                results[c.get("tool_use_id")] = r if isinstance(r, str) else ""

    seen, kept = {}, []
    counts = {"analysis": 0, "patch": 0, "trivial": 0, "dup": 0}
    for cid, inp in calls:
        cmd = inp.get("command", "")
        if not cmd:
            continue
        kind = classify(cmd)
        counts[kind] += 1
        if kind != "analysis":
            continue
        if args.heredoc_only and "python - <<" not in cmd:
            continue
        key = hashlib.md5(normalise(cmd).encode()).hexdigest()[:10]
        if key in seen:
            counts["dup"] += 1
            seen[key][1] += 1
            continue
        entry = [len(kept) + 1, 1, inp.get("description", ""), strip(cmd),
                 (results.get(cid) or "")]
        seen[key] = entry
        kept.append(entry)

    with open(args.out, "w") as f:
        f.write(f"# {len(kept)} unique ad-hoc analyses recovered from "
                f"{args.transcript}\n")
        f.write("# Each: ID, how many times a near-identical command ran, "
                "the description,\n# the command, and the first lines of "
                "its output.\n\n")
        for idx, n, desc, cmd, out in kept:
            f.write(f"{'='*72}\n[{idx}] x{n}  {desc}\n{'-'*72}\n{cmd}\n")
            lines = [l for l in (out or "").splitlines() if l.strip()]
            if lines:
                f.write("--- output (truncated) ---\n")
                f.write("\n".join(lines[:args.out_lines]) + "\n")
            f.write("\n")
    print(f"analysis={counts['analysis']} (unique {len(kept)}, "
          f"dup {counts['dup']})  patch={counts['patch']}  "
          f"trivial={counts['trivial']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
