"""Compare the moment-coder arms at MATCHED distortion.

Reading rate-distortion tables by eye is unreliable: the arms were run at
common quantizer steps, not common errors, so every comparison is really
an interpolation and the answer depends on how you do it. For a Gaussian
source R = 0.5 log2(sigma^2/D), i.e. bits are linear in log(error), so
that is the parametrization interpolated in here -- linear-in-bits
interpolation on a convex curve overstates whichever arm has the sparser
sampling.

Consumes the JSONs written by moment_rd.py (arm = pure autoencoder) and
moment_hybrid.py (arms = stage1-only / hybrid / linear-only).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os


def bits_at(points, target):
    """Interpolate bits needed for `target` error. points: [(bits, err)]."""
    pts = sorted(points, key=lambda p: p[1])          # ascending error
    lt = math.log(target)
    for (b0, e0), (b1, e1) in zip(pts, pts[1:]):
        if e0 <= target <= e1:
            f = (lt - math.log(e0)) / (math.log(e1) - math.log(e0))
            return b0 + f * (b1 - b0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rd", default="moment_rd.json")
    ap.add_argument("--targets", default="0.25,0.20,0.15,0.12,0.07,0.02")
    args = ap.parse_args()

    arms = {}

    rd = os.path.join(args.dir, args.rd)
    if os.path.exists(rd):
        rows = json.load(open(rd))
        # the pure-AE frontier: best dim at each rate
        best = {}
        for r in rows:
            if not r["bits"]:
                continue
            k = r["bits"]
            if k not in best or r["all_rel"] < best[k][1]:
                best[k] = (r["bits"], r["all_rel"])
        arms["AE only (best dim)"] = list(best.values())

    for p in sorted(glob.glob(os.path.join(args.dir, "hybrid_d*_b*.json"))):
        rows = json.load(open(p))
        tag = os.path.basename(p)[len("hybrid_"):-len(".json")]
        h = [(r["bits"], r["all_rel"]) for r in rows if r["arm"] == "hybrid"]
        s1 = [(r["bits"], r["all_rel"]) for r in rows
              if r["arm"] == "stage1 only"]
        arms[f"hybrid {tag}"] = h + s1
        lin = [(r["bits"], r["all_rel"]) for r in rows
               if r["arm"] == "linear only"]
        if lin:
            arms["linear only"] = lin

    targets = [float(v) for v in args.targets.split(",")]
    names = list(arms)
    w = max(len(n) for n in names) + 1
    print(f"{'arm':<{w}} " + " ".join(f"{'@'+str(t):>9}" for t in targets))
    print(f"{'':<{w}} " + " ".join(f"{'bits':>9}" for _ in targets))
    table = {}
    for n in names:
        vals = [bits_at(arms[n], t) for t in targets]
        table[n] = vals
        print(f"{n:<{w}} " + " ".join(
            f"{v:>9.0f}" if v is not None else f"{'-':>9}" for v in vals))

    print(f"\nbest arm per distortion, and its margin over the runner-up:")
    for i, t in enumerate(targets):
        cand = [(n, table[n][i]) for n in names if table[n][i] is not None]
        if len(cand) < 2:
            continue
        cand.sort(key=lambda c: c[1])
        (n0, b0), (n1, b1) = cand[0], cand[1]
        print(f"  err {t:<6} {n0:<28} {b0:>7.0f} bits  "
              f"({100*(b1-b0)/b1:>5.1f}% under {n1})")


if __name__ == "__main__":
    main()
