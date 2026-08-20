"""Compare training runs from wandb: leaderboard, trace, rates, outliers.

Formalises the single most-repeated ad-hoc analysis of the Tessera
investigation -- 25 separate inline queries that were really five
questions. Each mode below replaces a family of one-liners.

WHY THE `rates` MODE EXISTS. On a bounded metric, an absolute delta is
misleading: 0.10 -> 0.20 and 0.80 -> 0.90 are the same +0.10 and are not
remotely the same achievement. `rates` reports the fraction of the
REMAINING gap closed per 1k steps, d(1-m)/(1-m), which is the honest unit
and which exposed a depth-graded improvement (coarse 20%, mid 9%, full
0.9%) that the absolute numbers flattened into "only coarse is moving".

WHY `--runs name+offset` EXISTS. A run interrupted and resumed appears in
wandb as several runs whose step counters all restart at zero. Comparing
against a baseline then silently compares step 3000 of a resume with step
3000 of a cold start. Offsets stitch the segments back into one timeline.

CAVEAT WORTH KEEPING IN MIND. `api.runs()` returns run objects whose
`.config` is not always populated; filtering on it silently matches
nothing. `leaderboard` therefore re-fetches each run by id, which is
slower but correct -- a bug that cost an hour when a filter for
window==64 returned zero of 89 runs.
"""
from __future__ import annotations

import argparse
import math

DEFAULT_PROJECT = "alifrancis/georvq"
# the metrics worth looking at by default on this codec: quantized full
# depth, the unquantized ceiling, and their mean-removed counterparts
DEFAULT_KEYS = ["val/cos_full", "val/ccos_full", "val/cos_ceiling",
                "val/ccos_ceiling"]


def parse_runs(spec):
    """'name' or 'name+19000' -> (name, step_offset)."""
    out = []
    for s in spec.split(","):
        s = s.strip()
        if not s:
            continue
        if "+" in s:
            n, off = s.rsplit("+", 1)
            out.append((n, int(off)))
        else:
            out.append((s, 0))
    return out


def history(run, keys, samples=800):
    h = run.history(keys=keys, samples=samples, pandas=False)
    return sorted([x for x in h if any(x.get(k) is not None for k in keys)],
                  key=lambda x: x["_step"])


def resolve(api, project, wanted):
    """(name, offset) -> list of (offset, run), re-fetched so config is real."""
    index = {}
    for r in api.runs(project):
        index.setdefault(r.name, []).append(r.id)
    out = []
    for name, off in wanted:
        ids = index.get(name)
        if not ids:
            # fall back to treating it as a run ID. Necessary in practice:
            # a resumed experiment reuses its run_tag, so several runs share
            # one name and only the id identifies a segment.
            try:
                out.append((off, api.run(f"{project}/{name}")))
            except Exception:
                print(f"  ! no run named or with id {name}")
            continue
        if len(ids) > 1:
            print(f"  ! {len(ids)} runs named {name}; using all. Pass ids to "
                  f"stitch segments individually: {','.join(ids)}")
        for rid in ids:
            out.append((off, api.run(f"{project}/{rid}")))
    return out


def cmd_trace(api, args):
    keys = args.keys.split(",")
    print(f"{'cum step':>9} " + " ".join(f"{k.split('/')[-1]:>12}" for k in keys)
          + "   run")
    for off, r in resolve(api, args.project, parse_runs(args.runs)):
        for x in history(r, keys):
            if (x["_step"] + off) % args.every:
                continue
            print(f"{x['_step']+off:>9} " + " ".join(
                f"{x.get(k, float('nan')):>12.4f}" for k in keys)
                + f"   {r.name}")


def cmd_rates(api, args):
    """Relative gap closed per 1k steps -- the right unit for bounded metrics."""
    keys = args.keys.split(",")
    series = []
    for off, r in resolve(api, args.project, parse_runs(args.runs)):
        for x in history(r, keys):
            series.append((x["_step"] + off, x))
    series.sort(key=lambda t: t[0])   # ties would compare the dicts
    print(f"{'cum step':>9} " + " ".join(f"{k.split('/')[-1]:>11}" for k in keys)
          + "  |  relative gap closed %/1k steps")
    prev = None
    for st, x in series:
        vals = [x.get(k) for k in keys]
        line = f"{st:>9} " + " ".join(
            f"{(v if v is not None else float('nan')):>11.4f}" for v in vals)
        if prev and st > prev[0]:
            n = (st - prev[0]) / 1000.0
            parts = []
            for v, pv in zip(vals, prev[1]):
                if v is None or pv is None or pv >= 1.0:
                    parts.append(f"{'-':>8}")
                else:
                    parts.append(f"{100*(1-(1-v)/(1-pv))/n:>8.2f}")
            line += "  |" + " ".join(parts)
        print(line)
        prev = (st, vals)


def cmd_leaderboard(api, args):
    """Best value of a metric per run, with the config columns that explain it."""
    filt = dict(kv.split("=", 1) for kv in args.filter.split(",") if kv)
    cols = [c for c in args.config_cols.split(",") if c]
    rows = []
    for r in api.runs(args.project):
        if args.name_contains and args.name_contains not in (r.name or ""):
            continue
        try:
            full = api.run(f"{args.project}/{r.id}")      # config is real here
            c = full.config or {}
            ok = True
            for k, v in filt.items():
                cur = c
                for part in k.split("."):
                    cur = (cur or {}).get(part) if isinstance(cur, dict) else None
                if str(cur) != v:
                    ok = False
                    break
            if not ok:
                continue
            h = history(full, [args.metric])
            if not h:
                continue
            best = max(x[args.metric] for x in h if x.get(args.metric))
            last = max(x["_step"] for x in h)
            vals = []
            for col in cols:
                cur = c
                for part in col.split("."):
                    cur = (cur or {}).get(part) if isinstance(cur, dict) else None
                vals.append(str(cur))
            rows.append((best, last, r.name, vals))
        except Exception:
            continue
    rows.sort(reverse=True)
    print(f"{args.metric:>22} {'steps':>8}  " +
          " ".join(f"{c.split('.')[-1][:10]:>11}" for c in cols) + "  run")
    for best, last, name, vals in rows[:args.top]:
        print(f"{best:>22.4f} {last:>8}  " +
              " ".join(f"{v[:11]:>11}" for v in vals) + f"  {name}")


def cmd_outliers(api, args):
    """Spikes against a LOCAL median -- a global threshold cannot separate a
    spike from the steep early descent of a loss curve."""
    import numpy as np
    for off, r in resolve(api, args.project, parse_runs(args.runs)):
        h = history(r, [args.metric])
        s = np.array([x[args.metric] for x in h if x.get(args.metric)])
        st = np.array([x["_step"] + off for x in h if x.get(args.metric)])
        if len(s) < 2 * args.window + 1:
            print(f"  {r.name}: too few points")
            continue
        w = args.window
        med = np.array([np.median(s[max(0, i-w):i+w+1]) for i in range(len(s))])
        ratio = s / np.maximum(med, 1e-12)
        print(f"\n{r.name}: {len(s)} points, median {np.median(s):.5f}, "
              f"max/local-median {ratio.max():.2f}x")
        for i in np.argsort(-ratio)[:args.top]:
            if ratio[i] < args.thresh:
                continue
            print(f"  step {st[i]:>7}: {args.metric}={s[i]:.5f}  "
                  f"local median {med[i]:.5f}  ratio {ratio[i]:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["trace", "rates", "leaderboard",
                                     "outliers"])
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--runs", default="",
                    help="comma list; 'name' or 'name+OFFSET' to stitch a "
                         "resumed run onto a shared timeline")
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    ap.add_argument("--metric", default="val/ccos_ceiling")
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--filter", default="",
                    help="config equality filter, e.g. model.window=64")
    ap.add_argument("--name-contains", default="")
    ap.add_argument("--config-cols", default="model.window,train.batch_windows")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--window", type=int, default=9,
                    help="outliers: half-width of the local median")
    ap.add_argument("--thresh", type=float, default=1.25)
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    {"trace": cmd_trace, "rates": cmd_rates, "leaderboard": cmd_leaderboard,
     "outliers": cmd_outliers}[args.mode](api, args)


if __name__ == "__main__":
    main()
