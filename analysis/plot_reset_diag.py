"""Render the per-level reset diagnostic figure from a run's wandb history.

For runs launched before `figs/reset` existed: they log the per-level
reset/* scalars, which wandb renders as ~168 unreadable single-value
panels. This pulls them back down and draws the figure that should have
been logged in the first place -- one page per reset event.

  python analysis/plot_reset_diag.py tessera_w32hd_e2e_tree --project <entity>/georvq
  python analysis/plot_reset_diag.py <run_id> --project <entity>/georvq --out reports/
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from georvq.config import load_config
from georvq.train.figures import reset_figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="wandb run id, or the run_tag/display name")
    ap.add_argument("--project", required=True,
                    help="wandb <entity>/<project> the runs were logged to")
    ap.add_argument("--config", default=None,
                    help="config yaml for tier specs (default: infer from "
                         "the run's own stored config)")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    try:
        run = api.run(f"{args.project}/{args.run}")
    except Exception:                       # not an id -- match by name
        cands = [r for r in api.runs(args.project)
                 if r.name == args.run]
        if not cands:
            raise SystemExit(f"no run named or id'd {args.run!r}")
        run = sorted(cands, key=lambda r: r.created_at)[-1]
    print(f"run {run.name} ({run.id}), state={run.state}")

    # tier specs: explicit config > the run dir's resolved_config.yaml >
    # the run's stored wandb config (empty until the run finishes syncing)
    cand = args.config or f"checkpoints/{run.name}/resolved_config.yaml"
    if os.path.exists(cand):
        specs = load_config(cand, []).model.tiers
        print(f"specs from {cand}")
    else:
        from georvq.config import Config
        cfg = dict(run.config)
        if not cfg.get("model"):
            raise SystemExit(
                f"no tier specs: {cand} not found and the run's wandb "
                f"config is empty (still syncing?). Pass --config.")
        specs = Config.model_validate(cfg).model.tiers

    keys = [k for k in run.summary.keys() if k.startswith("reset/")] or None
    hist = run.history(keys=None, pandas=True)
    cols = [c for c in hist.columns if re.match(r"reset/pyr\d+_L\d+_", c)]
    if not cols:
        raise SystemExit("no per-level reset/* series in this run's history "
                         "-- nothing to plot (no reset has fired yet?)")
    rows = hist.dropna(subset=cols, how="all")
    print(f"{len(rows)} reset events, {len(cols)} per-level series")

    os.makedirs(args.out, exist_ok=True)
    for _, r in rows.iterrows():
        step = int(r["_step"])
        rst = {c: r[c] for c in cols if r[c] == r[c]}     # drop NaN
        if not rst:
            continue
        fig = reset_figure(rst, specs, step)
        p = os.path.join(args.out, f"reset_{run.name}_{step:06d}.png")
        fig.savefig(p, dpi=90)
        plt.close(fig)
        n = sum(v for k, v in rst.items() if k.endswith("_n_reset"))
        print(f"  step {step:>6}: {n:>6.0f} codes revived -> {p}")


if __name__ == "__main__":
    main()
