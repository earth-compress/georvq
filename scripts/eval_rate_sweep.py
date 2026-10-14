"""Full downstream-eval report: A/B metric delta across the rate Pareto front.

For each checkpoint, sweeps a set of RVQ rate tuples (the operating points the
codec was trained for), reconstructs every labelled pixel at each rate, and runs
the frozen-embedding probes on the reconstruction -- giving downstream
performance (and its gap to the original embedding) as a function of bits/pixel.
Emits a tidy CSV, a console table, and a rate-distortion-style plot. Overlay
multiple checkpoints (e.g. a 64px and a 128px codec) on one axis. A single
tuple (`--tuples 8-8-8`) is the plain A/B of one operating point.

`--sanity identity|noise` adds a codec stand-in that checks the harness
without a checkpoint: identity should give a delta of zero, and noise
(`--noise-sigma`) a small negative one. Stand-in rows have no rate and are
left off the plot.

Examples:
  python scripts/eval_rate_sweep.py --dataset /data/AE-samples/eval/aster_ged \
      --ckpt checkpoints/pyr128/model_best.pt \
             checkpoints/pyr64/model_best.pt \
      --n-front 8 --out-dir plots/ae_downstream --wandb
  python scripts/eval_rate_sweep.py --dataset /data/AE-samples/eval/aster_ged \
      --sanity noise --noise-sigma 0.05 --out-dir plots/ae_downstream
"""
from __future__ import annotations

import argparse
import json
import os

# cap BLAS threads before numpy import (OpenBLAS crashes past its 128-thread
# build limit on many-core boxes when probing large datasets, e.g. 100k LUCAS)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "32")

import numpy as np
import pandas as pd
import torch

from georvq.eval import codec_ab
from georvq.eval.probes import run_probes
from georvq.model.codec import tuple_bits


def model_tag(path):
    return os.path.basename(os.path.dirname(path))


def sweep_ckpt(ckpt, windows, t, X_orig, y, groups, meta, args, device):
    model, W, specs = codec_ab.load_codec(ckpt, device)
    if args.tuples:
        tuples = [[int(x) for x in s.split("-")] for s in args.tuples.split(",")]
    elif args.n_front > 0:
        tuples = codec_ab.select_front_tuples(specs, model.cfg.min_depths,
                                              args.n_front)
    else:
        tuples = [[int(x) for x in s.split("-")]
                  for s in model.full_config.eval.recon_tuples]
    tag = model_tag(ckpt)
    print(f"\n=== {tag}: window={W}, {len(tuples)} rate points ===", flush=True)

    win, inv, lr, lc = codec_ab.unique_windows(windows, t, W)
    recons = codec_ab.reconstruct_tuples(model, win, inv, lr, lc, tuples,
                                         device)
    budgets = [None if b == "all" else int(b) for b in args.budgets.split(",")]
    probes = tuple(args.probes.split(","))

    rows = []
    for ls, Xr in zip(tuples, recons):
        bpp = tuple_bits(specs, ls) / (W * W)
        cos = float((X_orig * Xr).sum(1).mean())
        rr = run_probes(Xr, y, groups, meta["task"], budgets, probes,
                        args.test_frac, args.seed)
        for r in rr:
            rows.append({"model": tag, "window": W, "tuple": "-".join(map(str, ls)),
                         "bits_per_px": bpp, "embed_cos": cos,
                         "probe": r["probe"], "budget": r["budget"],
                         "recon": r["value"]})
        print(f"  {'-'.join(map(str,ls)):8s} {bpp:7.3f} b/px  cos={cos:.4f}  "
              + "  ".join(f"{r['probe']}/{r['budget']}={r['value']:.3f}"
                          for r in rr), flush=True)
    return pd.DataFrame(rows)


def sanity_rows(kind, X_orig, y, groups, meta, args):
    """Probe a codec stand-in: the original embeddings (identity) or
    those plus Gaussian noise, renormalised (noise)."""
    from georvq.eval.extract import l2norm_np
    if kind == "identity":
        Xr = X_orig.copy()
    else:
        rng = np.random.default_rng(args.seed)
        Xr = l2norm_np(X_orig + rng.normal(
            0, args.noise_sigma, X_orig.shape).astype(np.float32))
    budgets = [None if b == "all" else int(b) for b in args.budgets.split(",")]
    probes = tuple(args.probes.split(","))
    cos = float((X_orig * Xr).sum(1).mean())
    rr = run_probes(Xr, y, groups, meta["task"], budgets, probes,
                    args.test_frac, args.seed)
    tag = f"sanity_{kind}"
    print(f"\n=== {tag}: cos={cos:.4f}  "
          + "  ".join(f"{r['probe']}/{r['budget']}={r['value']:.3f}"
                      for r in rr), flush=True)
    return pd.DataFrame([{"model": tag, "window": np.nan, "tuple": kind,
                          "bits_per_px": np.nan, "embed_cos": cos,
                          "probe": r["probe"], "budget": r["budget"],
                          "recon": r["value"]} for r in rr])


def plot_report(df, orig, meta, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = meta["metric"]
    bud = "all" if "all" in df.budget.unique() else df.budget.iloc[0]
    d = df[(df.budget == bud) & df.bits_per_px.notna()]
    probes = list(dict.fromkeys(d.probe))
    models = list(dict.fromkeys(d.model))
    cmap = plt.get_cmap("tab10")
    mc = {mo: cmap(i) for i, mo in enumerate(models)}
    ls_for = {p: s for p, s in zip(probes, ["-", "--", ":", "-."])}

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for mo in models:
        for p in probes:
            s = d[(d.model == mo) & (d.probe == p)].sort_values("bits_per_px")
            ax[0].plot(s.bits_per_px, s.recon, ls_for[p], marker="o",
                       color=mc[mo], label=f"{mo} · {p}")
            ax[1].plot(s.bits_per_px, s.recon - orig[(p, bud)], ls_for[p],
                       marker="o", color=mc[mo], label=f"{mo} · {p}")
    for p in probes:                                   # original = rate-free
        ax[0].axhline(orig[(p, bud)], color="0.4", ls=ls_for[p], lw=1,
                      alpha=0.7, label=f"original · {p}")
    ax[1].axhline(0, color="0.4", lw=1)
    for a_ in ax:
        a_.set_xscale("log"); a_.set_xlabel("bits / pixel"); a_.grid(alpha=0.3)
    ax[0].set_ylabel(m); ax[0].set_title(
        f"Downstream {m} vs rate ({meta['benchmark']}, budget={bud})")
    ax[1].set_ylabel(f"Δ {m} (recon − orig)")
    ax[1].set_title("Compression cost vs rate")
    ax[0].legend(fontsize=7, ncol=1)
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"\nwrote plot -> {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", nargs="+", default=[])
    ap.add_argument("--sanity", choices=["identity", "noise"], default=None,
                    help="codec stand-in that checks the harness without "
                         "a checkpoint")
    ap.add_argument("--noise-sigma", type=float, default=0.05)
    ap.add_argument("--n-front", type=int, default=8,
                    help="auto-pick N rate points spread across bits "
                         "(0 => use the checkpoint's recon_tuples)")
    ap.add_argument("--tuples", default=None,
                    help="explicit rate tuples, e.g. 8-1-0,8-8-0,8-8-8")
    ap.add_argument("--budgets", default="200,all")
    ap.add_argument("--probes", default="knn3,linear")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out-dir", default="plots/ae_downstream")
    ap.add_argument("--label-col", default="label",
                    help="label column to evaluate (e.g. label_lu for land use)")
    ap.add_argument("--restrict-parquet", default=None,
                    help="targets.parquet of a matched dataset (e.g. the "
                         "tessera re-extraction): keep only this dataset's "
                         "rows whose index appears in its ae_idx column, so "
                         "both sources are probed on the same points")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    if not args.ckpt and not args.sanity:
        ap.error("pass --ckpt <model.pt> ... and/or --sanity {identity,noise}")
    os.makedirs(args.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.dataset, "meta.json")))
    windows = np.load(os.path.join(args.dataset, "windows.npy"), mmap_mode="r")
    t = pd.read_parquet(os.path.join(args.dataset, "targets.parquet"))
    if args.restrict_parquet:
        keep = set(pd.read_parquet(args.restrict_parquet,
                                   columns=["ae_idx"]).ae_idx)
        t = t[t.index.isin(keep)]
        print(f"restricted to {len(t)} matched targets "
              f"({args.restrict_parquet})", flush=True)
    lc = args.label_col
    if lc not in t.columns:
        raise SystemExit(f"--label-col {lc} not in dataset ({list(t.columns)})")
    t = t.dropna(subset=[lc]).reset_index(drop=True)
    if meta["task"] == "classification":               # drop rare -> stable BA
        vc = t[lc].value_counts()
        t = t[t[lc].isin(vc[vc >= 50].index)].reset_index(drop=True)
    tag = meta["benchmark"] + ("" if lc == "label" else f"_{lc}")
    y, groups = t[lc].to_numpy(), t.fid.to_numpy()
    X_orig = codec_ab.orig_embeddings(windows, t,
                                      source=meta.get("source", "ae"))
    print(f"dataset: {tag} ({meta['task']}, {meta['metric']}) | "
          f"{len(t)} targets / {t.fid.nunique()} tiles | "
          f"classes={sorted(set(y.tolist())) if meta['task']=='classification' else 'reg'}",
          flush=True)

    # original-embedding baseline (rate-independent), keyed by (probe, budget)
    budgets = [None if b == "all" else int(b) for b in args.budgets.split(",")]
    probes = tuple(args.probes.split(","))
    o_rows = run_probes(X_orig, y, groups, meta["task"], budgets, probes,
                        args.test_frac, args.seed)
    orig = {(r["probe"], r["budget"]): r["value"] for r in o_rows}
    print("original baseline: " + "  ".join(
        f"{k[0]}/{k[1]}={v:.3f}" for k, v in orig.items()), flush=True)

    parts = [sweep_ckpt(c, windows, t, X_orig, y, groups, meta, args,
                        args.device) for c in args.ckpt]
    if args.sanity:
        parts.append(sanity_rows(args.sanity, X_orig, y, groups, meta, args))
    df = pd.concat(parts, ignore_index=True)
    df["orig"] = [orig[(p, b)] for p, b in zip(df.probe, df.budget)]
    df["delta"] = df.recon - df.orig

    out_csv = os.path.join(args.out_dir, f"ratesweep_{tag}.csv")
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 220)
    show = df[df.budget == ("all" if "all" in df.budget.values
                            else df.budget.iloc[0])]
    print(f"\n==== {tag} downstream A/B across rate (budget=all) ====\n"
          f"{show[['model','tuple','bits_per_px','embed_cos','probe','orig','recon','delta']].to_string(index=False)}",
          flush=True)
    print(f"\nwrote {out_csv}", flush=True)
    if args.ckpt:
        plot_report(df, orig, {**meta, "benchmark": tag},
                    os.path.join(args.out_dir, f"ratesweep_{tag}.png"))

    if args.wandb:
        import wandb
        wandb.init(project="georvq", job_type="downstream_ratesweep",
                   name=f"{tag}_ratesweep",
                   config={**meta, "label_col": lc, "sanity": args.sanity,
                           "ckpts": [model_tag(c) for c in args.ckpt]})
        wandb.log({"table": wandb.Table(dataframe=df)})
        wandb.finish()


if __name__ == "__main__":
    main()
