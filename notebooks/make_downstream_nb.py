"""Generate notebooks/demo_downstream_eval.ipynb — downstream A/B
(AE-paper-style experiments) across the rate ladder."""
import json

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None,
                  "source": src.splitlines(keepends=True)})


md("""# Downstream cost of compression — AE-paper experiments, A/B

The AlphaEarth paper evaluates its embeddings with frozen-embedding
probes on downstream tasks. We reimplemented three of those benchmarks
and run them **A/B: original embeddings vs codec reconstructions**, at
every point on the rate ladder. The probes, the geographic train/test
split, and the labels are identical between the two arms — the *only*
difference is the codec — so the metric delta is the downstream price
of compression at each rate.

| benchmark | task | metric | targets |
|---|---|---|---|
| ASTER GED | emissivity regression | R² | 13,890 |
| LUCAS | land-cover classification | balanced acc. | 100,037 (subsampled 20k here) |
| LCMAP | land-cover classification | balanced acc. | 12,010 |

Protocol notes: probes are kNN(k=1) and a linear model on the frozen
64-d embedding at each labelled pixel; test split is by hashed
geographic block (no spatial leakage); reconstructions decode each
window once and read the labelled pixel back — the deployment path.
(Rates shown are nominal index bits; the 8* sidecar is not applied in
this path, so full-ladder numbers are marginally conservative.)

**Data requirements**: the eval datasets are local artifacts (12/12/98
GB; built from public sources by `scripts/eval_build_dataset.py`, with
train-contamination exclusion based on the training-store indices).
This notebook runs whichever of them exist under `EVAL_ROOT` and skips
the rest with a pointer — the committed, executed copy carries the full
results either way.""")

code("""import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "32")           # BLAS thread cap before numpy
import json, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from georvq.eval import codec_ab
from georvq.eval.extract import dequant_np, l2norm_np
from georvq.eval.probes import run_probes
from georvq.model.codec import tuple_bits

BUNDLE = "/home/ali/projects/georvq/checkpoints/ae_release_beta/bundle/model.pt"
EVAL_ROOT = "/data/AE-samples/eval"
DEVICE = "cuda:1"
LADDER = [[8,1,0],[8,8,0],[8,8,1],[8,8,2],[8,8,4],[8,8,8]]
PROBES, BUDGETS = ("knn1", "linear"), (None,)      # full-budget probes
DATASETS = {"aster_ged": None, "lcmap_lc": None, "lucas_lc": 20000}
                                                   # name -> target subsample""")

md("""## Run the sweep

Each dataset: encode every unique codec window once, decode all ladder
points from the cached cumulative codes, read back the labelled pixels,
and score both arms with the identical probe harness.""")

code("""model, W, specs = codec_ab.load_codec(BUNDLE, DEVICE)
rows = []
for name, sub in DATASETS.items():
    d = f"{EVAL_ROOT}/{name}"
    if not os.path.exists(f"{d}/meta.json"):
        print(f"== {name}: not found at {d} — skipping. Build it with\\n"
              f"   python scripts/eval_build_dataset.py --benchmark {name} "
              f"--out-dir {d}", flush=True)
        continue
    meta = json.load(open(f"{d}/meta.json"))
    windows = np.load(f"{d}/windows.npy", mmap_mode="r")
    t = pd.read_parquet(f"{d}/targets.parquet")
    if sub and len(t) > sub:
        t = t.sample(sub, random_state=0).reset_index(drop=True)
    wi, r, c = t.window_idx.to_numpy(), t.r.to_numpy(), t.c.to_numpy()
    X_orig = l2norm_np(dequant_np(windows[wi, :, r, c]))
    y, groups = t.label.to_numpy(), t.fid.to_numpy()
    print(f"== {name}: {len(t)} targets ({meta['task']}, {meta['metric']})",
          flush=True)

    o_rows = run_probes(X_orig, y, groups, meta["task"], BUDGETS, PROBES)
    orig = {rr["probe"]: rr["value"] for rr in o_rows}
    print("   orig:", {k: round(v, 4) for k, v in orig.items()}, flush=True)

    win, inv, lr, lc = codec_ab.unique_windows(windows, t, W)
    t0 = time.time()
    recons = codec_ab.reconstruct_tuples(model, win, inv, lr, lc,
                                         LADDER, DEVICE)
    print(f"   {len(win)} unique windows x {len(LADDER)} tuples "
          f"decoded in {time.time()-t0:.0f}s", flush=True)
    for ls, Xr in zip(LADDER, recons):
        Xr = l2norm_np(Xr)
        bpp = tuple_bits(specs, ls) / (W * W)
        cos = float((X_orig * Xr).sum(1).mean())
        for rr in run_probes(Xr, y, groups, meta["task"], BUDGETS, PROBES):
            rows.append(dict(benchmark=name, metric=meta["metric"],
                             tuple="-".join(map(str, ls)), bpp=bpp,
                             embed_cos=cos, probe=rr["probe"],
                             recon=rr["value"], orig=orig[rr["probe"]],
                             delta=rr["value"] - orig[rr["probe"]]))
        print(f"   {'-'.join(map(str,ls)):6s} {bpp:6.2f} b/px cos {cos:.4f}  "
              + "  ".join(f"{k}Δ={rows[-2+i]['delta']:+.4f}"
                          for i, k in enumerate(PROBES)), flush=True)
assert rows, "no eval datasets found — see the skip messages above"
df = pd.DataFrame(rows)
df.to_csv("downstream_ab_ladder.csv", index=False)
df.head()""")

md("""## Performance delta vs compression rate

The AE-paper view: how much downstream performance does each rate cost,
relative to the uncompressed embeddings (dashed zero line)? Compression
factor is relative to the int8 source (512 bits/px).""")

code("""fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), sharex=True)
colors = {"knn1": "tab:blue", "linear": "tab:orange"}
for ax, (name, g) in zip(axes, df.groupby("benchmark", sort=False)):
    for probe, gg in g.groupby("probe"):
        gg = gg.sort_values("bpp")
        ax.plot(gg.bpp, gg.delta, "o-", color=colors[probe],
                label=f"{probe} (orig {gg.orig.iloc[0]:.3f})")
    for _, rr in g[g.probe == "linear"].iterrows():
        ax.annotate(rr["tuple"], (rr.bpp, rr.delta), fontsize=7,
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", color="gray")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_title(f"{name}  ({g.metric.iloc[0]})")
    ax.set_xlabel("bits / pixel (nominal)")
    ax.legend(fontsize=8, loc="lower right")
    sec = ax.secondary_xaxis("top", functions=(lambda b: 512/b,
                                               lambda f: 512/f))
    sec.set_xlabel("compression factor vs int8", fontsize=8)
axes[0].set_ylabel("metric delta (compressed − original)")
fig.tight_layout()""")

md("""## Reading (of the measured run above)

- **Regression (ASTER GED) is free**: linear-probe deltas are slightly
  *positive* at every rate (+0.4 pt R² at full ladder — mild denoising),
  kNN within ~1 pt from 0.43 b/px up.
- **kNN probes on the classification tasks are free from ~0.4 b/px**:
  deltas of both signs within ~±1.8 pts (LCMAP's +1.8 at full rate and
  its non-monotone path tell you the metric's own variance is of that
  order on these 12–20k-target benchmarks).
- **Linear classification probes carry a small persistent cost**:
  −0.6 to −1.8 pts balanced accuracy even at full rate, consistently
  signed on LUCAS — linear probes are more sensitive to small geometric
  distortion of the embedding space than kNN. This is the honest
  downstream price of the codec: ≲2 pts on the most sensitive probe.
- **The extreme end is striking**: 0.07 b/px (≈7,300× vs int8) costs
  only 2–6 pts across tasks — the coarse tiers carry most of the
  probe-relevant geometry.
- Single-split numbers; treat |delta| ≲ 1–2 pts as within benchmark
  noise (the sign mixing above calibrates it).

Full per-cell results: `downstream_ab_ladder.csv`. Deeper machinery:
`scripts/eval_rate_sweep.py` (multi-checkpoint sweeps, budgets),
`scripts/eval_run_ab.py` (single-point A/B with sanity harnesses),
`georvq/eval/` (probes, geographic splits, benchmark builders).""")

nb = {"cells": CELLS, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i}"
with open("notebooks/demo_downstream_eval.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/demo_downstream_eval.ipynb")
