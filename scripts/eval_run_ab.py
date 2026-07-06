"""A/B downstream eval: original vs compressed->decompressed AEF embeddings.

Loads an eval dataset (eval_build_dataset.py) and a codec checkpoint, samples the
original embedding and the codec's reconstruction at each labelled pixel, and
runs the identical frozen-embedding probes (kNN k=1/k=3, linear) on both. Only
the codec differs between the two columns, so the metric delta is the downstream
cost of compression.

The codec compresses spatial windows: each stored context crop is sub-tiled into
the checkpoint's native window (e.g. 64 or 128), every needed window is
encoded->decoded once, and the reconstructed *pixel* is read back at the label
location -- faithful to how the codec runs in deployment.

Examples:
  # harness sanity before any checkpoint exists (delta should be ~0 / negative):
  python scripts/eval_run_ab.py --dataset /data/AE-samples/eval/aster_ged --sanity identity
  python scripts/eval_run_ab.py --dataset /data/AE-samples/eval/aster_ged --sanity noise --noise-sigma 0.05
  # real model:
  python scripts/eval_run_ab.py --dataset /data/AE-samples/eval/aster_ged \
      --ckpt checkpoints/pyr64/model_best.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from georvq.eval.codec_ab import load_codec
from georvq.eval.extract import dequant_np, l2norm_np
from georvq.eval.probes import run_probes
from georvq.sources.ae import dequant


@torch.no_grad()
def reconstruct_windows(model, win_int8, device, tuple_ls, batch=256):
    """(Nw,64,W,W) int8 -> (Nw,64,W,W) float32 unit-norm reconstruction."""
    outs = []
    for i in range(0, len(win_int8), batch):
        s = torch.from_numpy(win_int8[i:i + batch]).to(device)
        x = dequant(s)                              # training-identical input
        if tuple_ls is None:
            o = model.eval_ceiling(x)               # full-depth (best quality)
        else:
            _, cums = model.level_cumsums(x)
            o = model.eval_tuple(cums, tuple_ls, x)
        outs.append(o.float().cpu().numpy())
    return np.concatenate(outs)


def recon_pixels(model, W, crop, windows, t, device, tuple_ls):
    """Reconstruct each target pixel through the codec at its native window."""
    assert crop % W == 0 and crop >= W, f"crop {crop} not a multiple of W {W}"
    wi = t.window_idx.to_numpy()
    r, c = t.r.to_numpy(), t.c.to_numpy()
    sr, sc, lr, lc = r // W, c // W, r % W, c % W
    # one codec forward per unique (crop, sub-window)
    key = np.stack([wi, sr, sc], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    sub = np.empty((len(uniq), 64, W, W), np.int8)
    for k, (w, a, b) in enumerate(uniq):
        sub[k] = windows[w][:, a * W:(a + 1) * W, b * W:(b + 1) * W]
    recon = reconstruct_windows(model, sub, device, tuple_ls)   # (U,64,W,W)
    return recon[inv, :, lr, lc]                                # (N,64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--sanity", choices=["identity", "noise"], default=None,
                    help="codec stand-in to validate the harness w/o a ckpt")
    ap.add_argument("--noise-sigma", type=float, default=0.05)
    ap.add_argument("--tuple", default=None,
                    help="comma rate tuple e.g. 8,1,0 (default: full depth)")
    ap.add_argument("--budgets", default="50,200,all")
    ap.add_argument("--probes", default="knn1,knn3,linear")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", default=None, help="results csv (default <ds>/ab_<tag>.csv)")
    args = ap.parse_args()
    if not args.ckpt and not args.sanity:
        ap.error("pass --ckpt <model.pt> or --sanity {identity,noise}")

    meta = json.load(open(os.path.join(args.dataset, "meta.json")))
    windows = np.load(os.path.join(args.dataset, "windows.npy"), mmap_mode="r")
    t = pd.read_parquet(os.path.join(args.dataset, "targets.parquet"))
    crop = int(meta["crop"])
    y = t.label.to_numpy()
    groups = t.fid.to_numpy()
    budgets = [None if b == "all" else int(b) for b in args.budgets.split(",")]
    probes = tuple(args.probes.split(","))
    tuple_ls = [int(x) for x in args.tuple.split(",")] if args.tuple else None
    print(f"dataset: {meta['benchmark']} ({meta['task']}, {meta['metric']}) | "
          f"{len(t)} targets / {t.fid.nunique()} tiles / {len(windows)} crops",
          flush=True)

    # X_orig: published embedding at each pixel (advanced index, no big alloc)
    wi, r, c = t.window_idx.to_numpy(), t.r.to_numpy(), t.c.to_numpy()
    X_orig = l2norm_np(dequant_np(windows[wi, :, r, c]))         # (N,64)

    tag = args.sanity or os.path.basename(os.path.dirname(
        args.ckpt + "/")) or "ckpt"
    if args.sanity == "identity":
        X_recon = X_orig.copy()
    elif args.sanity == "noise":
        rng = np.random.default_rng(args.seed)
        X_recon = l2norm_np(X_orig + rng.normal(
            0, args.noise_sigma, X_orig.shape).astype(np.float32))
    else:
        model, W, _ = load_codec(args.ckpt, args.device)
        print(f"codec: window={W}, tuple={tuple_ls or 'full'}", flush=True)
        X_recon = l2norm_np(recon_pixels(model, W, crop, windows, t,
                                         args.device, tuple_ls))

    # 1 - cosine between orig and recon embeddings (codec fidelity, not task)
    cos = float((X_orig * X_recon).sum(1).mean())
    print(f"mean embedding cos(orig, recon) = {cos:.5f} "
          f"(1-cos={1-cos:.2e})", flush=True)

    o_rows = run_probes(X_orig, y, groups, meta["task"], budgets, probes,
                        args.test_frac, args.seed)
    r_rows = run_probes(X_recon, y, groups, meta["task"], budgets, probes,
                        args.test_frac, args.seed)

    res = pd.DataFrame(o_rows).rename(columns={"value": "orig"})
    res["recon"] = [row["value"] for row in r_rows]
    res["delta"] = res["recon"] - res["orig"]
    res = res[["probe", "budget", "n_train", "n_test", "metric",
               "orig", "recon", "delta"]]
    pd.set_option("display.width", 200)
    print(f"\nA/B downstream ({meta['metric']}), codec={tag}, "
          f"embed cos={cos:.4f}:\n{res.to_string(index=False)}", flush=True)

    out = args.out or os.path.join(args.dataset, f"ab_{tag}.csv")
    res.to_csv(out, index=False)
    print(f"\nwrote {out}", flush=True)

    if args.wandb:
        import wandb
        wandb.init(project="georvq", job_type="downstream_eval",
                   name=f"{meta['benchmark']}_{tag}",
                   config={**meta, "codec": tag, "embed_cos": cos,
                           "tuple": tuple_ls})
        for _, row in res.iterrows():
            wandb.log({"probe": row.probe, "budget": str(row.budget),
                       f"{meta['metric']}/orig": row.orig,
                       f"{meta['metric']}/recon": row.recon,
                       f"{meta['metric']}/delta": row.delta})
        wandb.summary["embed_cos"] = cos
        wandb.finish()


if __name__ == "__main__":
    main()
