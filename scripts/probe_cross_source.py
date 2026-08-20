"""Cross-source informational overlap: predict between DEM and S2 latents
on co-located Major-TOM footprints.

Pairs: beta_v2 cache patches (252 px @10 m, indexed by scene cell +
patch_row/col with stride 272) joined to Core-DEM chips by grid cell; the
dem128 codec encodes the 128 px @30 m window centered on each patch
footprint. Descriptors: DEM = coarse-tier latent (256d); S2 = mean CLS
(128d) ++ mean HQ (128d) from the cached pre-VQ latents.

Both directions, ridge (linear overlap) + MLP (nonlinear overlap),
train/val split — held-out R^2 is the overlap number (predictive
information shared between sources), train R^2 reported for reference.

  python scripts/probe_cross_source.py --gpu 0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from georvq.data.store import ChipStore
from georvq.factory import from_checkpoint

CACHE = "/mnt/raid/data/earth-compress-prototypes/beta_v2_deblock_cache_250k"
PARTS = ["part0", "part1", "part2"]
STRIDE_10M, PATCH_10M = 272, 252


def build_pairs(store, max_pairs, rng):
    dem = store.index.set_index("grid_cell")[["shard", "slot"]]
    frames = []
    for p in PARTS:
        s = pd.read_parquet(os.path.join(CACHE, p, "index.parquet"))
        s["cell"] = s.scene_relpath.str.split("/").str[-2]
        s["part"] = p
        s["row_in_part"] = np.arange(len(s))
        frames.append(s[["cell", "part", "row_in_part", "patch_row",
                         "patch_col"]])
    s2 = pd.concat(frames)
    j = s2.join(dem, on="cell", how="inner")
    print(f"{len(j)} matched patches across {j.cell.nunique()} cells "
          f"({len(s2)} S2 patches, {len(dem)} DEM cells)", flush=True)
    return j.sample(min(max_pairs, len(j)), random_state=rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt",
                    default="checkpoints/dem128_overnight_resume/model_best.pt")
    ap.add_argument("--max-pairs", type=int, default=45000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.set_float32_matmul_precision("high")

    model = from_checkpoint(args.ckpt, device).eval()
    W = model.window
    store = ChipStore("/data/georvq-stores/copdem_356")
    pairs = build_pairs(store, args.max_pairs, 0)

    # ---- DEM descriptors: coarse-tier latent of the centered window ----
    pairs = pairs.sort_values(["shard", "slot"])
    dem_z, wins, shard = [], [], {}
    metas = list(pairs.itertuples())
    with torch.no_grad():
        for k, r in enumerate(metas):
            sh = int(r.shard)
            if sh not in shard:
                shard = {sh: store.open_shard(sh)}
            a = np.asarray(shard[sh][int(r.slot)])
            c10r = r.patch_row * STRIDE_10M + PATCH_10M // 2
            c10c = r.patch_col * STRIDE_10M + PATCH_10M // 2
            tr = int(np.clip(round(c10r / 3) - W // 2, 0,
                             store.chip_px - W))
            tc = int(np.clip(round(c10c / 3) - W // 2, 0,
                             store.chip_px - W))
            wins.append(a[:, tr:tr + W, tc:tc + W].copy())
            if len(wins) == 512 or k == len(metas) - 1:
                x, _ = model.source.prep(
                    torch.from_numpy(np.stack(wins)), device)
                dem_z.append(model.encode_levels(x)[0][:, 0].float().cpu())
                wins = []
    D = torch.cat(dem_z)
    print(f"DEM descriptors {tuple(D.shape)}", flush=True)

    # ---- S2 descriptors: mean CLS ++ mean HQ from the cache ----
    s2_feats = torch.empty(len(metas), 256)
    for p in PARTS:
        sel = [(k, r.row_in_part) for k, r in enumerate(metas)
               if r.part == p]
        if not sel:
            continue
        ks, rowids = map(np.asarray, zip(*sel))
        cl = np.load(os.path.join(CACHE, p, "cls.npy"), mmap_mode="r")
        hq = np.load(os.path.join(CACHE, p, "hq.npy"), mmap_mode="r")
        order = np.argsort(rowids)
        for kk in np.array_split(order, max(1, len(order) // 2000)):
            rid = rowids[kk]
            s2_feats[ks[kk], :128] = torch.from_numpy(
                np.asarray(cl[rid]).astype(np.float32).mean(1))
            s2_feats[ks[kk], 128:] = torch.from_numpy(
                np.asarray(hq[rid]).astype(np.float32).mean((1, 2)))
    S = s2_feats
    print("S2 descriptors done", flush=True)

    # ---- fit both directions ----
    for nm, A in [("DEM", D), ("S2", S)]:
        bad = ~torch.isfinite(A).all(1)
        if bad.any():
            print(f"WARNING: {int(bad.sum())} non-finite {nm} rows "
                  f"(sanitized to 0)", flush=True)
    keep = torch.isfinite(D).all(1) & torch.isfinite(S).all(1)
    D, S = D[keep], S[keep]
    print(f"{D.shape[0]} clean pairs", flush=True)

    n = D.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_tr = int(n * (1 - args.val_frac))
    tr, va = perm[:n_tr], perm[n_tr:]

    def standardize(A):
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
        return (A - mu) / sd

    Dn, Sn = standardize(D).to(device), standardize(S).to(device)

    def r2(pred, y, mu):
        return float(1 - ((pred - y) ** 2).mean()
                     / (y - mu).pow(2).mean())

    for name, Xf, Yf in [("S2 -> DEM", Sn, Dn), ("DEM -> S2", Dn, Sn)]:
        muY = Yf[tr].mean(0)
        # ridge (linear overlap)
        X, Y = Xf[tr], Yf[tr]
        lam = 1e-2 * n_tr
        Wr = torch.linalg.solve(
            X.T @ X + lam * torch.eye(X.shape[1], device=device), X.T @ Y)
        print(f"{name} ridge:  train R2={r2(X @ Wr, Y, muY):.3f}  "
              f"val R2={r2(Xf[va] @ Wr, Yf[va], muY):.3f}", flush=True)
        # MLP (nonlinear overlap)
        mlp = torch.nn.Sequential(
            torch.nn.Linear(Xf.shape[1], args.hidden), torch.nn.GELU(),
            torch.nn.Linear(args.hidden, args.hidden), torch.nn.GELU(),
            torch.nn.Linear(args.hidden, Yf.shape[1])).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr,
                                weight_decay=0.01)
        best, best_r2t, since = -1e9, 0.0, 0
        for step in range(1, args.steps + 1):
            idx = tr[torch.randint(0, n_tr, (args.batch,))]
            loss = torch.nn.functional.mse_loss(mlp(Xf[idx]), Yf[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 200 == 0:
                mlp.eval()
                with torch.no_grad():
                    v = r2(mlp(Xf[va]), Yf[va], muY)
                    t = r2(mlp(Xf[tr]), Yf[tr], muY)
                mlp.train()
                print(f"  [{name}] step {step}: train R2={t:.3f} "
                      f"val R2={v:.3f}"
                      f"{'  *' if v > best else ''}", flush=True)
                if v > best:
                    best, best_r2t, since = v, t, 0
                else:
                    since += 1
                    if since >= 8:
                        print(f"  [{name}] early stop at step {step}",
                              flush=True)
                        break
        print(f"{name} MLP:    train R2={best_r2t:.3f}  val R2={best:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
