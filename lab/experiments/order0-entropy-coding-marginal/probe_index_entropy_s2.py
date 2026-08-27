"""Beta_v2 (ring-space Sentinel-2) variant of the index-entropy probe.

beta_v2 tiles are (CLS 1x128 + HQ 49x128) x 8 RVQ levels x K=16384. The
deblock cache ships the PRE-VQ continuous latents for 500k Major-TOM chips
(36 tiles each), so we quantize a chip sample against the release codebooks
(plain greedy nearest-neighbour; norm_rvq=False) and reuse the georvq
entropy-model machinery: CLS is the grid-1 "tier", HQ the grid-7 tier, and
cross-tier context = the CLS accumulated vector.

  python scripts/probe_index_entropy_s2.py \
      --ckpt ~/projects/ring-space/releases/beta_v2/model.pt \
      --cache-dir /mnt/raid/data/earth-compress-prototypes/beta_v2_deblock_cache_250k \
      --chips 12000 --arch shared
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_index_entropy import run_entropy_models  # noqa: E402

PARTS = ["part0", "part1", "part2"]
DEPTH, K, D, N_HQ = 8, 16384, 128, 49


def load_codebooks(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd["model_state"]
    return [[sd[f"rvq_{t}.levels.{j}.codebook.weight"] for j in range(DEPTH)]
            for t in ("cls", "hq")]


@torch.no_grad()
def rvq_quantize(z, books, device, batch=1 << 17):
    """Greedy RVQ indices: z (N, D) -> (N, depth) int16. Matmul argmax with
    the constant ||E||^2/2 correction (||z||^2 drops out of the argmin)."""
    Es = [b.to(device).float() for b in books]
    half_sq = [E.pow(2).sum(1) * 0.5 for E in Es]
    out = torch.empty(z.shape[0], len(Es), dtype=torch.int16)
    for s in range(0, z.shape[0], batch):
        r = z[s:s + batch].to(device).float()
        for j, (E, hs) in enumerate(zip(Es, half_sq)):
            idx = (r @ E.t() - hs).argmax(1)
            r = r - E[idx]
            out[s:s + batch, j] = idx.to(torch.int16).cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--chips", type=int, default=12000,
                    help="chips sampled evenly across cache parts "
                         "(36 tiles each)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--train-batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-samples", type=int, default=5_000_000)
    ap.add_argument("--context", default="all",
                    choices=["all", "prev1", "prev2", "first_own",
                             "first_all"])
    ap.add_argument("--arch", default="shared",
                    choices=["probe", "tiny", "shared"])
    ap.add_argument("--proj-dim", type=int, default=96)
    ap.add_argument("--head-rank", type=int, default=128)
    ap.add_argument("--save-idx", default=None)
    ap.add_argument("--load-idx", default=None)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() and args.gpu >= 0
                          else "cpu")
    torch.set_float32_matmul_precision("high")

    books_cls, books_hq = load_codebooks(args.ckpt)
    if args.load_idx:
        z = np.load(args.load_idx)
        idxs = [z["tier0"], z["tier1"]]
        print(f"loaded {idxs[0].shape[0]} tiles from {args.load_idx}",
              flush=True)
    else:
        rng = np.random.default_rng(0)
        cls_idx, hq_idx = [], []
        per_part = args.chips // len(PARTS)
        for p in PARTS:
            hq = np.load(os.path.join(args.cache_dir, p, "hq.npy"),
                         mmap_mode="r")
            cl = np.load(os.path.join(args.cache_dir, p, "cls.npy"),
                         mmap_mode="r")
            sel = np.sort(rng.choice(hq.shape[0], per_part, replace=False))
            for s in range(0, per_part, 500):     # chip-chunked mmap reads
                ch = sel[s:s + 500]
                zc = torch.from_numpy(np.asarray(cl[ch]).reshape(-1, D))
                zh = torch.from_numpy(np.asarray(hq[ch]).reshape(-1, D))
                cls_idx.append(rvq_quantize(zc, books_cls, device))
                hq_idx.append(rvq_quantize(zh, books_hq, device))
            print(f"{p}: {per_part} chips quantized", flush=True)
        n_tiles = args.chips // len(PARTS) * len(PARTS) * 36
        idxs = [torch.cat(cls_idx).view(n_tiles, 1, DEPTH).numpy(),
                torch.cat(hq_idx).view(n_tiles, N_HQ, DEPTH).numpy()]
        perm = np.random.default_rng(1).permutation(n_tiles)
        idxs = [a[perm] for a in idxs]
        if args.save_idx:
            np.savez(args.save_idx,
                     **{f"tier{i}": a for i, a in enumerate(idxs)})
        print(f"quantized {n_tiles} tiles from {args.chips} chips",
              flush=True)

    fake = lambda w: SimpleNamespace(codebook=SimpleNamespace(weight=w))
    levels = [[fake(b.float().to(device)) for b in books_cls],
              [fake(b.float().to(device)) for b in books_hq]]
    specs = [SimpleNamespace(cells=1, depth=DEPTH, codes=K),
             SimpleNamespace(cells=N_HQ, depth=DEPTH, codes=K)]
    # px: one tile covers 42x42 @10m (+20m/60m bands); bpp is per 10m px
    run_entropy_models(idxs, levels, specs, args, device, px=42 * 42)


if __name__ == "__main__":
    main()
