"""AlphaEarth variable-compression GIF over a grid of eval patches.

4 original 128px AE patches (rows), each shown under TWO PCA renderings (PCs
0-2 and PCs 3-5 as RGB) interleaved with their reconstruction, plus a
per-pixel cosine-similarity quality map over all channels (white=100%,
white->red, fixed scale across all patches/rates/models). ONE merged sweep:
the first checkpoint over its full rate range, then any further checkpoint's
high-bitrate tail; cycles with ~2s end pauses.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse

import torch

from georvq.config import TierSpec, parse_tier
from georvq.eval import codec_ab
from georvq.viz import panels

DEFAULT_CKPTS = ["checkpoints/ae_pyr128_deep/model_best.pt",
                 "checkpoints/ae_pyr64_divfull/model_best.pt"]
DEFAULT_TIERS = ["1:256:2048:8", "4:192:8192:8", "16:64:16384:8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=DEFAULT_CKPTS,
                    help="checkpoints, coarsest-window first (its range leads "
                         "the sweep; later models add the high-bitrate tail)")
    ap.add_argument("--windows",
                    default="/data/AE-samples/eval/aster_ged/windows.npy",
                    help="eval windows array to pick display patches from")
    ap.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS,
                    help="tier specs grid:dim:codes:depth (shared by ckpts)")
    ap.add_argument("--min-depths", default="6,1,0")
    ap.add_argument("--n-rates", type=int, default=14)
    ap.add_argument("--n-chips", type=int, default=4)
    ap.add_argument("--out", default="plots/ae_gifs/ae_variable_compression.gif")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    specs = [TierSpec(**parse_tier(s)) for s in args.tiers]
    mins = [int(x) for x in args.min_depths.split(",")]
    tuples = codec_ab.select_front_tuples(specs, mins, args.n_rates)
    chips = panels.pick_chips(args.windows, n=args.n_chips)
    models = [(os.path.basename(os.path.dirname(p)) or "ckpt", p)
              for p in args.ckpt]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    panels.build_grid_gif(chips, models, specs, tuples, args.out, args.device)


if __name__ == "__main__":
    main()
