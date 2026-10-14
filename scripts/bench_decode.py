"""Decode-side throughput benchmark: windows/s from packets to pixels at
several depth tuples.

  python scripts/bench_decode.py --ckpt checkpoints/pyr64/model_best.pt
"""
from __future__ import annotations

import argparse
import time

import torch

from georvq.inference.api import load_codec
from georvq.train.validate import canonical_tuples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda",
                    help="torch device; cuda is the card Slurm allocated")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")

    codec = load_codec(args.ckpt, device)
    m = codec.model
    x = torch.nn.functional.normalize(
        torch.randn(args.batch, m.bands, m.window, m.window, device=device),
        dim=1)
    packets = codec.encode(x)
    canon = canonical_tuples(m.specs, m.cfg.min_depths)
    for name, t in canon.items():
        codec.decode(packets, t)                      # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            codec.decode(packets, t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        wps = args.iters * args.batch / dt
        px = wps * m.window ** 2
        print(f"{name:12s} {t}: {wps:9,.0f} windows/s "
              f"({px/1e6:8.1f} Mpx/s)")


if __name__ == "__main__":
    main()
