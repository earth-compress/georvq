"""Build a real entropy-coded archive from a chip store with a trained
codec: calibrated rANS bitstreams in parquet chunks, byte-exact verified,
with honest ratio metadata.

  python scripts/build_archive.py --ckpt checkpoints/pyr64/model_best.pt \
      --store-dir /data/AE-samples/chips_1024 --out /data/archives/pyr64 \
      --windows 2048
"""
from __future__ import annotations

import argparse
import json

import torch

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import ChipStore
from georvq.inference.api import load_codec
from georvq.inference.archive import write_archive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--windows", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() and args.gpu >= 0
                          else "cpu")

    codec = load_codec(args.ckpt, device)
    store = ChipStore(args.store_dir)
    train_rows, val_rows = store.split(args.val_frac, seed=args.split_seed)
    rows = val_rows if args.split == "val" else train_rows
    toks = fixed_val_tokens(store, rows, token=codec.window,
                            max_tokens=args.windows)
    print(f"encoding {toks.shape[0]} windows of {codec.window}px", flush=True)

    def batches():
        for i in range(0, toks.shape[0], args.batch):
            s = toks[i:i + args.batch]
            x, _ = codec.model.source.prep(s, device)
            meta = [{"window_id": i + j} for j in range(s.shape[0])]
            yield x, None, meta

    meta = write_archive(args.out, codec, batches())
    print(json.dumps(meta, indent=1))
    assert meta["verify_mismatches"] == 0, "round-trip verification failed!"
    print(f"archive -> {args.out}: "
          f"{meta['bits_per_window_coded']:.0f} coded bits/window vs "
          f"{meta['bits_per_window_naive']:.0f} naive "
          f"(x{meta['coding_gain']:.3f} gain)")


if __name__ == "__main__":
    main()
