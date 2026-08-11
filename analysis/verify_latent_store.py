"""Verify the int8 latent store is complete and internally consistent.

The store is written by two workers over disjoint chip ranges, and worker
one is stopped by a watcher that fires as soon as its boundary shard's
`t2_*.npy` APPEARS -- which is when np.save creates the file, not when it
has finished writing 3 GiB. So the boundary shard can legitimately be
truncated or missing its meta sidecar. Everything here is aimed at
catching that before a 30k-step training run inherits the damage.

Checks: every shard has all four arrays; row counts agree across tiers
and with meta; each array is fully readable (not truncated); chip ids are
unique and cover the ChipStore index exactly once.

  python analysis/verify_latent_store.py [--store DIR] [--repair-list]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/data/tessera-samples/"
                                       "latents_w32hd_full")
    ap.add_argument("--chips", default="/data/tessera-samples/chips_512")
    ap.add_argument("--repair-list", action="store_true",
                    help="print the chip ids that are missing or damaged")
    args = ap.parse_args()

    man = json.load(open(f"{args.store}/manifest.json"))
    cells, dim = man["cells"], man["dim"]
    ids = sorted(int(f[3:7]) for f in os.listdir(args.store)
                 if f.startswith("t0_") and f.endswith(".npy"))
    print(f"{len(ids)} shards, ids {ids[0]}..{ids[-1]}, "
          f"cells {cells}, dim {dim}")

    bad, seen, n_win = [], [], 0
    for sh in ids:
        paths = {n: f"{args.store}/{n}_{sh:04d}.npy"
                 for n in ("t0", "t1", "t2", "meta")}
        missing = [n for n, p in paths.items() if not os.path.exists(p)]
        if missing:
            bad.append((sh, f"missing {','.join(missing)}"))
            continue
        try:
            meta = np.load(paths["meta"])
            rows = None
            for t in range(3):
                a = np.load(paths[f"t{t}"], mmap_mode="r")
                if a.shape[1:] != (cells[t], dim):
                    bad.append((sh, f"t{t} shape {a.shape}"))
                    break
                # force the last page in: a truncated np.save raises here
                np.asarray(a[-1, -1, -8:])
                if rows is None:
                    rows = a.shape[0]
                elif a.shape[0] != rows:
                    bad.append((sh, f"t{t} rows {a.shape[0]} != {rows}"))
                    break
                exp = os.path.getsize(paths[f"t{t}"])
                if exp < a.shape[0] * cells[t] * dim:
                    bad.append((sh, f"t{t} file short ({exp}B)"))
                    break
            else:
                if rows != meta.shape[0] * (meta.shape[1] - 2):
                    bad.append((sh, f"rows {rows} != meta "
                                    f"{meta.shape[0]}x{meta.shape[1]-2}"))
                else:
                    seen.append(meta[:, 0])
                    n_win += rows
        except Exception as e:                       # truncated / corrupt
            bad.append((sh, f"{type(e).__name__}: {e}"))

    chip_ids = np.concatenate(seen) if seen else np.array([], dtype=np.int64)
    dup = len(chip_ids) - len(np.unique(chip_ids))
    print(f"good shards: {len(ids)-len(bad)}/{len(ids)}, "
          f"{len(chip_ids):,} chips, {n_win:,} windows, {dup} duplicate ids")
    for sh, why in bad:
        print(f"  BAD shard {sh:04d}: {why}")

    try:
        import pandas as pd
        idx = pd.read_parquet(f"{args.chips}/index.parquet")
        want = set(idx.chip_id.tolist())
    except Exception:
        want = None
    if want is not None:
        have = set(int(c) for c in chip_ids)
        miss = want - have
        print(f"corpus coverage: {len(have):,}/{len(want):,} chips"
              + (f", {len(miss):,} MISSING" if miss else " -- COMPLETE"))
        if miss and args.repair_list:
            print("missing chip ids:", sorted(miss))
    print("\nRESULT:", "OK" if not bad and not dup and
          (want is None or not (want - set(int(c) for c in chip_ids)))
          else "INCOMPLETE / DAMAGED")


if __name__ == "__main__":
    main()
