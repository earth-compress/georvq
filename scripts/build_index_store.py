"""Cache the frozen codec's BITSTREAM for a whole corpus.

Writes the RVQ indices every window would be encoded to, plus the
stats_cond mean where the model transmits one. With frozen codebooks the
indices are all a decoder-side experiment needs: index -> latent is a
table lookup, so the encoder never has to run again.

Indices, not latents. Every tier of the released Tessera codec has
K=256, so an index is exactly one byte and a 64 px window costs
20 + 320 + 5120 = 5,460 B. The dequantized latents for the same window
are 74,240 int8 -- 14x larger for information that is a lookup away.
The indices are also literally the bitstream, so building this store is
an end-to-end encode of the corpus rather than a proxy for one.

Output layout (--out):
    idx_t{i}.u8.npy   (N, cells_i, depth_i) uint8 memmap per tier
    mean.f16.npy      (N, bands) float16, when the model has stats_cond
    index.parquet     window -> chip_id, shard, slot, r0, c0, lat, lon
    manifest.json     source checkpoint, bitstream_id, geometry, timings

Usage:
  python scripts/build_index_store.py --ckpt <ckpt> --store <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import torch

from georvq.data.store import open_store
from georvq.factory import from_checkpoint
from georvq.inference.api import Codec
from georvq.inference.release import bitstream_id


def window_grid(chip_px, win):
    """Non-overlapping tiling; the ragged edge is dropped."""
    n = chip_px // win
    return [(r * win, c * win) for r in range(n) for c in range(n)]


def reader(store, rows, offs, win, q, n_threads, tid, aux):
    """Feed whole chips; the GPU side cuts them into windows."""
    for j in range(tid, len(rows), n_threads):
        r = rows.iloc[j]
        sh, slot = int(r["shard"]), int(r["slot"])
        chip = np.asarray(store.open_shard(sh)[slot])
        if aux:
            chip = np.concatenate(
                [chip, store.aux_bytes(sh, slot, 0, store.chip_px)])
        q.put((j, chip))
    q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256,
                    help="windows per encode call")
    ap.add_argument("--readers", type=int, default=8)
    ap.add_argument("--max-chips", type=int, default=None,
                    help="smoke-test subset")
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                       else "cpu")
    model = from_checkpoint(args.ckpt, dev).eval()
    cfg = model.full_config
    win = cfg.model.window
    codec = Codec(model)
    bid = bitstream_id(model)

    store = open_store(cfg.data.store_dir if args.store == "auto"
                       else args.store)
    rows = store.index
    if args.max_chips:
        rows = rows.head(args.max_chips)
    offs = window_grid(store.chip_px, win)
    N = len(rows) * len(offs)
    print(f"{len(rows)} chips x {len(offs)} windows = {N:,} windows "
          f"at {win}px; bitstream {bid}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    specs = model.specs
    maps = [np.lib.format.open_memmap(
        os.path.join(args.out, f"idx_t{i}.u8.npy"), mode="w+",
        dtype=np.uint8, shape=(N, sp.cells, sp.depth))
        for i, sp in enumerate(specs)]
    has_mean = model.stats_cond is not None
    mmap = (np.lib.format.open_memmap(
        os.path.join(args.out, "mean.f16.npy"), mode="w+",
        dtype=np.float16, shape=(N, cfg.model.bands)) if has_mean else None)
    total_bytes = sum(m.nbytes for m in maps) + (mmap.nbytes if has_mean else 0)
    print(f"store size {total_bytes/2**30:.2f} GiB", flush=True)

    aux = getattr(store, "aux_bands", 0)
    q: queue.Queue = queue.Queue(maxsize=2 * args.readers)
    threads = [threading.Thread(target=reader,
                                args=(store, rows, offs, win, q,
                                      args.readers, t, aux), daemon=True)
               for t in range(args.readers)]
    for t in threads:
        t.start()

    meta, done, finished = [], 0, 0
    t0 = time.time()
    n_enc, enc_time = 0, 0.0
    buf_x, buf_slot = [], []

    def flush():
        nonlocal n_enc, enc_time
        if not buf_x:
            return
        raw = torch.from_numpy(np.stack(buf_x)).to(dev)
        te = time.time()
        x, valid = model.source.prep(raw, dev)
        stacked, _, mean = codec.encode_batch(x, valid=valid)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        enc_time += time.time() - te
        n_enc += len(buf_slot)
        # fancy-index the whole batch in one C-level scatter; the Python
        # per-window loop this replaces cost more than the encode did
        slots = np.asarray(buf_slot)
        for i, arr in enumerate(stacked):
            maps[i][slots] = arr
        if has_mean:
            mmap[slots] = mean.cpu().numpy().astype(np.float16)
        buf_x.clear(); buf_slot.clear()

    while finished < args.readers:
        item = q.get()
        if item is None:
            finished += 1
            continue
        j, chip = item
        r = rows.iloc[j]
        for oi, (r0, c0) in enumerate(offs):
            buf_x.append(chip[:, r0:r0 + win, c0:c0 + win])
            buf_slot.append(j * len(offs) + oi)
            meta.append((j * len(offs) + oi, int(r["chip_id"]),
                         int(r["shard"]), int(r["slot"]), r0, c0,
                         float(r.get("lat", 0.0)), float(r.get("lon", 0.0))))
            if len(buf_x) >= args.batch:
                flush()
        done += 1
        if done % 200 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(rows)} chips  {n_enc:,} windows  "
                  f"{n_enc/max(el,1e-9):,.0f} win/s overall  "
                  f"{n_enc/max(enc_time,1e-9):,.0f} win/s encode-only",
                  flush=True)
    flush()
    for m in maps:
        m.flush()
    if has_mean:
        mmap.flush()

    idx = pd.DataFrame(meta, columns=["window_id", "chip_id", "shard", "slot",
                                      "r0", "c0", "lat", "lon"])
    idx.to_parquet(os.path.join(args.out, "index.parquet"), index=False)
    el = time.time() - t0
    manifest = {
        "kind": "index_store", "ckpt": os.path.abspath(args.ckpt),
        "bitstream_id": bid, "window": win, "bands": cfg.model.bands,
        "tiers": [str(s) for s in specs], "n_windows": int(N),
        "has_mean": bool(has_mean), "chip_store": os.path.abspath(args.store),
        "bytes": int(total_bytes),
        "wall_s": round(el, 1), "encode_s": round(enc_time, 1),
        "windows_per_s": round(n_enc / max(el, 1e-9), 1),
        "encode_windows_per_s": round(n_enc / max(enc_time, 1e-9), 1),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\ndone: {n_enc:,} windows in {el/60:.1f} min "
          f"({manifest['windows_per_s']:,} win/s overall, "
          f"{manifest['encode_windows_per_s']:,} win/s encode-only)")
    print(f"wrote {args.out} ({total_bytes/2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
