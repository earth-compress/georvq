"""Cache the frozen stage-1 encoder's latents for the whole live corpus.

One row per 128px window on a 3x3 grid (offsets 0/114/228, full chip
coverage) of every live-store cell. Per tier, an int8 memmap with one
fixed symmetric scale (calibrated with margin on a sample; stage-1
latents are unit-std by construction, and the sigma-trained decoder is
provably indifferent to int8 noise — R^2 0.999336 -> 0.999326).

Output layout (--out-dir):
    lat_t{i}.i8.npy   (N, cells_i, dim_i) int8 memmap per tier
    index.parquet     window -> (chip_id, grid_cell, r0, c0, lat, lon,
                      valid_frac, px_mean, px_std)
    manifest.json     scales, encoder checkpoint + hash, geometry

Usage:
  python scripts/build_latent_store.py
"""
from __future__ import annotations

import argparse
import hashlib
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

OFFSETS = (0, 114, 228)                  # 3x3 grid on a 356px chip
WIN = 128


def reader(store, ids, q, n_threads, tid):
    for j in range(tid, len(ids), n_threads):
        q.put((j, ids[j], store.read_chip(int(ids[j]))))
    q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/dem128_stage1_live/model.pt")
    ap.add_argument("--store", default="/data/georvq-stores/copdem_live")
    ap.add_argument("--out-dir", default="/data/georvq-stores/copdem_lat128")
    ap.add_argument("--device", default="cuda",
                    help="torch device; cuda is the card Slurm allocated")
    ap.add_argument("--batch-chips", type=int, default=384)   # x9 windows
    ap.add_argument("--readers", type=int, default=8)
    ap.add_argument("--calib-chips", type=int, default=2000)
    ap.add_argument("--scale-margin", type=float, default=1.2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    model = from_checkpoint(args.ckpt, dev)
    source = model.source
    store = open_store(args.store)
    n_chips = len(store.index) if args.limit is None else args.limit
    n_win = n_chips * 9
    specs = model.specs
    os.makedirs(args.out_dir, exist_ok=True)

    grid = [(r, c) for r in OFFSETS for c in OFFSETS]

    @torch.no_grad()
    def encode(chips_np):
        """(B, 1, 356, 356) -> per-tier latents for the 9 windows each,
        window-major: row w*B+b corresponds to (chip b, window w)."""
        x = torch.from_numpy(np.concatenate(
            [chips_np[:, :, r:r + WIN, c:c + WIN] for r, c in grid]))
        x = x.to(dev, non_blocking=True)
        xq, valid = source.prep(x, dev)
        mu, sd = source.last_stats()
        with torch.autocast("cuda", torch.bfloat16):
            zs = model.encode_stage1(xq)
        return ([z.float() for z in zs], valid.float().mean((1, 2)),
                mu[:, 0], sd[:, 0])

    # ---- calibrate one symmetric int8 scale per tier ----
    rng = np.random.default_rng(0)
    calib = rng.choice(n_chips, min(args.calib_chips, n_chips), replace=False)
    amax = [0.0] * len(specs)
    for i0 in range(0, len(calib), args.batch_chips):
        chips = np.stack([store.read_chip(int(i))
                          for i in calib[i0:i0 + args.batch_chips]])
        zs, _, _, _ = encode(chips)
        amax = [max(a, float(z.abs().amax())) for a, z in zip(amax, zs)]
    scales = [a * args.scale_margin / 127.0 for a in amax]
    print(f"calibrated scales (amax={['%.2f' % a for a in amax]}): "
          f"{['%.4f' % s for s in scales]}", flush=True)

    mms = [np.lib.format.open_memmap(
        os.path.join(args.out_dir, f"lat_t{i}.i8.npy"), mode="w+",
        dtype=np.int8, shape=(n_win, s.grid ** 2, s.dim))
        for i, s in enumerate(specs)]

    ids = np.arange(n_chips)
    q = queue.Queue(maxsize=args.batch_chips * 3)
    threads = [threading.Thread(target=reader, daemon=True,
                                args=(store, ids, q, args.readers, t))
               for t in range(args.readers)]
    for t in threads:
        t.start()

    meta = {"chip_id": np.zeros(n_win, np.int64),
            "valid_frac": np.zeros(n_win, np.float32),
            "px_mean": np.zeros(n_win, np.float32),
            "px_std": np.zeros(n_win, np.float32)}
    buf, done_readers, n_done, t0 = [], 0, 0, time.time()

    def flush(buf):
        nonlocal n_done
        js = np.array([b[0] for b in buf])
        cids = np.array([b[1] for b in buf])
        chips = np.stack([b[2] for b in buf])
        zs, vfrac, mu, sd = encode(chips)
        B = len(buf)
        for w in range(9):
            rows = js * 9 + w                       # window-major store
            sl = slice(w * B, (w + 1) * B)
            for i, (z, s) in enumerate(zip(zs, scales)):
                qz = torch.round(z[sl] / s).clamp(-127, 127).to(torch.int8)
                mms[i][rows] = qz.cpu().numpy()
            meta["chip_id"][rows] = cids
            meta["valid_frac"][rows] = vfrac[sl].cpu().numpy()
            meta["px_mean"][rows] = mu[sl].cpu().numpy()
            meta["px_std"][rows] = sd[sl].cpu().numpy()
        n_done += B
        if n_done % (args.batch_chips * 20) < args.batch_chips:
            r = n_done / (time.time() - t0)
            print(f"{n_done}/{n_chips} chips ({r:.0f}/s, "
                  f"eta {(n_chips - n_done) / r / 60:.0f}min)", flush=True)

    while done_readers < args.readers:
        item = q.get()
        if item is None:
            done_readers += 1
            continue
        buf.append(item)
        if len(buf) >= args.batch_chips:
            flush(buf)
            buf = []
    if buf:
        flush(buf)
    for m in mms:
        m.flush()

    ridx = store.index.set_index("chip_id")
    cells = ridx.grid_cell.reindex(meta["chip_id"]).to_numpy()
    idx = pd.DataFrame({
        "window": np.arange(n_win),
        "chip_id": meta["chip_id"],
        "grid_cell": cells,
        "r0": np.tile(np.repeat(OFFSETS, 3), n_chips),
        "c0": np.tile(np.tile(OFFSETS, 3), n_chips),
        "lat": ridx.lat.reindex(meta["chip_id"]).to_numpy(),
        "lon": ridx.lon.reindex(meta["chip_id"]).to_numpy(),
        "valid_frac": meta["valid_frac"],
        "px_mean": meta["px_mean"], "px_std": meta["px_std"],
    })
    idx.to_parquet(os.path.join(args.out_dir, "index.parquet"))

    with open(args.ckpt, "rb") as f:
        ck_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({
            "kind": "latent_store", "window": WIN, "offsets": list(OFFSETS),
            "n_windows": n_win, "n_chips": n_chips,
            "tiers": [str(s) for s in specs],
            "scales": scales, "dtype": "int8",
            "norm": "encode_stage1 (L2 per cell x sqrt(dim))",
            "encoder_ckpt": args.ckpt, "encoder_sha256_16": ck_hash,
            "pixel_store": args.store,
            "builder_config": vars(args),
        }, f, indent=1)
    print(f"done: {n_win} windows -> {args.out_dir} "
          f"({sum(m.nbytes for m in mms) / 2**30:.0f} GiB)", flush=True)


if __name__ == "__main__":
    main()
