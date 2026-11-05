"""Fit the mean channel of a release, a `RotatedChannel`
(`RotatedChannel.fit_pca`), as the AE v2 and Tessera v1.1 releases send
the window mean.

A stats_cond model trains against the exact per-window mean; the release
codes that mean with a channel fitted afterwards on the training split's
window means. The channel depends only on the store and the split, not
on the trained model. The preset (`--config`) gives the store, the split,
the source, the window and the band count; the defaults are those of the
AE v2 release.

Stage 1 computes the per-window means. Each chip goes through the
source's `prep` (for AE, dequant, per-pixel L2 normalisation, nodata
zeroed; for Tessera, x = q * scale / XNORM with the scale planes read
from the store's sidecars) and each of its non-overlapping windows gets
the masked mean over valid pixels, as the model's `patch_stats` computes
it, summed in float32 (`--float64`: float64). Windows with no valid
pixel are dropped. About 80
minutes on 24 CPU workers for every chip of chips_1024. `--train-chips N`
takes N training chips drawn with seed 0 instead of all of them.

Stage 2 is the fit: the principal axes of `--pca-sample` training means
drawn with seed 0, every component bounded by a hard bound and companded
by its standard deviation, and `--total-bits` shared greedily by the
squared error removed on `--alloc-sample` training means drawn with seed
1. The bound is that of a mean of unit vectors (|y| <= 1 + |mu0|), or
with `--box B` that of values which all lie in [-B, B]. The AE v2
release used 672 bits:

    python scripts/fit_mean_channel.py --means means.npz \\
        --out mean_channel.npz

The Tessera v1.1 release used 1280 bits with the box 127 * SCALE_MAX /
XNORM = 3.96875 (georvq.tessera.v1_1.BOX), the axes fitted on every mean
of 8000 training chips and the bits allocated on 200,000 of them:

    python scripts/fit_mean_channel.py --config configs/tessera/tessera_v11.yaml \\
        --train-chips 8000 --float64 --means means.npz --box 3.96875 \\
        --total-bits 1280 --pca-sample 600000 --alloc-sample 200000 \\
        --out mean_channel.npz

`--means` caches stage 1: it is read when the file exists and written
otherwise. The fit runs in float64 numpy and has no device dependence.
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from georvq.config import load_config
from georvq.inference.mean_codec import RotatedChannel
from georvq.data.store import open_store

_ctx = {}


# ---- stage 1: per-window means ----

def _init(cfg_path, float64=False):
    torch.set_num_threads(1)
    _ctx["float64"] = float64
    from georvq.sources import get_source
    cfg = load_config(cfg_path, [])
    _ctx["store"] = open_store(cfg.data.store_dir)
    _ctx["src"] = get_source(cfg.data.source, cfg.model)
    _ctx["W"], _ctx["C"] = int(cfg.model.window), int(cfg.model.bands)


def _chip(job):
    shard, slot, _ = job
    s, src, W, C = _ctx["store"], _ctx["src"], _ctx["W"], _ctx["C"]
    a = np.array(s.open_shard(shard)[slot])
    if getattr(s, "aux_bands", 0):
        # per-pixel aux planes (Tessera's scales) ride as trailing bands
        a = np.concatenate([a, s.aux_bytes(shard, slot, 0, a.shape[-2])])
    x, v = src.prep(torch.from_numpy(a)[None], "cpu")
    g = a.shape[-1] // W
    xv = x[0] * v[0]
    if _ctx.get("float64"):
        xv = xv.double()
    xs = xv.reshape(C, g, W, g, W).sum((2, 4))
    n = v[0].reshape(g, W, g, W).sum((1, 3)).to(xs.dtype)
    m = (xs / n.clamp_min(1)).permute(1, 2, 0).reshape(g * g, C)
    keep = (n.reshape(-1) > 0)
    return (m[keep].numpy().astype(np.float32),
            (n.reshape(-1)[keep] / W ** 2).numpy().astype(np.float16),
            keep.nonzero().flatten().numpy().astype(np.uint8))


def window_means(rows, workers, cfg_path, label, float64=False):
    """Means (N, bands), valid fractions, chip ids and window positions
    (row * windows per chip row + col) of every window with a valid pixel
    in `rows`."""
    # chip_id, not the frame label: store.split renumbers its frames from 0
    jobs = [(int(r["shard"]), int(r["slot"]), int(r["chip_id"])) for _, r in
            rows.sort_values(["shard", "slot"]).iterrows()]
    out_m, out_f, out_c, out_p = [], [], [], []
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(cfg_path, float64)) as p:
        for i, (m, f, pos) in enumerate(p.imap(_chip, jobs, chunksize=4)):
            out_m.append(m)
            out_f.append(f)
            out_p.append(pos)
            out_c.append(np.full(len(m), jobs[i][2], np.int32))
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"{label}: {i + 1}/{len(jobs)} chips, {el / 60:.1f} min, "
                      f"eta {el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min",
                      flush=True)
    return (np.concatenate(out_m), np.concatenate(out_f),
            np.concatenate(out_c), np.concatenate(out_p))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/ae/ae_v2.yaml",
                    help="preset giving the store and the train/val split")
    ap.add_argument("--means", default=None,
                    help="npz cache of the window means: read if it exists, "
                         "written otherwise")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--train-chips", type=int, default=None,
                    help="stage 1 on this many training chips drawn with "
                         "seed 0 (default all)")
    ap.add_argument("--float64", action="store_true",
                    help="sum each window in float64 (default float32)")
    ap.add_argument("--box", type=float, default=None,
                    help="every value lies in [-BOX, BOX] (default: the "
                         "bound of a mean of unit vectors)")
    ap.add_argument("--pca-sample", type=int, default=2_000_000,
                    help="rotated: training means the axes are fitted on")
    ap.add_argument("--alloc-sample", type=int, default=400_000,
                    help="rotated: training means the bits are allocated on")
    ap.add_argument("--total-bits", type=int, default=672)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.means and os.path.exists(a.means):
        z = np.load(a.means)
        Mtr, Mva = z["Mtr"], z["Mva"]
    else:
        cfg = load_config(a.config, [])
        store = open_store(cfg.data.store_dir)
        tr, va = store.split(cfg.data.val_frac, seed=cfg.data.split_seed)
        if a.train_chips:
            pick = np.random.default_rng(0).choice(len(tr), a.train_chips,
                                                   replace=False)
            tr = tr.iloc[np.sort(pick)]
        Mva, Fva, Cva, Pva = window_means(va, a.workers, a.config, "val",
                                          a.float64)
        Mtr, Ftr, Ctr, Ptr = window_means(tr, a.workers, a.config, "train",
                                          a.float64)
        if a.means:
            np.savez(a.means, Mtr=Mtr, Ftr=Ftr, Ctr=Ctr, Ptr=Ptr,
                     Mva=Mva, Fva=Fva, Cva=Cva, Pva=Pva)
    print(f"means: train {Mtr.shape}, val {Mva.shape}", flush=True)

    n = len(Mtr)

    def draw(seed, k):
        return Mtr[np.sort(np.random.default_rng(seed).choice(
            n, min(k, n), replace=False))]
    ch = RotatedChannel.fit_pca(draw(0, a.pca_sample), draw(1, a.alloc_sample),
                                a.total_bits, box=a.box)
    ch.save(a.out)
    rel = np.median(np.linalg.norm(ch.round_trip(Mva) - Mva, axis=1)
                    / np.linalg.norm(Mva, axis=1))
    print(f"wrote {a.out}: {ch.cost_bits()} bits, widths "
          f"{ch.bits.min()} to {ch.bits.max()}, val median relative L2 "
          f"{rel:.5f}, clipped {int(ch.clipped(Mva).any(1).sum())}",
          flush=True)


if __name__ == "__main__":
    main()
