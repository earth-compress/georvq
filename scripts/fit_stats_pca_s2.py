"""Fit the Sentinel-2 per-tile stats side-channel basis.

`S2Source.prep` removes 13 per-band means and 13 stds per window; the decode
is meaningless without them, so they are part of the rate. Shipping them raw
costs 26 x f16 = 416 bits. The ring-space beta releases showed a PCA basis
with tolerance-coded uint8 components carries the same 26 numbers in roughly
a tenth of that, which is where `georvq.inference.stats_codec.StatsPacker`
comes from -- this script fits the basis it needs.

The basis is fitted on OUR windows, not copied from the beta release: those
statistics were measured over 42 px tiles and ours over the configured
window, so the distribution the PCA has to span is a different one.

    python scripts/fit_stats_pca_s2.py --window 96 --out artifacts/s2_stats_pca.npz

Writes the npz `StatsPacker` reads, plus the measured packed size that
`S2Source.side_bits` charges (stored in the file so the rate a run reports
is reproducible from the artifact rather than re-measured each time).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from georvq.data.live_s2 import LiveS2Store, cell_lat_lon
from georvq.data.pool import ChipPool
from georvq.inference.stats_codec import StatsPacker, fit_stats_pca


def sample_index(root, n_scenes, seed=0):
    """A random subset of the catalogue, so the basis is not fitted on one
    corner of the world."""
    from georvq.data.live_s2 import build_index
    idx = build_index(root)
    rng = np.random.default_rng(seed)
    take = rng.choice(len(idx), min(n_scenes, len(idx)), replace=False)
    idx = idx.iloc[np.sort(take)].reset_index(drop=True)
    # chip_id is a POSITION into the store's path table, and shard/slot
    # factor it -- so a subset has to be renumbered or the reads land on
    # whatever scene the original global id pointed at (or off the end).
    idx["chip_id"] = np.arange(len(idx))
    idx["shard"] = idx.chip_id // LiveS2Store.SHARD
    idx["slot"] = idx.chip_id % LiveS2Store.SHARD
    return idx


def collect_stats(store, rows, window, n_windows, batch=64, seed=0):
    """(N, 26) stats vectors from real windows, via the real prep path."""
    from georvq.sources.s2 import S2Normalizer

    pool = ChipPool(store, rows, pool_size=min(len(rows), 128),
                    device=torch.device("cpu"), workers=4,
                    refresh_chips_per_s=8.0, seed=seed, pool_device="cpu",
                    period=store.grid_period)
    norm = S2Normalizer()
    out = []
    try:
        while sum(len(s) for s in out) < n_windows:
            raw = pool.sample_batch(batch, token=window)
            x = {k: v.float() / 10000.0 for k, v in raw.items()}
            _, stats = norm.normalize(x, {k: v != 0 for k, v in raw.items()})
            out.append(norm.stats_vector(stats).numpy())
    finally:
        pool.shutdown()
    return np.concatenate(out)[:n_windows]


def measure(basis, stats, tolerance):
    """Packed size and reconstruction error at one tolerance."""
    packer = StatsPacker(data=basis, tolerance=tolerance)
    sizes, errs, raws = [], [], 0
    for s in stats:
        b = packer.pack(s)
        sizes.append(len(b))
        if b[0] == 255:
            raws += 1
        errs.append(np.abs(packer.unpack(b) - s).max())
    return (np.mean(sizes), np.percentile(sizes, 99), np.mean(errs),
            np.max(errs), raws / len(stats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/Major-TOM/Core-S2L1C")
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--scenes", type=int, default=512)
    ap.add_argument("--windows", type=int, default=8192)
    ap.add_argument("--fit-frac", type=float, default=0.5,
                    help="split: fit on this fraction, measure on the rest")
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--out", default="artifacts/s2_stats_pca.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = sample_index(args.root, args.scenes, args.seed)
    store = LiveS2Store(args.root, index=rows)
    print(f"fitting on {len(rows)} scenes -> {args.windows} windows "
          f"at {args.window} px", flush=True)
    stats = collect_stats(store, rows, args.window, args.windows,
                          seed=args.seed)
    n_fit = int(len(stats) * args.fit_frac)
    fit, held = stats[:n_fit], stats[n_fit:]
    print(f"stats {stats.shape}: fit {len(fit)}, held-out {len(held)}")

    basis = fit_stats_pca(fit)
    # how much of the variance the leading components actually carry
    proj = (fit - basis["mean"]) @ basis["components"].T
    var = proj.var(0)
    cum = np.cumsum(var) / var.sum()
    print("components for 99/99.9% of stats variance: "
          f"{int(np.searchsorted(cum, 0.99) + 1)} / "
          f"{int(np.searchsorted(cum, 0.999) + 1)} of {len(var)}")

    print("\ntolerance   mean B   p99 B   mean|err|   max|err|   raw-escape")
    chosen = None
    for tol in (0.05, 0.02, 0.01, 0.005, 0.002):
        m, p99, me, mx, raw = measure(basis, held, tol)
        mark = ""
        if abs(tol - args.tolerance) < 1e-9:
            chosen = (m, p99, tol)
            mark = "  <- charged"
        print(f"  {tol:<9.3f} {m:6.2f}  {p99:6.1f}   {me:.5f}   {mx:.5f}"
              f"   {raw:6.1%}{mark}")

    mean_b, p99_b, tol = chosen
    basis = dict(basis)
    basis["tolerance"] = np.float32(tol)
    basis["mean_bytes"] = np.float32(mean_b)
    basis["p99_bytes"] = np.float32(p99_b)
    basis["window"] = np.int64(args.window)
    basis["n_fit"] = np.int64(len(fit))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **basis)

    raw_bits = 26 * 16
    print(f"\nwrote {args.out}")
    print(f"  charged: {mean_b * 8:.1f} bits/window "
          f"({mean_b * 8 / args.window ** 2:.4f} b/px) "
          f"vs {raw_bits} raw f16 -> {raw_bits / (mean_b * 8):.1f}x smaller")


if __name__ == "__main__":
    main()
