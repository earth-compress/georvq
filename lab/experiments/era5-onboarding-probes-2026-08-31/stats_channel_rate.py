"""Corrected rate for the ERA5 stats channel (mean, mean+std).

The training runs charge the channel flat: bits_per_band x bands
(x2 with std) = 222 / 444 bits per 64px window. The 37 levels are
strongly correlated (d95 well under 37 at every scale), so the vectors
should compress far below that. This measures how far, by reverse
water-filling on the PCA spectrum (analysis/moment_rd.py:gauss_bits),
at the distortion the flat 6-bit uniform coder (MeanChannel) actually
achieves -- so the corrected number is an equal-fidelity comparison,
not a different operating point.

Vectors are what the channel transmits, in model space: per-window
per-level mean, and std floored at stats_cond.std_floor. Covariance
and MeanChannel are fitted on train-split windows and scored on
val-split windows. The std block is also water-filled in log space
(stds vary multiplicatively; see analysis/patch_moments_pca.py).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model/analysis")
from moment_rd import gauss_bits                          # noqa: E402

from georvq.cli import load_config                        # noqa: E402
from georvq.data.store import open_store                  # noqa: E402
from georvq.inference.mean_codec import MeanChannel       # noqa: E402
from georvq.sources import get_source                     # noqa: E402

WIN, FLOOR, BITS = 64, 0.05, 6
PX = WIN * WIN


def collect(store, rows, src, n_slices, crops=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = rows.sample(n=n_slices, random_state=seed)
    vs = []
    for shard, grp in rows.groupby("shard"):
        sh = store.open_shard(shard)
        for slot in grp.slot:
            raw = torch.from_numpy(np.asarray(sh[slot]))[None]
            x, _ = src.prep(raw, "cpu")
            x = x[0]
            for _ in range(crops):
                i, j = rng.integers(0, 128 - WIN, 2)
                w = x[:, i:i + WIN, j:j + WIN]
                mu = w.mean(dim=(1, 2))
                sd = w.std(dim=(1, 2), unbiased=False).clamp_min(FLOOR)
                vs.append(torch.cat([mu, sd]))
    return torch.stack(vs)


def rel_err(ref, approx, energy_of):
    """Relative L2: rms error over rms magnitude of `energy_of`."""
    e = float((ref - approx).pow(2).sum(1).mean())
    return (e / float(energy_of.pow(2).sum(1).mean())) ** 0.5


def report(tag, Xtr, Xva, cols):
    A, V = Xtr[:, cols], Xva[:, cols]
    ch = MeanChannel.fit(A.numpy(), bits=BITS)
    rt = torch.from_numpy(ch.round_trip(V.numpy()))
    rel = rel_err(V, rt, V)
    flat = len(cols) * BITS
    bits = gauss_bits(A, [rel, rel / 2, rel * 2])
    corr = bits[rel]
    print(f"{tag:14s} flat {flat:4d} b/win ({flat / PX:.4f} b/px) "
          f"rel={rel:.4f} -> water-filled {corr:6.1f} b/win "
          f"({corr / PX:.4f} b/px, {flat / max(corr, 1e-9):.1f}x)"
          f"   [half-dist {bits[rel / 2]:.1f}, double-dist {bits[rel * 2]:.1f}]")
    lam = torch.linalg.eigvalsh(torch.cov((A - A.mean(0)).T.double()))
    cum = torch.cumsum(lam.flip(0), 0) / lam.sum()
    k99 = int((cum < 0.99).sum()) + 1
    print(f"{'':14s} PCA: {k99}/{len(cols)} components for 99% variance")
    return rel


cfg = load_config(
    "/home/ali/projects/georvq/.claude/worktrees/era5-model/configs/era5/era5_t64.yaml")
src = get_source(cfg.data.source, cfg.model)
store = open_store(cfg.data.store_dir)
tr_rows, va_rows = store.split(cfg.data.val_frac, seed=cfg.data.split_seed)

Xtr = collect(store, tr_rows, src, n_slices=512, seed=1)
Xva = collect(store, va_rows, src, n_slices=256, seed=2)
print(f"vectors: {len(Xtr)} train / {len(Xva)} val, dim {Xtr.shape[1]}\n")

C = 37
report("mean only", Xtr, Xva, list(range(C)))
report("std only", Xtr, Xva, list(range(C, 2 * C)))
report("mean+std", Xtr, Xva, list(range(2 * C)))

# std in log space, scored at its own 6-bit reference
L = torch.cat([Xtr[:, :C], Xtr[:, C:].log()], 1)
Lv = torch.cat([Xva[:, :C], Xva[:, C:].log()], 1)
print()
report("mean+logstd", L, Lv, list(range(2 * C)))
report("logstd only", L, Lv, list(range(C, 2 * C)))
