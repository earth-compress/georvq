"""Marginal purchase along the fill-order pareto: how many bits of
T500 structure each transmitted bit buys, rung by rung.

Decode era5_t64_scstd once into cumulative sums, evaluate ~18 path
rungs, convert each rung's T500 RMSE to Gaussian-equivalent structure
bits (validated against certified BitNet numbers to ~0.2 bits), and
report Delta(structure) / Delta(bits per value). The stats channel is
the first segment: its 444 bits/window buy the mean+std field before
any codebook fires. Bits per value are over the full 37-level field
(the bits pay for all levels; T500 is the measurement plane).
"""
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import numpy as np
import torch

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import from_checkpoint
from georvq.model.codec import tuple_bits
from georvq.model.stats_cond import patch_stats, uncenter

torch.set_num_threads(8)
LEVEL = 21
RUNGS = [(15, 0, 0), (17, 0, 0), (20, 0, 0), (20, 2, 0), (20, 5, 0),
         (20, 10, 0), (20, 15, 0), (20, 20, 0), (20, 20, 1), (20, 20, 2),
         (20, 20, 3), (20, 20, 5), (20, 20, 8), (20, 20, 12),
         (20, 20, 16), (20, 20, 20)]
VALUES = 37 * 64 * 64

m = from_checkpoint(
    "/home/ali/projects/georvq/.claude/worktrees/era5-model/"
    "checkpoints/era5_t64_scstd/model.pt", torch.device("cpu")).eval()
cfg = m.full_config
store = open_store(cfg.data.store_dir)
_, va_rows = store.split(cfg.data.val_frac, cfg.data.split_seed)
val = fixed_val_tokens(store, va_rows, token=64, max_tokens=480)
sb = m.side_bits()
print(f"{val.shape[0]} val windows; side channel {sb:.0f} bits/window")

import zarr
zr = zarr.open("/data/ERA5-samples/train/cubes.zarr", mode="r")
t_sc = np.asarray(zr["t_scale"])
prov = va_rows.head(256).iloc[
    np.arange(val.shape[0]) // 16].shard.to_numpy().astype(int)
sc_px = np.repeat(t_sc[prov, LEVEL], 64 * 64)

sq = {r: 0.0 for r in RUNGS}
sq_stats = 0.0
n_px = 0
with torch.no_grad():
    for i in range(0, val.shape[0], 16):
        x, v = m.source.prep(val[i:i + 16], torch.device("cpu"))
        st = m._stats_of(x, v)
        _, cums = m.level_cumsums(x, valid=v)
        truth = (x * m.source._sd + m.source._mu)[:, LEVEL]
        # stats-only reconstruction: zero latents through uncenter
        so = uncenter(torch.zeros_like(x), st, v)
        so_phys = (so * m.source._sd + m.source._mu)[:, LEVEL]
        sq_stats += float((so_phys - truth).pow(2).sum())
        n_px += truth.numel()
        for r in RUNGS:
            out = m.eval_tuple(cums, list(r), x, valid=v).float()
            phys = (out * m.source._sd + m.source._mu)[:, LEVEL]
            sq[r] += float((phys - truth).pow(2).sum())

med_lsb = float(np.median(sc_px))


def struct_bits(mse):
    sigma = np.sqrt(mse / n_px)
    h = np.log2(max(sigma / med_lsb, 1e-9) * np.sqrt(2 * np.e * np.pi))
    return 16 - h, sigma


rows = [("stats-only", sb, *struct_bits(sq_stats))]
for r in RUNGS:
    b = tuple_bits(m.specs, list(r), side_bits=sb)
    rows.append(("-".join(map(str, r)), b, *struct_bits(sq[r])))

print(f"\n{'rung':>11} {'b/value':>9} {'struct':>7} {'sigma_K':>8} "
      f"{'d_struct':>9} {'d_bits':>9} {'buy ratio':>9}")
prev_b, prev_s = 0.0, 0.0
out_rows = []
for name, bits, s, sig in rows:
    bpv = bits / VALUES
    db, ds = bpv - prev_b, s - prev_s
    ratio = ds / db if db > 0 else float("nan")
    print(f"{name:>11} {bpv:>9.4f} {s:>7.2f} {sig:>8.3f} "
          f"{ds:>9.3f} {db:>9.5f} {ratio:>9.1f}")
    out_rows.append((name, bpv, s, sig, ratio))
    prev_b, prev_s = bpv, s

np.save("/data/ERA5-samples/analysis_2026-09-04/rung_purchase.npy",
        np.array([(b, s, sig, r) for _, b, s, sig, r in out_rows]))
