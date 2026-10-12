"""Marginal purchase curves for the tessera and AE release bundles.

Same construction as the era5 rung sweep, but the purchase ratio is
computed from sigma-halvings (Delta structure = log2(sigma_prev /
sigma_new)), which needs no LSB reference -- absolute bit offsets
cancel in differences, so AE's signed-square companding and tessera's
per-pixel scales don't enter. RMSE pooled over all bands in each
model's own prep space; bits per value over the full band stack.

Usage: bitplane_rungs_multi.py {tessera|ae}
"""
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import numpy as np
import torch

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import from_checkpoint
from georvq.model.codec import tuple_bits
from georvq.model.stats_cond import uncenter

torch.set_num_threads(8)
WHICH = sys.argv[1]

if WHICH == "tessera":
    CKPT = "/data/georvq-releases/tessera-beta-v1/bundle/model.pt"
    RUNGS = [(4, 0, 0), (10, 0, 0), (20, 0, 0), (20, 3, 0), (20, 7, 0),
             (20, 13, 0), (20, 20, 0), (20, 20, 1), (20, 20, 3),
             (20, 20, 7), (20, 20, 13), (20, 20, 20)]
else:
    CKPT = "/data/georvq-releases/ae-beta-v1/bundle/model.pt"
    RUNGS = [(6, 1, 0), (8, 1, 0), (8, 4, 0), (8, 8, 0), (8, 8, 2),
             (8, 8, 4), (8, 8, 8)]

m = from_checkpoint(CKPT, torch.device("cpu")).float().eval()
cfg = m.full_config
store = open_store(cfg.data.store_dir)
_, va_rows = store.split(cfg.data.val_frac, cfg.data.split_seed)
val = fixed_val_tokens(store, va_rows, token=cfg.model.window,
                       max_tokens=256)
sb = m.side_bits()
VALUES = cfg.model.bands * cfg.model.window ** 2
print(f"{WHICH}: {val.shape[0]} windows, side {sb:.0f} bits, "
      f"{VALUES} values/window")

sq = {r: 0.0 for r in RUNGS}
sq_stats, n_val = 0.0, 0
with torch.no_grad():
    for i in range(0, val.shape[0], 8):
        x, v = m.source.prep(val[i:i + 8], torch.device("cpu"))
        st = m._stats_of(x, v)
        _, cums = m.level_cumsums(x, valid=v)
        mask = v.unsqueeze(1).expand_as(x)
        n_val += int(mask.sum())
        if m.stats_cond is not None:
            so = uncenter(torch.zeros_like(x), st, v)
            sq_stats += float(((so - x) * mask).pow(2).sum())
        for r in RUNGS:
            out = m.eval_tuple(cums, list(r), x, valid=v).float()
            sq[r] += float(((out - x) * mask).pow(2).sum())

rows = []
if m.stats_cond is not None:
    rows.append(("stats-only", sb, np.sqrt(sq_stats / n_val)))
for r in RUNGS:
    rows.append(("-".join(map(str, r)),
                 tuple_bits(m.specs, list(r), side_bits=sb),
                 np.sqrt(sq[r] / n_val)))

print(f"\n{'rung':>11} {'b/value':>9} {'sigma':>9} {'d_halvings':>11} "
      f"{'d_bits':>9} {'buy ratio':>9}")
prev_b, prev_s = 0.0, None
out = []
for name, bits, s in rows:
    bpv = bits / VALUES
    dh = (np.log2(prev_s / s) if prev_s else float("nan"))
    db = bpv - prev_b
    ratio = dh / db if (db > 0 and prev_s) else float("nan")
    print(f"{name:>11} {bpv:>9.4f} {s:>9.4f} {dh:>11.3f} "
          f"{db:>9.5f} {ratio:>9.1f}")
    out.append((bpv, s, dh, ratio))
    prev_b, prev_s = bpv, s
np.save(f"/data/ERA5-samples/analysis_2026-09-04/rungs_{WHICH}.npy",
        np.array(out))
