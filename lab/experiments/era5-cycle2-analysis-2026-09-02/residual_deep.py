"""Residual structure of the converged era5_t64_scstd model (60k), on
4096 held-out windows: spatial whiteness (autocorrelation vs source),
cross-level PCA of the per-pixel residual vectors, and the level
profiles of the leading residual eigenvectors -- the "trends" question:
does the error concentrate on particular pressure levels or vertical
patterns, or is it unstructured?

Scratch extension of analysis/residual_struct.py (which stays as-is).
"""
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import numpy as np
import torch
from georvq.data.pool import fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import from_checkpoint

LEVELS = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
          225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
          775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]

dev = torch.device("cuda:0")
torch.set_grad_enabled(False)
m = from_checkpoint(
    "/home/ali/projects/georvq/.claude/worktrees/era5-model/"
    "checkpoints/era5_t64_scstd/model.pt", dev).eval()
cfg = m.full_config
store = open_store(cfg.data.store_dir)
_, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
val = fixed_val_tokens(store, va, token=cfg.model.window,
                       max_tokens=4096).to(dev)
print(f"{val.shape[0]} held-out windows, model step 60000 (full depth)")

X, R = [], []
for i in range(0, val.shape[0], 32):
    x, v = m.source.prep(val[i:i + 32], dev)
    st = m._stats_of(x, v)
    zs = m.encode_levels(x, st, v)
    Ls = torch.tensor([[20, 20, 20]], device=dev).expand(x.shape[0], -1)
    zqs, _, _ = m.quantize(zs, Ls)
    out = m.decode(zqs, Ls, st, v).float()
    X.append(x.float().cpu()); R.append((x.float() - out).cpu())
X, R = torch.cat(X), torch.cat(R)
print(f"residual energy: {100*float(R.pow(2).mean()/X.pow(2).mean()):.3f}% "
      "of source (standardized space)")

def autocorr(t, lags):
    x = t - t.mean(dim=(0, 2, 3), keepdim=True)
    den = x.pow(2).mean(dim=(0, 2, 3)).clamp_min(1e-12)
    out = {}
    for d in lags:
        h = (x[:, :, :, :-d] * x[:, :, :, d:]).mean(dim=(0, 2, 3))
        vv = (x[:, :, :-d, :] * x[:, :, d:, :]).mean(dim=(0, 2, 3))
        out[d] = float((0.5 * (h + vv) / den).mean())
    return out

lags = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
ar = autocorr(R.to(dev), lags)
ax = autocorr(X.to(dev), lags)
print("\nspatial autocorrelation (mean over levels):")
print("  lag     " + " ".join(f"{d:>6}" for d in lags))
print("  resid   " + " ".join(f"{ar[d]:>6.3f}" for d in lags))
print("  source  " + " ".join(f"{ax[d]:>6.3f}" for d in lags))

# per-level residual share
per_lvl = R.pow(2).mean(dim=(0, 2, 3))
top = per_lvl / per_lvl.sum()
print("\nper-level share of residual energy (top 8):")
for i in torch.argsort(top, descending=True)[:8]:
    print(f"  {LEVELS[i]:>5} hPa: {100*float(top[i]):.1f}%")

# where does the error live, relative to the std13 set? Interior levels
# are interpolation targets; the 8 levels above 50 hPa are extrapolation
# into the semi-decoupled stratosphere.
import numpy as _np
idx13 = [8, 10, 12, 14, 16, 17, 19, 21, 23, 25, 30, 33, 36]
strat = list(range(0, 8))                       # 1..30 hPa
interior = [i for i in range(37) if i not in idx13 and i not in strat]
src_lvl = X.pow(2).mean(dim=(0, 2, 3))
print("\nresidual grouped by level class (share of residual energy /"
      " share of source energy / group R2):")
for name, ii in (("std13", idx13), ("interior16", interior),
                 ("strat8", strat)):
    ii = torch.tensor(ii)
    r2 = 1 - float(per_lvl[ii].sum() / src_lvl[ii].sum())
    print(f"  {name:>10}: {100*float(top[ii].sum()):5.1f}% resid | "
          f"{100*float(src_lvl[ii].sum()/src_lvl.sum()):5.1f}% source | "
          f"R2 {r2:.5f}")

# cross-level PCA of per-pixel residual vectors
r = R.permute(0, 2, 3, 1).reshape(-1, R.shape[1])
r = r[torch.randperm(r.shape[0])[:2_000_000]].to(dev).double()
rc = r - r.mean(0)
C = (rc.T @ rc) / rc.shape[0]
ev, vec = torch.linalg.eigh(C)
ev, vec = ev.flip(0), vec.flip(1)
share = ev / ev.sum()
cum = torch.cumsum(share, 0)
d95 = int((cum < 0.95).sum()) + 1
print(f"\nresidual PCA: d95 = {d95}/37; top-5 eigenvalue shares: "
      + " ".join(f"{100*float(s):.1f}%" for s in share[:5]))
print("\nleading eigenvector level profiles (loading per level):")
for k in range(3):
    prof = vec[:, k].cpu().numpy()
    if prof[np.abs(prof).argmax()] < 0:
        prof = -prof
    tops = np.argsort(-np.abs(prof))[:5]
    print(f"  PC{k+1} ({100*float(share[k]):.1f}%): peaks at "
          + ", ".join(f"{LEVELS[i]} hPa ({prof[i]:+.2f})" for i in tops))
