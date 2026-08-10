"""Codebook health + gain/shape error decomposition for a stage-2 model.

Per tier and RVQ level:
  - usage (batch), perplexity, EMA-alive fraction
  - residual norm distribution (CV) -> is there gain dynamic range to code?
  - quantization error split into RADIAL (magnitude) vs TANGENTIAL
    (direction) components -> would gain-shape VQ actually buy error?
"""
import numpy as np
import torch

from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens

DEV = torch.device("cuda:1")
CKPT = "/home/ali/projects/ring-space/checkpoints/tessera_uf_vq8_s2/model.pt"

m = from_checkpoint(CKPT, DEV)
cfg = m.full_config
store = ChipStore(cfg.data.store_dir)
_, val_rows = store.split(cfg.data.val_frac, cfg.data.split_seed)
val_s = fixed_val_tokens(store, val_rows, token=cfg.model.window,
                         max_tokens=1024)
print(f"tiers: {[str(s) for s in m.specs]}")

acc = {}
with torch.no_grad():
    for i in range(0, val_s.shape[0], 64):
        x, _ = m.source.prep(val_s[i:i + 64], DEV)
        zs = m.encode_levels(x)
        for ti, (z, rvq) in enumerate(zip(zs, m.rvqs)):
            resid = z.reshape(-1, z.shape[-1]).float()
            for li, lev in enumerate(rvq.levels):
                E = lev.codebook.weight.float()
                d = (resid.pow(2).sum(1, keepdim=True)
                     - 2 * resid @ E.t() + E.pow(2).sum(1)[None])
                idx = d.argmin(1)
                q = E[idx]
                err = resid - q
                rn = resid.norm(dim=1)
                # radial vs tangential error split
                u = resid / rn.clamp_min(1e-9)[:, None]
                rad = (err * u).sum(1)                     # along resid dir
                tan2 = (err.pow(2).sum(1) - rad.pow(2)).clamp_min(0)
                a = acc.setdefault((ti, li), {
                    "cnt": torch.zeros(lev.num_codes, device=DEV),
                    "rn": [], "rad2": 0.0, "tan2": 0.0, "err2": 0.0,
                    "n": 0, "alive": lev.ema_usage()})
                a["cnt"] += torch.bincount(idx, minlength=lev.num_codes).float()
                a["rn"].append(rn[::17].cpu())
                a["rad2"] += float(rad.pow(2).sum())
                a["tan2"] += float(tan2.sum())
                a["err2"] += float(err.pow(2).sum())
                a["n"] += rn.numel()
                resid = resid - q

print(f"\n{'tier':>4} {'lvl':>3} {'used%':>6} {'perplex':>8} {'alive%':>7} "
      f"{'|r| CV':>7} {'radial%':>8}")
for (ti, li), a in sorted(acc.items()):
    K = a["cnt"].numel()
    p = a["cnt"] / a["cnt"].sum()
    perp = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
    used = float((a["cnt"] > 0).float().mean()) * 100
    rn = torch.cat(a["rn"])
    cv = float(rn.std() / rn.mean().clamp_min(1e-9))
    radial = a["rad2"] / max(a["err2"], 1e-9) * 100
    print(f"{ti:>4} {li:>3} {used:>6.1f} {perp:>8.0f} {a['alive']*100:>7.1f} "
          f"{cv:>7.3f} {radial:>8.1f}")

# summary per tier
print("\nper-tier totals:")
for ti in sorted({t for t, _ in acc}):
    ls = [a for (t, l), a in acc.items() if t == ti]
    perp = np.mean([float(torch.exp(-((a["cnt"]/a["cnt"].sum()) *
                   (a["cnt"]/a["cnt"].sum()).clamp_min(1e-12).log()).sum()))
                   for a in ls])
    K = ls[0]["cnt"].numel()
    print(f"  tier {ti}: K={K}, mean perplexity {perp:.0f} "
          f"({perp/K*100:.1f}% of K), mean alive "
          f"{np.mean([a['alive'] for a in ls])*100:.1f}%")
