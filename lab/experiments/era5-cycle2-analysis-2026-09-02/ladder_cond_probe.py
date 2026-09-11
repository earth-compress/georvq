"""Is the ladder's per-tier depth conditioning doing work?

On the era5_t64_ladder checkpoint: (1) the fine-depth rung curve
around the mid->fine exit handover, (2) the same 20-20-1 latents
decoded with honest conditioning (L2=1) and with a lie (L2=20). A gap
between honest and lying decodes means the thermometer gates; no gap
means the conditioning path is dead weight.
"""
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import torch
from georvq.data.pool import fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import from_checkpoint
from georvq.metrics import masked_r2

dev = torch.device("cuda:0")
torch.set_grad_enabled(False)
m = from_checkpoint(
    "/home/ali/projects/georvq/.claude/worktrees/era5-model/"
    "checkpoints/era5_t64_ladder/model.pt", dev).eval()
cfg = m.full_config
store = open_store(cfg.data.store_dir)
_, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
val = fixed_val_tokens(store, va, token=cfg.model.window,
                       max_tokens=1024).to(dev)

def score(tuples, lie=None):
    num = {t: [0.0, 0] for t in tuples}
    for i in range(0, val.shape[0], 64):
        x, v = m.source.prep(val[i:i + 64], dev)
        st = m._stats_of(x, v)
        zs = m.encode_levels(x, st, v)
        for t in tuples:
            Ls = torch.tensor(list(t), device=dev)[None] \
                .expand(x.shape[0], -1)
            zqs, _, _ = m.quantize(zs, Ls)
            cond = Ls if lie is None else \
                torch.tensor(lie, device=dev)[None].expand(x.shape[0], -1)
            out = m.decode(zqs, cond, st, v).float()
            n = int(v.sum())
            num[t][0] += float(masked_r2(out, x, v)) * n
            num[t][1] += n
    return {t: s / n for t, (s, n) in num.items()}

rungs = [(20, 20, 0), (20, 20, 1), (20, 20, 2), (20, 20, 3), (20, 20, 5),
         (20, 20, 10), (20, 20, 20)]
r = score(rungs)
print("fine-depth rung curve (val R2):")
for t in rungs:
    print(f"  {'-'.join(map(str, t))}: {r[t]:.5f}")

honest = score([(20, 20, 1)])[(20, 20, 1)]
lied = score([(20, 20, 1)], lie=(20, 20, 20))[(20, 20, 1)]
print(f"\n20-20-1 latents, honest cond (L2=1):  {honest:.5f}")
print(f"20-20-1 latents, lying cond (L2=20):  {lied:.5f}")
print(f"conditioning effect: {honest - lied:+.5f} "
      "(positive = the thermometer is gating)")
