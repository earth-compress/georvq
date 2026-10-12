"""Bit-plane structure of T500 under the full 3D-neighbourhood prior.

The strongest local prior: all 26 neighbours of a voxel in the 3x3x3
cube around it -- the 8 same-level lateral/diagonal neighbours (0.25
deg spacing) and all 9 voxels on the levels above (450 hPa) and below
(550 hPa) -- in physical units, with the target's affine appended.
Same 16 teacher-forced bit heads as the earlier probes, so the
structure numbers are directly comparable; also reports a plain
regression RMSE in K from the same trunk.
"""
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(8)
LEVEL = 21              # 500 hPa; neighbours at index 20 (450), 22 (550)
BITS = 16


def draw(split, n_cubes, seed):
    import zarr
    r = zarr.open(f"/data/ERA5-samples/{split}/cubes.zarr", mode="r")
    q = r["t"]
    sc = np.asarray(r["t_scale"]); off = np.asarray(r["t_offset"])
    rng = np.random.default_rng(seed)
    codes, feats, affs = [], [], []
    for c in rng.choice(q.shape[0], n_cubes, replace=False):
        x = np.asarray(q[c, LEVEL - 1:LEVEL + 2, ::24]).astype(np.int64) \
            + 32768                                    # (3, T, y, x)
        phys = (x * sc[c, LEVEL - 1:LEVEL + 2, None, None, None]
                + off[c, LEVEL - 1:LEVEL + 2, None, None, None]
                ).astype(np.float32)
        T = phys.shape[1]
        for t in range(T):
            p = phys[:, t]                             # (3, 128, 128)
            ctr_code = (x[1, t, 1:125:2, 1:125:2]).reshape(-1)
            patch = np.stack([p[dz, 1 + dy:125 + dy:2, 1 + dx:125 + dx:2]
                              for dz in range(3)
                              for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
                             -1).reshape(-1, 27)
            keep = np.ones(27, bool); keep[13] = False  # drop the centre
            codes.append(ctr_code)
            feats.append(patch[:, keep])
            affs.append(np.tile([sc[c, LEVEL], off[c, LEVEL]],
                                (len(ctr_code), 1)).astype(np.float32))
    return (np.concatenate(codes), np.concatenate(feats),
            np.concatenate(affs))


ytr, ftr, atr = draw("train", 80, 1)
yva, fva, ava = draw("val", 40, 2)
print(f"{len(ytr):,} train / {len(yva):,} val voxels, 26 neighbours each")

mu_f, sd_f = ftr.mean(0), ftr.std(0) + 1e-6
mu_a, sd_a = atr.mean(0), atr.std(0) + 1e-6
Xtr = np.concatenate([(ftr - mu_f) / sd_f, (atr - mu_a) / sd_a],
                     1).astype(np.float32)
Xva = np.concatenate([(fva - mu_f) / sd_f, (ava - mu_a) / sd_a],
                     1).astype(np.float32)


def bits_of(y):
    b = np.zeros((len(y), BITS), dtype=np.float32)
    for k in range(BITS):
        b[:, k] = (y >> (BITS - 1 - k)) & 1
    return b


btr, bva = bits_of(ytr), bits_of(yva)


class BitNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(),
                                   nn.Linear(256, 256), nn.GELU(),
                                   nn.Linear(256, 128), nn.GELU())
        self.heads = nn.ModuleList(
            [nn.Linear(128 + BITS, 1) for _ in range(BITS)])
        self.reg = nn.Linear(128, 1)     # plain regression readout

    def forward(self, p, prev):
        h = self.trunk(p)
        outs = []
        for k, head in enumerate(self.heads):
            ctx = prev.clone()
            ctx[:, k:] = 0
            outs.append(head(torch.cat([h, ctx], -1)))
        return torch.cat(outs, -1), self.reg(h).squeeze(-1)


torch.manual_seed(0)
net = BitNet(Xtr.shape[1])
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
Xt = torch.tensor(Xtr); Bt = torch.tensor(btr)
phys_tr = torch.tensor((ytr * atr[:, 0] + atr[:, 1]).astype(np.float32))
phys_va = torch.tensor((yva * ava[:, 0] + ava[:, 1]).astype(np.float32))
pm, ps = phys_tr.mean(), phys_tr.std()
bce = nn.BCEWithLogitsLoss(reduction="none")
for epoch in range(8):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 8192):
        j = perm[i:i + 8192]
        logits, reg = net(Xt[j], Bt[j])
        loss = bce(logits, Bt[j]).mean() \
            + 0.1 * ((reg - (phys_tr[j] - pm) / ps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    Xv = torch.tensor(Xva); Bv = torch.tensor(bva)
    ce = torch.zeros(BITS); se = 0.0
    for i in range(0, len(Xv), 65536):
        logits, reg = net(Xv[i:i + 65536], Bv[i:i + 65536])
        ce += bce(logits, Bv[i:i + 65536]).sum(0)
        se += float(((reg * ps + pm) - phys_va[i:i + 65536]).pow(2).sum())
    H = (ce / len(Xv) / np.log(2)).numpy()
    rmse = (se / len(Xv)) ** 0.5

print(f"\nnbr3d_26: H(value|prior) = {H.sum():.2f} of 16 bits "
      f"(structure {16 - H.sum():.2f} bits)")
print("  bit(MSB->LSB): " + " ".join(f"{h:.2f}" for h in H))
print(f"  plain regression RMSE: {rmse:.3f} K")
print("\nreference (single-source priors): left_px 3.02, column36 2.04 bits")
