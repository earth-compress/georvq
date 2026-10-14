"""Bit-plane conditional-entropy profile: structure vs noise, per bit.

For T at 500 hPa in the store's own int16 code (per-cube affine, so a
bit is a per-cube physical size), train one 16-headed MLP per prior;
head k predicts bit k (MSB first) given the prior features and the
TRUE more-significant bits (teacher forcing). Per-plane cross-entropy
in bits then sums exactly to H(value | prior): the area under the
curve of (1 - H_k) is the structure that prior certifies, the area
above is noise relative to it. Every prior is a lower bound on
structure; the curves only move up as priors improve.

Priors: marginal (MSB context only), left-neighbour pixel (same
level), PCA-8 of the other 36 levels, the full 36-level column.
"""
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(8)
LEVEL = 21              # 500 hPa in the ascending 37-level axis
BITS = 16


def draw(split, n_cubes, seed):
    """Targets stay CODE bits; priors are PHYSICAL (dequantized), with
    the target level's scale/offset appended so a model can map physics
    back to code space -- cross-level priors in raw code space are
    scrambled by the per-(cube, level) affines."""
    import zarr
    r = zarr.open(f"/data/ERA5-samples/{split}/cubes.zarr", mode="r")
    q = r["t"]
    sc = np.asarray(r["t_scale"]); off = np.asarray(r["t_offset"])
    rng = np.random.default_rng(seed)
    codes, nbrs, cols, affs = [], [], [], []
    for c in rng.choice(q.shape[0], n_cubes, replace=False):
        x = np.asarray(q[c, :, ::48]).astype(np.int64) + 32768  # (37,T,y,x)
        phys = (x * sc[c][:, None, None, None]
                + off[c][:, None, None, None]).astype(np.float32)
        t = x[LEVEL]
        n = t[:, :, 1::4].size
        codes.append(t[:, :, 1::4].reshape(-1))
        pn = phys[LEVEL]
        nbrs.append(pn[:, :, 0::4][:, :, :pn.shape[2] // 4].reshape(-1))
        others = np.delete(phys, LEVEL, axis=0)[:, :, :, 1::4]
        cols.append(others.reshape(36, -1).T)
        affs.append(np.tile([sc[c, LEVEL], off[c, LEVEL]],
                            (n, 1)).astype(np.float32))
    return (np.concatenate(codes), np.concatenate(nbrs),
            np.concatenate(cols), np.concatenate(affs))


ytr, ntr, ctr, atr = draw("train", 100, 1)
yva, nva, cva, ava = draw("val", 50, 2)
print(f"{len(ytr):,} train / {len(yva):,} val pixels (physical priors)")

mu_c, sd_c = ctr.mean(0), ctr.std(0) + 1e-6
mu_n, sd_n = ntr.mean(), ntr.std() + 1e-6
mu_a, sd_a = atr.mean(0), atr.std(0) + 1e-6
aff_tr = (atr - mu_a) / sd_a
aff_va = (ava - mu_a) / sd_a
cc = (ctr - mu_c) / sd_c
_, _, Vt = np.linalg.svd(cc[:200000], full_matrices=False)
P8 = Vt[:8].T


def with_aff(tr, va):
    return (np.concatenate([tr, aff_tr], 1).astype(np.float32),
            np.concatenate([va, aff_va], 1).astype(np.float32))


PRIORS = {
    "marginal": (np.zeros((len(ytr), 0), np.float32),
                 np.zeros((len(yva), 0), np.float32)),
    "left_px": with_aff(((ntr - mu_n) / sd_n)[:, None],
                        ((nva - mu_n) / sd_n)[:, None]),
    "pca8_col": with_aff(cc @ P8, ((cva - mu_c) / sd_c) @ P8),
    "column36": with_aff(cc, (cva - mu_c) / sd_c),
}


def bits_of(y):
    b = np.zeros((len(y), BITS), dtype=np.float32)
    for k in range(BITS):
        b[:, k] = (y >> (BITS - 1 - k)) & 1
    return b


btr, bva = bits_of(ytr), bits_of(yva)


class BitNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.trunk = (nn.Sequential(nn.Linear(d_in, 256), nn.GELU(),
                                    nn.Linear(256, 256), nn.GELU(),
                                    nn.Linear(256, 128), nn.GELU())
                      if d_in else None)
        self.heads = nn.ModuleList(
            [nn.Linear(128 * (d_in > 0) + BITS, 1) for _ in range(BITS)])

    def forward(self, p, prev):                 # prev: (B, BITS) padded
        h = self.trunk(p) if self.trunk is not None else None
        outs = []
        for k, head in enumerate(self.heads):
            ctx = prev.clone()
            ctx[:, k:] = 0                       # only MORE significant bits
            inp = ctx if h is None else torch.cat([h, ctx], -1)
            outs.append(head(inp))
        return torch.cat(outs, -1)               # (B, BITS) logits


def run(name, Xtr, Xva):
    torch.manual_seed(0)
    net = BitNet(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Bt = torch.tensor(btr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for epoch in range(3):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 8192):
            j = perm[i:i + 8192]
            logits = net(Xt[j], Bt[j])
            loss = bce(logits, Bt[j]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Xv = torch.tensor(Xva, dtype=torch.float32)
        Bv = torch.tensor(bva)
        ce = torch.zeros(BITS)
        for i in range(0, len(Xv), 65536):
            ce += bce(net(Xv[i:i + 65536], Bv[i:i + 65536]),
                      Bv[i:i + 65536]).sum(0)
        H = (ce / len(Xv) / np.log(2)).numpy()   # bits per plane
    print(f"\n{name}: H(value|prior) = {H.sum():.2f} of 16 bits "
          f"(structure {16 - H.sum():.2f} bits)")
    print("  bit(MSB->LSB): " + " ".join(f"{h:.2f}" for h in H))
    return H


results = {n: run(n, a, b) for n, (a, b) in PRIORS.items()}
print("\nsummary (predictable bits of 16):")
for n, H in results.items():
    print(f"  {n:>10}: {16 - H.sum():5.2f}")
