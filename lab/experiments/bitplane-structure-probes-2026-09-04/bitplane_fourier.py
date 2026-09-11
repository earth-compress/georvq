"""Bit-plane structure of T500 under low-pass Fourier predictors.

Keep the 2D FFT coefficients of each 128px slice inside radius r
cycles/window, invert, and use the low-passed value at the pixel as
the prior (plus the target affine), through the same 16 teacher-forced
bit heads as the other probes. Sweeping r shows which bit-planes live
at which spatial frequencies -- an idealized version of what the
pyramid's coarse tiers transmit. Coefficient count ~ pi r^2 is the
rate of the truncation itself, so the sweep is also a classical
transform-coding R-D baseline in bit-plane form.
"""
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(8)
LEVEL, BITS = 21, 16
RADII = [1, 2, 4, 8, 16, 32]

fy = np.fft.fftfreq(128) * 128
fx = np.fft.rfftfreq(128) * 128
RAD = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def draw(split, n_cubes, seed):
    import zarr
    r = zarr.open(f"/data/ERA5-samples/{split}/cubes.zarr", mode="r")
    q = r["t"]
    sc = np.asarray(r["t_scale"]); off = np.asarray(r["t_offset"])
    rng = np.random.default_rng(seed)
    codes, preds, affs = [], {rr: [] for rr in RADII}, []
    for c in rng.choice(q.shape[0], n_cubes, replace=False):
        x = np.asarray(q[c, LEVEL, ::24]).astype(np.int64) + 32768  # (T,y,x)
        phys = (x * sc[c, LEVEL] + off[c, LEVEL]).astype(np.float32)
        F = np.fft.rfft2(phys, axes=(-2, -1))
        for rr in RADII:
            low = np.fft.irfft2(F * (RAD <= rr), s=(128, 128),
                                axes=(-2, -1)).astype(np.float32)
            preds[rr].append(low[:, ::2, ::2].reshape(-1))
        codes.append(x[:, ::2, ::2].reshape(-1))
        n = x[:, ::2, ::2].size
        affs.append(np.tile([sc[c, LEVEL], off[c, LEVEL]],
                            (n, 1)).astype(np.float32))
    return (np.concatenate(codes),
            {rr: np.concatenate(v) for rr, v in preds.items()},
            np.concatenate(affs))


ytr, ptr, atr = draw("train", 80, 1)
yva, pva, ava = draw("val", 40, 2)
print(f"{len(ytr):,} train / {len(yva):,} val pixels; radii {RADII}")

mu_a, sd_a = atr.mean(0), atr.std(0) + 1e-6


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

    def forward(self, p, prev):
        h = self.trunk(p)
        outs = []
        for k, head in enumerate(self.heads):
            ctx = prev.clone()
            ctx[:, k:] = 0
            outs.append(head(torch.cat([h, ctx], -1)))
        return torch.cat(outs, -1)


def run(rr):
    m, s = ptr[rr].mean(), ptr[rr].std() + 1e-6
    Xtr = np.stack([(ptr[rr] - m) / s,
                    (atr[:, 0] - mu_a[0]) / sd_a[0],
                    (atr[:, 1] - mu_a[1]) / sd_a[1]], 1).astype(np.float32)
    Xva = np.stack([(pva[rr] - m) / s,
                    (ava[:, 0] - mu_a[0]) / sd_a[0],
                    (ava[:, 1] - mu_a[1]) / sd_a[1]], 1).astype(np.float32)
    torch.manual_seed(0)
    net = BitNet(3)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt = torch.tensor(Xtr); Bt = torch.tensor(btr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for epoch in range(4):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 16384):
            j = perm[i:i + 16384]
            loss = bce(net(Xt[j], Bt[j]), Bt[j]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Xv = torch.tensor(Xva); Bv = torch.tensor(bva)
        ce = torch.zeros(BITS)
        for i in range(0, len(Xv), 65536):
            ce += bce(net(Xv[i:i + 65536], Bv[i:i + 65536]),
                      Bv[i:i + 65536]).sum(0)
        H = (ce / len(Xv) / np.log(2)).numpy()
    ncoef = int((RAD <= rr).sum())
    print(f"r<={rr:>2} ({ncoef:>4} coefs): structure "
          f"{16 - H.sum():5.2f} bits | " +
          " ".join(f"{h:.2f}" for h in H[:8]) + " ...")
    return H


print("\nbit(MSB->LSB), first 8 planes shown:")
for rr in RADII:
    run(rr)
print("\nreference: left_px 3.02 | column36 2.04 | 3x3x3 3.54 bits")
