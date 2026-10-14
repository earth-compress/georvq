"""The codec as a bit-plane prior: era5_t64_scstd (60k, the de facto
era5 release candidate) decoded at three saturation rungs, each
reconstruction used as the prior for T500's code bits -- directly
comparable with the analytical priors (left_px, column, 3x3x3,
Fourier) on the same axes.

The prior per pixel is the reconstructed physical value at that pixel
(plus the target affine); rate per rung is the index bits + stats
channel. CPU decode of ~320 val windows.
"""
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import numpy as np
import torch
import torch.nn as nn

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import from_checkpoint

torch.set_num_threads(8)
LEVEL, BITS = 21, 16
RUNGS = [(20, 0, 0), (20, 20, 0), (20, 20, 20)]

m = from_checkpoint(
    "/home/ali/projects/georvq/.claude/worktrees/era5-model/"
    "checkpoints/era5_t64_scstd/model.pt", torch.device("cpu")).eval()
cfg = m.full_config
store = open_store(cfg.data.store_dir)
_, va_rows = store.split(cfg.data.val_frac, cfg.data.split_seed)
val = fixed_val_tokens(store, va_rows, token=64, max_tokens=480)
print(f"decoding {val.shape[0]} val windows at {RUNGS} (cpu)")

# True store codes via token provenance: fixed_val_tokens iterates
# va_rows.head(256) with a 4x4 grid of crops per chip (16 tokens each,
# capped at max_tokens), so token t came from row t // 16 -- and that
# row's cube gives the exact affine. A fixed quantization grid is NOT
# usable here: the dequantized values live on each cube's own lattice
# and re-gridding creates beat-pattern aliasing that reads as fake
# low-bit structure (measured: uniform H=0.41 on every deep plane).
import zarr
zr = zarr.open("/data/ERA5-samples/train/cubes.zarr", mode="r")
t_sc = np.asarray(zr["t_scale"]); t_off = np.asarray(zr["t_offset"])
prov = va_rows.head(256).iloc[
    np.arange(val.shape[0]) // 16].shard.to_numpy().astype(int)
px_per = 64 * 64
sc_px = np.repeat(t_sc[prov, LEVEL], px_per)
off_px = np.repeat(t_off[prov, LEVEL], px_per)
recons = {r: [] for r in RUNGS}
truths = []
with torch.no_grad():
    for i in range(0, val.shape[0], 16):
        x, v = m.source.prep(val[i:i + 16], torch.device("cpu"))
        st = m._stats_of(x, v)
        zs = m.encode_levels(x, st, v)
        for r in RUNGS:
            Ls = torch.tensor([list(r)] * x.shape[0])
            zqs, _, _ = m.quantize(zs, Ls)
            out = m.decode(zqs, Ls, st, v).float()
            phys = (out * m.source._sd + m.source._mu)[:, LEVEL]
            recons[r].append(phys.reshape(-1).numpy())
        truths.append(
            ((x * m.source._sd + m.source._mu)[:, LEVEL])
            .reshape(-1).numpy())
truth = np.concatenate(truths)
codes = np.rint((truth - off_px) / sc_px).astype(np.int64)
assert codes.min() >= 0 and codes.max() < 2 ** BITS, \
    (codes.min(), codes.max())
back = codes * sc_px + off_px
print(f"targets: {len(codes):,} pixels, store codes recovered exactly: "
      f"max roundtrip err {np.abs(back - truth).max():.2e} K")

n = len(codes)
tr = np.arange(n) % 3 != 0
vaM = ~tr


def bits_of(y):
    b = np.zeros((len(y), BITS), dtype=np.float32)
    for k in range(BITS):
        b[:, k] = (y >> (BITS - 1 - k)) & 1
    return b


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


bce = nn.BCEWithLogitsLoss(reduction="none")
b_all = bits_of(codes)
for r in RUNGS:
    p = np.concatenate(recons[r])
    feat = np.stack([(p - p.mean()) / (p.std() + 1e-6),
                     (sc_px - sc_px.mean()) / (sc_px.std() + 1e-6),
                     (off_px - off_px.mean()) / (off_px.std() + 1e-6)],
                    1).astype(np.float32)
    torch.manual_seed(0)
    net = BitNet(3)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt = torch.tensor(feat[tr]); Bt = torch.tensor(b_all[tr])
    for _ in range(4):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 16384):
            j = perm[i:i + 16384]
            loss = bce(net(Xt[j], Bt[j]), Bt[j]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Xv = torch.tensor(feat[vaM]); Bv = torch.tensor(b_all[vaM])
        ce = torch.zeros(BITS)
        for i in range(0, len(Xv), 65536):
            ce += bce(net(Xv[i:i + 65536], Bv[i:i + 65536]),
                      Bv[i:i + 65536]).sum(0)
        H = (ce / len(Xv) / np.log(2)).numpy()
    name = "-".join(map(str, r))
    print(f"codec {name}: H={H.sum():5.2f}/16 struct {16 - H.sum():5.2f}"
          f" | " + " ".join(f"{h:.2f}" for h in H))
