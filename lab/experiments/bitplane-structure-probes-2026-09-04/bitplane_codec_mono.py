"""Bit-plane codec priors, mono vs column: the single-level t model
(era5_t64_mono_ladder @20k) and the 37-channel reference
(era5_t64_scstd @60k) decoded at three saturation rungs on the SAME
val T500 pixels, each reconstruction scored as a bit-plane prior with
the same BitNet as the analytical-prior probes. Saves all H arrays to
bitplane_codec_results.npz for plotting.
"""
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import json

import numpy as np
import torch
import torch.nn as nn

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import from_checkpoint

torch.set_num_threads(8)
LEVEL, BITS = 21, 16
RUNGS = [(20, 0, 0), (20, 20, 0), (20, 20, 20)]
CK = "/home/ali/projects/georvq/.claude/worktrees/era5-model/checkpoints"

m37 = from_checkpoint(f"{CK}/era5_t64_scstd/model.pt",
                      torch.device("cpu")).eval()
mono = from_checkpoint(f"{CK}/era5_t64_mono_ladder/model.pt",
                       torch.device("cpu")).eval()
print("mono checkpoint step:",
      torch.load(f"{CK}/era5_t64_mono_ladder/model.pt",
                 map_location="cpu", weights_only=False)["step"])

cfg = m37.full_config
store = open_store(cfg.data.store_dir)
_, va_rows = store.split(cfg.data.val_frac, cfg.data.split_seed)
val = fixed_val_tokens(store, va_rows, token=64, max_tokens=480)
print(f"decoding {val.shape[0]} val windows at {RUNGS} (cpu)")

import zarr
zr = zarr.open("/data/ERA5-samples/train/cubes.zarr", mode="r")
t_sc = np.asarray(zr["t_scale"]); t_off = np.asarray(zr["t_offset"])
prov = va_rows.head(256).iloc[
    np.arange(val.shape[0]) // 16].shard.to_numpy().astype(int)
px_per = 64 * 64
sc_px = np.repeat(t_sc[prov, LEVEL], px_per)
off_px = np.repeat(t_off[prov, LEVEL], px_per)

stats = json.load(open(
    "/home/ali/projects/georvq/.claude/worktrees/era5-model/"
    "georvq/sources/era5_stats.json"))["t"]
mu21, sd21 = stats["mean"][LEVEL], stats["std"][LEVEL]

recons = {("col",) + r: [] for r in RUNGS}
recons.update({("mono",) + r: [] for r in RUNGS})
truths = []
with torch.no_grad():
    for i in range(0, val.shape[0], 16):
        x, v = m37.source.prep(val[i:i + 16], torch.device("cpu"))
        st = m37._stats_of(x, v)
        zs = m37.encode_levels(x, st, v)
        for r in RUNGS:
            Ls = torch.tensor([list(r)] * x.shape[0])
            zqs, _, _ = m37.quantize(zs, Ls)
            out = m37.decode(zqs, Ls, st, v).float()
            phys = (out * m37.source._sd + m37.source._mu)[:, LEVEL]
            recons[("col",) + r].append(phys.reshape(-1).numpy())
        truth_p = (x * m37.source._sd + m37.source._mu)[:, LEVEL]
        truths.append(truth_p.reshape(-1).numpy())
        # mono: the same physical plane, standardized with the level's
        # own moments (the single-level store's model space)
        xm = ((truth_p - mu21) / sd21).unsqueeze(1)
        vm = torch.ones(xm.shape[0], 64, 64, dtype=torch.bool)
        stm = mono._stats_of(xm, vm)
        zsm = mono.encode_levels(xm, stm, vm)
        for r in RUNGS:
            Ls = torch.tensor([list(r)] * xm.shape[0])
            zqs, _, _ = mono.quantize(zsm, Ls)
            out = mono.decode(zqs, Ls, stm, vm).float()
            phys = out[:, 0] * sd21 + mu21
            recons[("mono",) + r].append(phys.reshape(-1).numpy())
        print(f"  {i + 16}/{val.shape[0]}", flush=True)

truth = np.concatenate(truths)
codes = np.rint((truth - off_px) / sc_px).astype(np.int64)
assert codes.min() >= 0 and codes.max() < 2 ** BITS
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
results = {}
for key, chunks in recons.items():
    p = np.concatenate(chunks)
    rmse = float(np.sqrt(np.mean((p - truth) ** 2)))
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
    name = key[0] + "-" + "-".join(map(str, key[1:]))
    results[name] = H
    results[name + "_rmse"] = np.array([rmse])
    print(f"{name}: H={H.sum():5.2f}/16 struct {16 - H.sum():5.2f} "
          f"rmse {rmse:.3f} K | "
          + " ".join(f"{h:.2f}" for h in H), flush=True)

np.savez("/data/ERA5-samples/analysis_2026-09-04/"
         "bitplane_codec_results.npz", **results)
print("saved bitplane_codec_results.npz")
