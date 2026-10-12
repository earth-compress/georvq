"""Bit-plane structure for Sentinel-2 L1C, straight from the JP2s.

Targets: B03 (green, 10 m; cross-band prior from B02/B04/B08 at the
same grid) and B10 (cirrus, 60 m; spatial priors only -- its own
statistics are the point: thin-cirrus signal over a near-empty band).
Raw DNs are uint16 that do not fill 16 bits, so the marginal certifies
a lot by itself; the comparable number is structure BEYOND marginal.
"""
import glob
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import numpy as np
import rasterio
import torch
import torch.nn as nn

torch.set_num_threads(8)
BITS = 16
ROOT = "/mnt/raid/data/s2get/Sentinel-2/MSI/L1C/2026/05"


def granules():
    return sorted(glob.glob(f"{ROOT}/*/S2*.SAFE/GRANULE/*/IMG_DATA"))


def read_band(img_dir, band, scale=1):
    f = glob.glob(f"{img_dir}/*_{band}.jp2")[0]
    with rasterio.open(f) as ds:
        a = ds.read(1)
    return a[::scale, ::scale] if scale > 1 else a


def collect(img_dirs, bands, stride, xr=(0.0, 1.0)):
    """(target_code, nbr8, others, granule_stats) pooled over granules;
    bands[0] is the target. granule_stats = per-scene mean/std of the
    target band (the per-tile side info any real coder has -- without
    it, val scenes sit outside the training distribution and the
    certified bounds explode)."""
    ys, nbrs, others, gstats = [], [], [], []
    for d in img_dirs:
        arrs = [read_band(d, b) for b in bands]
        h, w = arrs[0].shape
        ii = np.arange(1, h - 1, stride)
        jj = np.arange(max(1, int(w * xr[0])),
                       min(w - 1, int(w * xr[1])), stride)
        y, x = np.meshgrid(ii, jj, indexing="ij")
        t = arrs[0]
        ok = t[y, x] > 0                       # 0 = L1C nodata
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ok &= t[y + dy, x + dx] > 0
        y, x = y[ok], x[ok]
        if len(y) == 0:
            continue
        vals = t[t > 0]
        gm, gs = float(vals.mean()), float(vals.std()) + 1e-6
        ys.append(t[y, x].astype(np.int64))
        nbrs.append((np.stack([t[y + dy, x + dx] for dy in (-1, 0, 1)
                               for dx in (-1, 0, 1) if (dy, dx) != (0, 0)],
                              -1).astype(np.float32) - gm) / gs)
        if len(bands) > 1:
            others.append((np.stack([a[y, x] for a in arrs[1:]], -1)
                           .astype(np.float32) - gm) / gs)
        gstats.append(np.tile([gm, gs], (len(y), 1)).astype(np.float32))
    return (np.concatenate(ys), np.concatenate(nbrs),
            np.concatenate(others) if others else None,
            np.concatenate(gstats))


def bits_of(y):
    b = np.zeros((len(y), BITS), dtype=np.float32)
    for k in range(BITS):
        b[:, k] = (y >> (BITS - 1 - k)) & 1
    return b


class BitNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.trunk = (nn.Sequential(nn.Linear(d_in, 256), nn.GELU(),
                                    nn.Linear(256, 256), nn.GELU(),
                                    nn.Linear(256, 128), nn.GELU())
                      if d_in else None)
        self.heads = nn.ModuleList(
            [nn.Linear(128 * (d_in > 0) + BITS, 1) for _ in range(BITS)])

    def forward(self, p, prev):
        h = self.trunk(p) if self.trunk is not None else None
        outs = []
        for k, head in enumerate(self.heads):
            ctx = prev.clone()
            ctx[:, k:] = 0
            outs.append(head(ctx if h is None
                             else torch.cat([h, ctx], -1)))
        return torch.cat(outs, -1)


def fit(Xtr, btr, Xva, bva):
    torch.manual_seed(0)
    net = BitNet(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt = torch.tensor(Xtr); Bt = torch.tensor(btr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(4):
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
        return (ce / len(Xv) / np.log(2)).numpy()


def norm(a):
    m, s = a.mean(0), a.std(0) + 1e-6
    return (a - m) / s


gs = granules()
print(f"{len(gs)} granules; WITHIN-scene split (left 58% train, "
      f"right 40% val, 2% guard) over 36 scenes")
gtr = gva = gs[:36]

for target, bands, stride in (("B03", ["B03", "B02", "B04", "B08"], 16),
                              ("B10", ["B10"], 5)):
    ytr, ntr, otr, gtr_s = collect(gtr, bands, stride, (0.0, 0.58))
    yva, nva, ova, gva_s = collect(gva, bands, stride, (0.60, 1.0))
    btr_, bva_ = bits_of(ytr), bits_of(yva)
    gsn = norm(np.concatenate([gtr_s, gva_s]))
    gtr_n, gva_n = gsn[:len(gtr_s)], gsn[len(gtr_s):]

    def wg(tr_a, va_a):
        return (np.concatenate([tr_a, gtr_n], 1).astype(np.float32),
                np.concatenate([va_a, gva_n], 1).astype(np.float32))

    priors = {"marginal": (np.zeros((len(ytr), 0), np.float32),
                           np.zeros((len(yva), 0), np.float32)),
              "left_px": wg(ntr[:, 3:4], nva[:, 3:4]),
              "nbr8": wg(ntr, nva)}
    if otr is not None:
        priors["others3"] = wg(otr, ova)
    print(f"\n=== S2 {target}: {len(ytr):,} train / {len(yva):,} val, "
          f"DN range {ytr.min()}..{ytr.max()} ===")
    Hm = None
    for pn, (Xtr, Xva) in priors.items():
        H = fit(Xtr, btr_, Xva, bva_)
        if pn == "marginal":
            Hm_planes = np.minimum(H, 1.0)
            H = Hm_planes
            Hm = H.sum()
        else:
            H = np.minimum(np.minimum(H, Hm_planes), 1.0)
        print(f"  {pn:>9}: H={H.sum():5.2f}/16 struct {16 - H.sum():5.2f} "
              f" vs-marginal {Hm - H.sum():5.2f} | "
              + " ".join(f"{h:.2f}" for h in H[:10]) + " ...")
