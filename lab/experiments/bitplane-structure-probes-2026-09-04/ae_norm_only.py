"""Bit-plane structure probes for AlphaEarth and Tessera (int8 codes).

Same framework as the era5 probes: teacher-forced per-bit heads over
the stored code (8 planes here), priors of increasing strength. Two
target dims per source (median- and max-variance), because embedding
dims need not share statistics. AE is unit-norm, so the all-other-63
prior should certify nearly everything (a validation, not a finding);
the 32-band prior is the informative one there. Reported per prior:
H(code | prior) and structure BEYOND the marginal, which is the
cross-source comparable number.
"""
import sys

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import numpy as np
import torch
import torch.nn as nn

from georvq.data.store import open_store

torch.set_num_threads(8)
BITS = 8


def sample_chips(store, n, seed, tessera=False):
    rng = np.random.default_rng(seed)
    rows = store.index.sample(n=n, random_state=seed)
    out = []
    for sh, grp in rows.groupby("shard"):
        shard = store.open_shard(sh)
        for sl in grp.slot:
            a = np.asarray(shard[int(sl)]).astype(np.int16)  # (B, px, px)
            if tessera:
                sc = np.load(f"{store.dir}/scales_{int(sh):04d}.npy",
                             mmap_mode="r")[int(sl)]
                valid = (a != 0).any(0) & (sc > 0) & (sc < 1.0)
                out.append((a, np.asarray(sc, np.float32), valid))
            else:
                valid = ~(a == -128).all(0)
                out.append((a, None, valid))
    return out


def build(chips, dim, stride=6):
    """Rows: code bits of `dim` + priors from each chip's interior."""
    ys, nbr8, others, scs = [], [], [], []
    for a, sc, valid in chips:
        px = a.shape[1]
        ii = np.arange(1, px - 1, stride)
        y, x = np.meshgrid(ii, ii, indexing="ij")
        ok = valid[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ok &= valid[y + dy, x + dx]
        y, x = y[ok], x[ok]
        if len(y) == 0:
            continue
        ys.append((a[dim, y, x] + 128).astype(np.int64))
        nbr8.append(np.stack(
            [a[dim, y + dy, x + dx] for dy in (-1, 0, 1)
             for dx in (-1, 0, 1) if (dy, dx) != (0, 0)], -1
        ).astype(np.float32))
        others.append(np.delete(a[:, y, x], dim, axis=0).T
                      .astype(np.float32))
        scs.append((sc[y, x] if sc is not None
                    else np.zeros(len(y))).astype(np.float32))
    return (np.concatenate(ys), np.concatenate(nbr8),
            np.concatenate(others), np.concatenate(scs)[:, None])


class BitNet(nn.Module):
    def __init__(self, d_in, bits=BITS):
        super().__init__()
        self.bits = bits
        self.trunk = (nn.Sequential(nn.Linear(d_in, 256), nn.GELU(),
                                    nn.Linear(256, 256), nn.GELU(),
                                    nn.Linear(256, 128), nn.GELU())
                      if d_in else None)
        self.heads = nn.ModuleList(
            [nn.Linear(128 * (d_in > 0) + bits, 1) for _ in range(bits)])

    def forward(self, p, prev):
        h = self.trunk(p) if self.trunk is not None else None
        outs = []
        for k, head in enumerate(self.heads):
            ctx = prev.clone()
            ctx[:, k:] = 0
            outs.append(head(ctx if h is None
                             else torch.cat([h, ctx], -1)))
        return torch.cat(outs, -1)


def bits_of(y, bits=BITS):
    b = np.zeros((len(y), bits), dtype=np.float32)
    for k in range(bits):
        b[:, k] = (y >> (bits - 1 - k)) & 1
    return b


def fit(Xtr, btr, Xva, bva, epochs=4):
    torch.manual_seed(0)
    net = BitNet(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr); Bt = torch.tensor(btr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(epochs):
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
    return (a - m) / s, m, s


def probe(name, path, tessera, n_chips=24):
    store = open_store(path)
    tr = sample_chips(store, n_chips, 1, tessera)
    va = sample_chips(store, max(6, n_chips // 3), 2, tessera)
    a0 = np.concatenate([c[0][:, ::16, ::16].reshape(c[0].shape[0], -1)
                         for c in tr], 1)
    var = a0.var(1)
    dims = [int(np.argsort(var)[len(var) // 2]), int(np.argmax(var))]
    print(f"\n=== {name}: target dims {dims} "
          f"(median-var, max-var of {store.bands}) ===")
    results = {}
    for dim in dims:
        ytr, ntr, otr, str_ = build(tr, dim)
        yva, nva, ova, sva = build(va, dim)
        btr_, bva_ = bits_of(ytr), bits_of(yva)
        nn_, nm, ns = norm(ntr)
        on_, om, osd = norm(otr)
        priors = {
            "marginal": (np.zeros((len(ytr), 0), np.float32),
                         np.zeros((len(yva), 0), np.float32)),
            "left_px": (nn_[:, 3:4], ((nva - nm) / ns)[:, 3:4]),
            "nbr8": (nn_, (nva - nm) / ns),
            "others": (on_, (ova - om) / osd),
        }
        if name == "AE":
            half = np.arange(0, otr.shape[1], 2)
            priors["others32"] = (on_[:, half],
                                  ((ova - om) / osd)[:, half])
        if tessera:
            for k in list(priors):
                if k == "marginal":
                    continue
                priors[k] = (
                    np.concatenate([priors[k][0], str_], 1),
                    np.concatenate([priors[k][1], sva], 1))
        print(f"  dim {dim}: {len(ytr):,} train / {len(yva):,} val")
        Hm = None
        for pn, (Xtr, Xva) in priors.items():
            H = fit(Xtr.astype(np.float32), btr_,
                    Xva.astype(np.float32), bva_)
            if pn == "marginal":
                Hm = H.sum()
            print(f"    {pn:>9}: H={H.sum():5.2f}/8  "
                  f"struct {8 - H.sum():5.2f}  vs-marginal "
                  f"{Hm - H.sum():5.2f} | "
                  + " ".join(f"{h:.2f}" for h in H))
            results[(dim, pn)] = H
    return results


# probe("AE", "/data/AE-samples/chips_1024", tessera=False)
# probe("tessera", "/data/tessera-samples/chips_512", tessera=True)


def ae_norm_feature_rerun():
    """AE `others` with the unit-norm geometry handed over: append
    r2 = 1 - sum((x/127)^2) over the other 63 dims and sqrt(max(r2,0)).
    Tests the rank-(D-1) argument properly: magnitude should become
    nearly free, leaving ~1 bit (the sign) plus quantization leakage."""
    store = open_store("/data/AE-samples/chips_1024")
    tr = sample_chips(store, 96, 1, False)
    va = sample_chips(store, 32, 2, False)
    a0 = np.concatenate([c[0][:, ::16, ::16].reshape(c[0].shape[0], -1)
                         for c in tr], 1)
    dim = int(np.argmax(a0.var(1)))
    ytr, ntr, otr, _ = build(tr, dim)
    yva, nva, ova, _ = build(va, dim)
    def feats(o):
        u = o / 127.5
        v = np.sign(u) * u ** 2                  # signed-square dequant
        r2 = np.clip(1.0 - (v ** 2).sum(1, keepdims=True), 0, None)
        mag_v = np.sqrt(r2)                      # |target value|
        mag_code = 127.5 * np.sqrt(mag_v)        # |target code| (compand)
        return np.concatenate(
            [(o - otr.mean(0)) / (otr.std(0) + 1e-6),
             r2, mag_v, mag_code / 127.5], 1).astype(np.float32)
    H = fit(feats(otr), bits_of(ytr), feats(ova), bits_of(yva), epochs=6)
    print(f"\nAE dim {dim} others+norm-feature: H={H.sum():5.2f}/8 "
          f"struct {8 - H.sum():5.2f} | "
          + " ".join(f"{h:.2f}" for h in H))


ae_norm_feature_rerun()
