"""Rotation + scalar quantization vs the deployed RVQ, at matched rate.

The tier-2 codebooks are the expensive part of this codec: 94% of the
bitstream and 6.25 MiB of download. A TurboQuant-style coder would remove
the codebook entirely -- rotate, then scalar-quantize against a KNOWN
distribution -- so it is worth knowing what that costs in distortion.

The theory, and why each arm is here:

  ROTATION. A random rotation of a vector in d dimensions makes its
  coordinates approximately i.i.d. N(0, ||x||^2/d) by concentration of
  measure. A scalar quantizer designed for that Gaussian is then
  near-optimal for every coordinate, with no data-dependent codebook.
  The cost is the space-filling loss of scalar vs vector quantization --
  1.53 dB, i.e. about 0.25 bits/dimension, which is exactly the gap a
  good VQ is supposed to buy back.

  PCA ROTATION. The same idea but with the data's own eigenbasis, which
  additionally COMPACTS energy so bits can be allocated where the variance
  is (reverse water-filling). Strictly better than random rotation on
  distortion; costs a d x d basis in the bitstream header.

  RANDOM PROJECTION to k < d. Included because it is the obvious way to
  get sub-1-bit, and because it is a trap: a random k-subspace captures
  only k/d of an isotropic vector's energy, so it discards (1-k/d)
  irrecoverably. PCA truncation to the same k keeps the top eigenvalues
  instead. Reported so the difference is measured rather than argued.

  RVQ. The deployed coder, using the model's OWN trained codebooks, so
  the comparison is against what actually ships rather than a strawman.

Rate accounting is honest for each arm: scalar arms are charged the
empirical ENTROPY of their symbols (what an arithmetic coder achieves),
RVQ is charged levels x log2(K). Basis/codebook storage is reported
separately since it amortises over the whole dataset, not per vector.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import ChipStore
from georvq.factory import from_checkpoint


def collect_latents(m, val, tier, dev, chunk=8, cap=200000):
    zs = []
    for i in range(0, val.shape[0], chunk):
        x, v = m.source.prep(val[i:i + chunk], dev)
        x = x.float()
        st = m._stats_of(x, v)
        z = m.encode_levels(x, st, v)[tier]
        zs.append(z.reshape(-1, z.shape[-1]).float().cpu())
        if sum(t.shape[0] for t in zs) >= cap:
            break
    Z = torch.cat(zs)[:cap]
    return Z


def entropy_of(idx):
    """Empirical entropy in bits of an integer tensor (what an arithmetic
    coder would spend), summed per-dimension."""
    tot = 0.0
    for j in range(idx.shape[1]):
        c = torch.bincount(idx[:, j] - idx[:, j].min()).double()
        p = c[c > 0] / c.sum()
        tot += float(-(p * p.log2()).sum())
    return tot


def lloyd_batch(X, L, iters=30, init_n=20000):
    """Lloyd-Max levels for ALL dimensions at once. X: (d, N) -> q: (d, L).

    Vectorised deliberately: a per-dimension loop is d x iters x L small
    GPU ops, each a synchronisation, which took tens of minutes at d=256.
    Batched searchsorted + scatter_add is the same computation in seconds.

    On rotated coordinates each row is Gaussian, so this converges to the
    classic Gaussian Lloyd-Max quantizer -- the 'known distribution' the
    method leans on.
    """
    d, N = X.shape
    sub = X[:, torch.randperm(N, device=X.device)[:min(N, init_n)]]
    qs = torch.linspace(0.5 / L, 1 - 0.5 / L, L, device=X.device)
    q = torch.quantile(sub, qs, dim=1).T.contiguous()          # (d, L)
    ones = torch.ones_like(X)
    for _ in range(iters):
        b = ((q[:, 1:] + q[:, :-1]) / 2).contiguous()
        a = torch.searchsorted(b, X.contiguous())              # (d, N)
        s = torch.zeros_like(q).scatter_add_(1, a, X)
        c = torch.zeros_like(q).scatter_add_(1, a, ones)
        q = torch.where(c > 0, s / c.clamp_min(1.0), q)
    return q


def quant_batch(X, q):
    """X: (d, N), q: (d, L) -> indices (d, N) and reconstruction (d, N)."""
    b = ((q[:, 1:] + q[:, :-1]) / 2).contiguous()
    a = torch.searchsorted(b, X.contiguous())
    return a, torch.gather(q, 1, a)


def rel_mse(a, b):
    return float((a - b).pow(2).sum() / b.pow(2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--levels", default="2,3,4,5,8")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    store = ChipStore(cfg.data.store_dir)
    _, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val = fixed_val_tokens(store, va, token=cfg.model.window,
                           max_tokens=args.tokens)
    Z = collect_latents(m, val, args.tier, dev, cap=args.cap)
    N, d = Z.shape
    sp = m.specs[args.tier]
    print(f"tier {args.tier}: {N:,} latents of dim {d}   "
          f"(deployed: {sp.depth} levels x {int(math.log2(sp.codes))} bits "
          f"= {sp.depth*math.log2(sp.codes)/d:.3f} bits/dim)\n")

    Zg = Z.to(dev)
    n_tr = N // 2
    tr, te = Zg[:n_tr], Zg[n_tr:]

    rows = []

    # ---- RVQ with the model's own trained codebooks ----
    rvq = m.rvqs[args.tier]
    res = te.clone()
    for lv in range(sp.depth):
        C = rvq.levels[lv].codebook.weight.data.float()
        a = torch.cdist(res, C).argmin(1)
        res = res - C[a]
        bits = (lv + 1) * math.log2(sp.codes)
        if (lv + 1) in (1, 2, 4, 8, 12, 16, 20):
            rows.append((f"RVQ x{lv+1}", bits / d,
                         float(res.pow(2).sum() / te.pow(2).sum())))

    # ---- rotation + Lloyd scalar, charged at empirical entropy ----
    g = torch.Generator(device=dev).manual_seed(0)
    R, _ = torch.linalg.qr(torch.randn(d, d, generator=g, device=dev))
    U = torch.linalg.eigh(torch.cov((tr - tr.mean(0)).T))[1].flip(1)
    for name, B in (("rand-rot", R), ("PCA-rot", U)):
        Ytr, Yte = tr @ B, te @ B
        for L in [int(v) for v in args.levels.split(",")]:
            q = lloyd_batch(Ytr.T, L)
            I, Yq = quant_batch(Yte.T, q)
            rows.append((f"{name} L={L}", entropy_of(I.T.cpu()) / d,
                         rel_mse(Yq.T @ B.T, te)))

    # ---- REFINEMENT: 1 bit/dim on top of the full RVQ ----
    # This is the operationally interesting version. The RVQ ladder stays,
    # the decoder keeps the distribution it was trained on, and the extra
    # bits code the LEFTOVER quantization noise. That residual should be
    # far closer to isotropic than the raw latent -- the RVQ has already
    # taken the structure -- which is precisely when a random rotation
    # costs nothing and a Gaussian-optimal scalar quantizer is right.
    def rvq_resid(Zs):
        r = Zs.clone()
        for lv in range(sp.depth):
            C = rvq.levels[lv].codebook.weight.data.float()
            r = r - C[torch.cdist(r, C).argmin(1)]
        return r
    rtr, rte = rvq_resid(tr), rvq_resid(te)
    base = float(rte.pow(2).sum() / te.pow(2).sum())
    lam = torch.linalg.eigvalsh(torch.cov((rte - rte.mean(0)).T)).clamp_min(0)
    pr_res = float(lam.sum() ** 2 / lam.pow(2).sum())
    zz = (rte - rte.mean(0)) / rte.std(0).clamp_min(1e-9)
    print(f"RVQ residual after {sp.depth} levels: rel MSE {base:.4f}, "
          f"participation ratio {pr_res:.1f}/{d}, "
          f"excess kurtosis {float((zz.pow(4).mean(0)-3).mean()):.2f}")
    print(f"  (isotropic => random rotation is free; the raw latent's PR "
          f"was much lower)\n")
    rvq_bits = sp.depth * math.log2(sp.codes)
    for name, B in (("+rand-rot", R), ("+PCA-rot", U)):
        Ytr, Yte = rtr @ B, rte @ B
        for L in [int(v) for v in args.levels.split(",")]:
            q = lloyd_batch(Ytr.T, L)
            I, Yq = quant_batch(Yte.T, q)
            add = entropy_of(I.T.cpu()) / d
            fixed = rte - Yq.T @ B.T
            rows.append((f"{name} L={L}", rvq_bits / d + add,
                         float(fixed.pow(2).sum() / te.pow(2).sum())))

    # ---- projection to k dims: random vs PCA, the trap ----
    for k in (32, 64, 128):
        Pr = R[:, :k]
        rows.append((f"rand-proj k={k}", float("nan"),
                     rel_mse((te @ Pr) @ Pr.T, te)))
        Pp = U[:, :k]
        rows.append((f"PCA-proj  k={k}", float("nan"),
                     rel_mse((te @ Pp) @ Pp.T, te)))

    print(f"{'method':>18} {'bits/dim':>9} {'rel MSE':>9} {'vs RVQ@same':>12}")
    rvq_pts = [(b, e) for n, b, e in rows if n.startswith("RVQ")]
    for n, b, e in rows:
        cmp = ""
        if not math.isnan(b) and not n.startswith("RVQ"):
            near = min(rvq_pts, key=lambda p: abs(p[0] - b))
            cmp = f"{e/near[1]:.2f}x" if near[1] > 0 else ""
        bs = "  (lossy proj)" if math.isnan(b) else f"{b:>9.3f}"
        print(f"{n:>18} {bs:>9} {e:>9.4f} {cmp:>12}")
    print("\nrel MSE is on held-out latents. 'vs RVQ@same' compares to the "
          "nearest\nRVQ rate point: >1 means worse than the deployed coder.")
    print(f"codebook storage: RVQ {sp.depth*sp.codes*d/2**20:.2f} MiB at "
          f"int8; rand-rot 0 (seed only); PCA-rot {d*d*4/2**20:.2f} MiB fp32")


if __name__ == "__main__":
    main()
