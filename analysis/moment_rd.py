"""Rate-distortion of a patch's moments, coded end-to-end.

encoder -> RVQ -> decoder on the 256-d vector [mean(128) || std(128)] of a
64px window. This is the smallest complete codec for the quantity the
coarse tier is really responsible for, and it answers the question
`patch_moments_pca.py` left open: PCA said the moments have no
low-dimensional LINEAR structure (107 of 128 components for 99% of the
mean's variance), so the remaining hope is that a NONLINEAR encoder finds
structure a linear one cannot.

The comparator is the Gaussian rate-distortion bound of the same 256-d
vector, computed by reverse water-filling on its measured covariance.
Because the Gaussian maximises R(D) at fixed covariance, that bound is
the rate a linear-Gaussian coder would need, and beating it is exactly
the claim "the moments have nonlinear structure worth exploiting".

Metrics are reported for the two blocks SEPARATELY and in original units.
A combined number hides the thing we most want to know: the mean block
carries 86% of the energy, so a coder can look good on a pooled metric
while ignoring the spread entirely.

Train/val are geographically disjoint (see load_or_collect).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_moments_pca import load_or_collect            # noqa: E402
from pixel_rd import RVQQuant                            # noqa: E402


class MomentAE(nn.Module):
    def __init__(self, d_in, dim, hidden=1024, depth=3, quant=None):
        super().__init__()

        def mlp(i, o):
            L, c = [], i
            for _ in range(depth - 1):
                L += [nn.Linear(c, hidden), nn.GELU()]
                c = hidden
            return nn.Sequential(*L, nn.Linear(c, o))

        self.enc, self.dec, self.q = mlp(d_in, dim), mlp(dim, d_in), quant

    def forward(self, x):
        z = self.enc(x)
        aux = torch.zeros((), device=x.device)
        if self.q is not None:
            z, aux = self.q(z)
        return self.dec(z), aux


@torch.no_grad()
def evaluate(m, Xn, X, mu, sd, nb):
    """Held-out errors in ORIGINAL units, per block. nb = bands per block."""
    m.eval()
    out = []
    for i in range(0, Xn.shape[0], 8192):
        out.append(m(Xn[i:i + 8192])[0])
    Xh = torch.cat(out) * sd + mu
    m.train()
    r = {}
    for name, sl in (("mean", slice(0, nb)), ("std", slice(nb, 2 * nb))):
        a, b = Xh[:, sl], X[:, sl]
        num = (a - b).pow(2).sum(1).sqrt()
        r[f"{name}_rel"] = float((num / b.pow(2).sum(1).sqrt()).mean())
        r[f"{name}_cos"] = float(F.cosine_similarity(a, b, dim=1).mean())
    num = (Xh - X).pow(2).sum(1).sqrt()
    r["all_rel"] = float((num / X.pow(2).sum(1).sqrt()).mean())
    return r


def train_one(X, Xv, dim, quant, steps, lr, batch, device, commit=0.25,
              loss_space="raw", drift_every=200, log_every=0):
    """X, Xv: original-unit (N, 256) tensors on `device`."""
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xn, Xvn = (X - mu) / sd, (Xv - mu) / sd
    nb = X.shape[1] // 2
    m = MomentAE(X.shape[1], dim, quant=quant).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device=device).manual_seed(0)
    is_rvq = isinstance(quant, RVQQuant)
    # the drift probe must come from TRAIN data: it sets the EMA decay, so
    # feeding it val data would let the val set influence the model
    probe_x = Xn[:4096] if is_rvq else None
    prev_z, prev_s = None, 0
    for s in range(steps):
        if is_rvq and not quant.inited and s >= quant.warmup:
            with torch.no_grad():
                zs = torch.cat([
                    m.enc(Xn[torch.randint(0, Xn.shape[0], (batch,),
                                           device=device)])
                    for _ in range(8)])
            quant.tree_init(zs)
            del zs
        # adaptive EMA window from the measured encoder velocity: the
        # codebook chases a moving encoder, and W* = (2 sig^2 K/(delta^2 B))^(1/3)
        if is_rvq and quant.inited and drift_every and s % drift_every == 0:
            with torch.no_grad():
                z_now = m.enc(probe_x)
                if prev_z is not None and s > prev_s:
                    C = quant.rvq.levels[0].codebook.weight.data.float()
                    d = (z_now.pow(2).sum(1, keepdim=True) - 2 * z_now @ C.t()
                         + C.pow(2).sum(1)[None])
                    sig = float((z_now - C[d.argmin(1)]).pow(2)
                                .sum(1).mean().sqrt())
                    delta = float((z_now - prev_z).norm(dim=-1).mean()) \
                        / (s - prev_s)
                    if delta > 0 and sig > 0:
                        W = (2 * sig ** 2 * quant.codes
                             / (delta ** 2 * batch)) ** (1 / 3)
                        dec = min(0.9995, max(0.5, (W - 1) / (W + 1)))
                        for lv in quant.rvq.levels:
                            lv.decay = dec
                prev_z, prev_s = z_now, s
        i = torch.randint(0, Xn.shape[0], (batch,), device=device,
                          generator=g)
        xb = Xn[i]
        out, aux = m(xb)
        # `raw`: squared error in ORIGINAL units, so each dimension counts
        # for what it contributes to the actual vector. `norm`: every
        # standardized dimension counts equally, which over-weights bands
        # that barely vary.
        err = (out - xb) * sd if loss_space == "raw" else (out - xb)
        loss = err.pow(2).sum(1).mean() + commit * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if log_every and (s + 1) % log_every == 0:
            print(f"    step {s+1}/{steps} loss={float(loss):.5f}",
                  flush=True)
    return evaluate(m, Xvn, Xv, mu, sd, nb)


def gauss_bits(X, rel_targets, blocks=None):
    """Reverse water-filling on the joint covariance -> bits per window.

    Returns {rel: bits}. Distortion is charged against E||x||^2 of the
    whole vector (or of `blocks`, a slice, when scoring one block).
    """
    A = X.double()
    lam = torch.linalg.eigvalsh(torch.cov((A - A.mean(0)).T)).clamp_min(1e-12)
    energy = float(A.pow(2).sum(1).mean())
    out = {}
    for rel in rel_targets:
        target = (rel ** 2) * energy
        if target >= float(lam.sum()):
            out[rel] = 0.0
            continue
        lo, hi = 1e-14, float(lam.max())
        for _ in range(200):
            th = 0.5 * (lo + hi)
            if float(torch.clamp(lam, max=th).sum()) > target:
                hi = th
            else:
                lo = th
        th = 0.5 * (lo + hi)
        out[rel] = 0.5 * float(torch.log2(lam[lam > th] / th).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="")
    ap.add_argument("--store", default="/data/tessera-samples/chips_512")
    ap.add_argument("--chips", type=int, default=4000)
    ap.add_argument("--val-chips", type=int, default=700)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--min-valid", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--dims", default="16,32,64,128")
    ap.add_argument("--bits", default="32,64,128,256")
    ap.add_argument("--codes", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss-space", default="raw", choices=["raw", "norm"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    M, S, Mv, Sv = load_or_collect(args.store, args.cache, args.chips,
                                   args.val_chips, args.win, args.min_valid,
                                   args.workers)
    X = torch.from_numpy(np.concatenate([M, S], 1)).float().to(dev)
    Xv = torch.from_numpy(np.concatenate([Mv, Sv], 1)).float().to(dev)
    nb = M.shape[1]
    print(f"train {X.shape[0]:,} windows, val {Xv.shape[0]:,} "
          f"(geographically disjoint), d={X.shape[1]}")

    # the linear-Gaussian comparator, on the same TRAIN covariance
    rels = [0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
    gb = gauss_bits(X, rels)
    print("\nGaussian bound on the joint 256-d vector "
          "(rate a linear-Gaussian coder needs):")
    print("  " + "  ".join(f"{100*r:g}%={gb[r]:.0f}b" for r in rels))

    dims = [int(v) for v in args.dims.split(",")]
    bitlist = [int(v) for v in args.bits.split(",")]
    kb = int(math.log2(args.codes))
    rows = []
    print(f"\n{'dim':>5} {'bits':>6} {'b/px':>7} {'mean rel':>9} "
          f"{'mean cos':>9} {'std rel':>8} {'all rel':>8} {'gauss b':>8} "
          f"{'vs bound':>9} {'s':>5}")
    for dim in dims:
        for bits in [0] + bitlist:
            if bits and bits % kb:
                continue
            nl = bits // kb
            if bits and nl < 1:
                continue
            q = RVQQuant(nl, args.codes, dim).to(dev) if bits else None
            t0 = time.time()
            r = train_one(X, Xv, dim, q, args.steps, args.lr, args.batch,
                          dev, loss_space=args.loss_space)
            dt = time.time() - t0
            # bits a linear-Gaussian coder would need for the SAME error
            need = gauss_bits(X, [r["all_rel"]])[r["all_rel"]]
            ratio = (need / bits) if bits else float("nan")
            rows.append(dict(dim=dim, bits=bits, **r, gauss_bits=need))
            tag = "ceiling" if not bits else f"{bits}"
            print(f"{dim:>5} {tag:>6} {bits/args.win**2:>7.4f} "
                  f"{r['mean_rel']:>9.4f} {r['mean_cos']:>9.5f} "
                  f"{r['std_rel']:>8.4f} {r['all_rel']:>8.4f} "
                  f"{need:>8.0f} {ratio:>9.2f} {dt:>5.0f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
