"""Single-pixel rate-distortion: the smallest honest experiment.

No spatial context, no pyramid, no pretrained encoder, no residual
cascade. One 128-band Tessera pixel goes into an MLP, through a
quantized bottleneck of `dim` dimensions at `bits` bits each, and back
out. Sweeps dim x bits.

The point is to measure the quantizer against an encoder that was TRAINED
WITH IT, which nothing we have done so far does. Every previous
comparison -- transform coding at 0.06062 dcos vs RVQ's 0.04119, the
FSQ-style probes -- applied a scalar scheme to latents shaped by an
encoder trained for something else, which is the same unfair direction as
quantizing raw latents with books fitted on normalized ones.

Quantizers:
  none  continuous bottleneck -- the ceiling for a given dim
  fsq   bounded tanh + per-dimension uniform levels, straight-through.
        No codebook: nothing to fit, nothing to reset, nothing to
        download.
  rvq   `levels` residual VQ stages over a K-entry book, matched on total
        bits, for the like-for-like comparison.

Each config trains in a couple of minutes, so the whole surface is
affordable in the time one pyramid run takes to prefill its pool.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.data.store import ChipStore

XNORM, SCALE_MAX = 32.0, 1.0


def sample_pixels(store_dir, n_chips, per_chip, seed=0, device="cuda"):
    """Valid 128-band pixels, drawn from many chips for spatial diversity."""
    store = ChipStore(store_dir)
    idx = store.index.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(idx), min(n_chips, len(idx)), replace=False)
    out = []
    for n, ci in enumerate(rows):
        r = idx.iloc[int(ci)]
        chip = np.asarray(store.open_shard(int(r["shard"]))[int(r["slot"])])
        sc = np.load(f"{store.dir}/scales_{int(r['shard']):04d}.npy",
                     mmap_mode="r")[int(r["slot"])]
        q = torch.from_numpy(chip.astype(np.float32))            # (128,H,W)
        s = torch.from_numpy(np.ascontiguousarray(sc).astype(np.float32))
        valid = (q.ne(0).any(0)) & (s > 0) & (s < SCALE_MAX)
        x = (q * (s / XNORM)).permute(1, 2, 0).reshape(-1, q.shape[0])
        v = valid.reshape(-1)
        x = x[v]
        if x.shape[0] > per_chip:
            sel = torch.from_numpy(rng.choice(x.shape[0], per_chip,
                                              replace=False))
            x = x[sel]
        out.append(x)
        if (n + 1) % 100 == 0:
            print(f"    {n+1}/{len(rows)} chips", flush=True)
    return torch.cat(out).to(device)


class FSQ(nn.Module):
    """Bounded scalar quantization: tanh then a uniform L-level grid."""

    def __init__(self, levels):
        super().__init__()
        self.levels = levels
        self.half = (levels - 1) / 2

    def forward(self, z):
        z = torch.tanh(z)
        q = torch.round(z * self.half) / self.half
        return z + (q - z).detach(), torch.zeros((), device=z.device)

    def bits(self, dim):
        return dim * math.log2(self.levels)


class SimpleRVQ(nn.Module):
    """Plain EMA-free RVQ with gradient codebooks -- deliberately the
    simplest thing that works, so the comparison is about the quantizer's
    STRUCTURE and not about codebook-maintenance machinery."""

    def __init__(self, n_levels, codes, dim):
        super().__init__()
        self.books = nn.ParameterList([
            nn.Parameter(torch.randn(codes, dim) * 0.1)
            for _ in range(n_levels)])
        self.codes = codes

    def forward(self, z):
        r, out, loss = z, torch.zeros_like(z), 0.0
        for B in self.books:
            d = (r.pow(2).sum(1, keepdim=True) - 2 * r @ B.t()
                 + B.pow(2).sum(1)[None])
            i = d.argmin(1)
            q = B[i]
            loss = loss + F.mse_loss(q, r.detach())
            out = out + q
            r = r - q.detach()
        return z + (out - z).detach(), loss

    def bits(self, dim):
        return len(self.books) * math.log2(self.codes)


class FourierExpand(nn.Module):
    """Per-dimension harmonic expansion of the quantized code.

    [x, sin(w1 x), cos(w1 x), sin(w2 x), cos(w2 x), ...] with w_j = j*w0,
    applied INDEPENDENTLY to each bottleneck dimension.

    Why this should matter for FSQ specifically: the decoder's first
    Linear reads each dimension affinely, so its response to a level index
    is a straight line, while the optimal reconstruction for L discrete
    levels is an arbitrary function of that index. Harmonics span a
    Fourier basis on the interval, so with J >= L/2 pairs the decoder can
    realise ANY per-level mapping -- i.e. it learns the companding a
    scalar quantizer would otherwise need a fitted level set for.

    sin alone is an odd basis and cannot represent even functions, so the
    cos terms are not optional.

    This recovers the SHAPING gain (matching the source density per
    dimension). It cannot recover the space-filling gain -- optimal cell
    SHAPE is inherently multi-dimensional -- so it should close part of
    the FSQ-vs-VQ gap, not all of it.
    """

    def __init__(self, dim, n_harm, levels=None, w0=None):
        """`levels` sets the harmonic spacing to avoid ALIASING.

        On L equally-spaced levels, sin(j*pi*x) and sin((L-j)*pi*x) are the
        SAME function -- the naive series w_j = j*pi wastes half its basis
        and comes out rank-deficient. Spacing the harmonics so the top one
        lands on the grid's Nyquist frequency, w0 = pi*(L-1)/(2J), makes
        the expansion complete instead."""
        super().__init__()
        self.dim, self.n_harm = dim, n_harm
        if w0 is None:
            w0 = (math.pi * (levels - 1) / (2 * n_harm)) if levels \
                else math.pi
        self.w0 = w0
        self.register_buffer("w", w0 * torch.arange(1, n_harm + 1).float())

    @property
    def out_dim(self):
        return self.dim * (1 + 2 * self.n_harm)

    def forward(self, z):
        zf = z[..., None] * self.w                       # (B, d, J)
        return torch.cat([z, torch.sin(zf).flatten(1),
                          torch.cos(zf).flatten(1)], dim=-1)


class PixelAE(nn.Module):
    def __init__(self, bands, dim, hidden=512, depth=3, quant=None,
                 harmonics=0, q_levels=None):
        super().__init__()
        def mlp(i, o):
            L, c = [], i
            for _ in range(depth - 1):
                L += [nn.Linear(c, hidden), nn.GELU()]
                c = hidden
            return nn.Sequential(*L, nn.Linear(c, o))
        self.exp = FourierExpand(dim, harmonics, levels=q_levels) \
            if harmonics else None
        din = self.exp.out_dim if self.exp is not None else dim
        self.enc, self.dec, self.q = mlp(bands, dim), mlp(din, bands), quant

    def forward(self, x):
        z = self.enc(x)
        aux = torch.zeros((), device=x.device)
        if self.q is not None:
            z, aux = self.q(z)
        if self.exp is not None:
            z = self.exp(z)
        return self.dec(z), aux


def run(X, Xv, dim, quant, steps, lr, batch, device, commit=0.25,
        harmonics=0):
    m = PixelAE(X.shape[1], dim, quant=quant, harmonics=harmonics,
                q_levels=getattr(quant, 'levels', None)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device=device).manual_seed(0)
    for s in range(steps):
        i = torch.randint(0, X.shape[0], (batch,), device=device, generator=g)
        xb = X[i]
        out, aux = m(xb)
        loss = F.mse_loss(out, xb) + commit * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        cs, ms = [], []
        for j in range(0, Xv.shape[0], 65536):
            xb = Xv[j:j + 65536]
            o, _ = m(xb)
            cs.append(F.cosine_similarity(o, xb, dim=1).sum())
            ms.append((o - xb).pow(2).sum(1).sum())
        cos = float(torch.stack(cs).sum() / Xv.shape[0])
        mse = float(torch.stack(ms).sum() / Xv.shape[0])
    return cos, mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/data/tessera-samples/chips_512")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--chips", type=int, default=400)
    ap.add_argument("--per-chip", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dims", default="2,4,8,16,32,64")
    ap.add_argument("--bits", default="1,2,3,4,6")
    ap.add_argument("--harmonics", default="",
                    help="'auto' for J=L/2 (a complete per-level basis "
                         "given the decoder bias), or an integer. When "
                         "set, every FSQ point is run twice -- plain and "
                         "expanded -- so the A/B is like-for-like.")
    ap.add_argument("--rvq", action="store_true",
                    help="also run rate-matched RVQ points")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    print(f"sampling pixels from {args.chips} chips ...", flush=True)
    t0 = time.time()
    P = sample_pixels(args.store, args.chips, args.per_chip, device=dev)
    n_val = min(1_000_000, P.shape[0] // 10)
    Xv, X = P[:n_val], P[n_val:]
    print(f"  {X.shape[0]:,} train / {Xv.shape[0]:,} val pixels, "
          f"{X.shape[1]} bands  [{time.time()-t0:.0f}s]")
    print(f"  |x| mean {X.norm(dim=1).mean():.3f}\n")

    dims = [int(d) for d in args.dims.split(",")]
    bits = [int(b) for b in args.bits.split(",")]
    rows = []

    print(f"{'quant':>6} {'dim':>5} {'b/dim':>6} {'bits':>7} "
          f"{'cos':>8} {'mse':>9} {'s':>5}")
    for d in dims:                                   # continuous ceiling
        t = time.time()
        cos, mse = run(X, Xv, d, None, args.steps, args.lr, args.batch, dev)
        print(f"{'none':>6} {d:>5} {'-':>6} {'-':>7} {cos:>8.4f} "
              f"{mse:>9.5f} {time.time()-t:>5.0f}", flush=True)
        rows.append({"quant": "none", "dim": d, "bits_per_dim": None,
                     "bits": None, "cos": cos, "mse": mse})

    for d, b in itertools.product(dims, bits):
        L = 2 ** b
        hs = [0]
        if args.harmonics:
            hs.append(L // 2 if args.harmonics == "auto"
                      else int(args.harmonics))
        for h in hs:
            q = FSQ(L)
            t = time.time()
            cos, mse = run(X, Xv, d, q, args.steps, args.lr, args.batch, dev,
                           harmonics=h)
            tag = "fsq" if not h else f"fsq+f{h}"
            print(f"{tag:>6} {d:>5} {b:>6} {q.bits(d):>7.0f} {cos:>8.4f} "
                  f"{mse:>9.5f} {time.time()-t:>5.0f}", flush=True)
            rows.append({"quant": tag, "dim": d, "bits_per_dim": b,
                         "harmonics": h, "bits": q.bits(d), "cos": cos,
                         "mse": mse})
            if args.out:
                json.dump(rows, open(args.out, "w"), indent=1)

    if args.rvq:
        for d, b in itertools.product(dims, bits):
            total = d * b
            for kbits in (8, 10, 12):
                nl = max(1, round(total / kbits))
                q = SimpleRVQ(nl, 2 ** kbits, d)
                t = time.time()
                cos, mse = run(X, Xv, d, q, args.steps, args.lr, args.batch,
                               dev)
                print(f"{'rvq':>6} {d:>5} {kbits:>6} {q.bits(d):>7.0f} "
                      f"{cos:>8.4f} {mse:>9.5f} {time.time()-t:>5.0f}",
                      flush=True)
                rows.append({"quant": "rvq", "dim": d, "levels": nl,
                             "code_bits": kbits, "bits": q.bits(d),
                             "cos": cos, "mse": mse})
                if args.out:
                    json.dump(rows, open(args.out, "w"), indent=1)

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
