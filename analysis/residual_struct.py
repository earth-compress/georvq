"""What structure is left in the codec's reconstruction error?

Decides the architecture of a residual coder BEFORE training one. If the
residual still carries spatial correlation, a per-pixel coder discards it
and a small convolutional stage will beat it at the same rate. If the
residual is spatially white, per-pixel is the right structure and the
design is simple.

Measures, on held-out chips at full depth:
  * spatial autocorrelation of the residual at lags 1..8 px, against the
    same statistic on the SOURCE -- the source's correlation is the thing
    the pyramid is supposed to have removed, so the comparison says how
    much of it survived;
  * cross-band correlation and the residual's effective rank (PCA d95),
    which decides how wide a per-pixel bottleneck would need to be;
  * the residual's energy distribution across pixels, since a heavy tail
    means a few pixels dominate and rate should be allocated, not uniform.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import from_checkpoint


@torch.no_grad()
def autocorr(t, lags):
    """Mean over bands of the spatial autocorrelation at each lag.

    t: (B, C, H, W). Correlation is computed on band-centred values, so a
    constant offset per band cannot manufacture correlation."""
    x = t - t.mean(dim=(0, 2, 3), keepdim=True)
    denom = x.pow(2).mean(dim=(0, 2, 3)).clamp_min(1e-12)
    out = {}
    for d in lags:
        h = (x[:, :, :, :-d] * x[:, :, :, d:]).mean(dim=(0, 2, 3))
        v = (x[:, :, :-d, :] * x[:, :, d:, :]).mean(dim=(0, 2, 3))
        out[d] = float((0.5 * (h + v) / denom).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="checkpoints/tessera_vq8_split2/model_best.pt")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--tuples", default="",
                    help="comma list like 8-0-0,8-8-0,8-8-8; "
                         "blank = full depth only")
    args = ap.parse_args()

    # Without this the residuals appended below keep `out`'s autograd
    # graph alive for every chunk -- the whole forward across all windows
    # stays resident, which is what OOM'd at 62 GiB even at chunk size 8.
    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    store = open_store(cfg.data.store_dir)
    _, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val = fixed_val_tokens(store, va, token=cfg.model.window,
                           max_tokens=args.tokens).to(dev)
    full = [sp.depth for sp in m.specs]
    print(f"{args.ckpt}\n  {val.shape[0]} held-out windows of "
          f"{cfg.model.window}px, full depth {full}")

    tuples = ([tuple(int(v) for v in t.split("-"))
               for t in args.tuples.split(",")] if args.tuples else [tuple(full)])

    # cell size in pixels for each tier: the scale that tier is
    # responsible for, and therefore the lag at which its own residual
    # should already look white
    cell_px = [cfg.model.window // sp.grid for sp in m.specs]
    print(f"  tier cell sizes: {cell_px} px")

    # small chunks and CPU accumulation: quantizing at full depth builds a
    # (chunk*cells, K) score matrix per level -- 8192x16384 at chunk 32 --
    # and this shares the card with other users' jobs
    lags = [1, 2, 4, 8, 16, 32]
    print(f"\n{'tuple':>10} {'b/px':>7} {'resid %':>8} " +
          " ".join(f"{'lag'+str(d):>7}" for d in lags))
    keep = None
    for tup in tuples:
        X, R = [], []
        for i in range(0, val.shape[0], 8):
            x, v = m.source.prep(val[i:i + 8], dev)
            zs = m.encode_levels(x, m._stats_of(x, v), v)
            Ls = torch.tensor(tup, device=dev)[None].expand(x.shape[0], -1)
            zq, _, _ = m.quantize(zs, Ls)
            out = m.decode(zq, Ls, m._stats_of(x, v), v).float()
            X.append(x.float().cpu()); R.append((x.float() - out).cpu())
            del x, zs, zq, out
            torch.cuda.empty_cache()
        X = torch.cat(X); R = torch.cat(R)
        import math
        b = sum(sp.cells * L * math.log2(sp.codes)
                for sp, L in zip(m.specs, tup)) / (cfg.model.window ** 2)
        ar = autocorr(R.to(dev), lags)
        pc = 100 * float(R.pow(2).sum(1).mean() / X.pow(2).sum(1).mean())
        print(f"{'-'.join(map(str,tup)):>10} {b:>7.3f} {pc:>7.1f}% " +
              " ".join(f"{ar[d]:>7.3f}" for d in lags), flush=True)
        keep = (X, R)
    ax = autocorr(keep[0].to(dev), lags)
    print(f"{'SOURCE':>10} {'-':>7} {'100.0':>7}% " +
          " ".join(f"{ax[d]:>7.3f}" for d in lags))
    X, R = keep

    # cross-band structure of the residual vector
    r = R.permute(0, 2, 3, 1).reshape(-1, R.shape[1])
    r = r[torch.randperm(r.shape[0])[:400000]].to(dev).double()
    rc = r - r.mean(0)
    C = (rc.T @ rc) / rc.shape[0]
    ev = torch.linalg.eigvalsh(C).flip(0)
    tot = ev.sum()
    d95 = int((torch.cumsum(ev, 0) / tot < 0.95).sum()) + 1
    xs = X.permute(0, 2, 3, 1).reshape(-1, X.shape[1])
    xs = xs[torch.randperm(xs.shape[0])[:400000]].to(dev).double()
    xc = xs - xs.mean(0)
    evx = torch.linalg.eigvalsh((xc.T @ xc) / xc.shape[0]).flip(0)
    d95x = int((torch.cumsum(evx, 0) / evx.sum() < 0.95).sum()) + 1
    print(f"\neffective rank (PCA d95 of {R.shape[1]} bands): "
          f"source {d95x}, residual {d95}")

    e = R.pow(2).sum(1).flatten().to(dev)
    q = torch.quantile(e.float(),
                       torch.tensor([0.5, 0.9, 0.99], device=dev))
    print(f"per-pixel residual energy: median {float(q[0]):.4f}, "
          f"p90 {float(q[1]):.4f}, p99 {float(q[2]):.4f} "
          f"(p99/median = {float(q[2]/q[0].clamp_min(1e-9)):.1f}x)")
    top = float(e.topk(int(0.1 * e.numel())).values.sum() / e.sum())
    print(f"  the worst 10% of pixels carry {100*top:.1f}% of the error")


if __name__ == "__main__":
    main()
