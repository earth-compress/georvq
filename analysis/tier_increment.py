"""What does each tier ADD, at what scale, and does it fix what was wrong?

Cumulative depth tuples conflate tiers: 20-20-0 tells you about tier 0 and
tier 1 together. This decomposes the cascade into per-tier INCREMENTS

    D_k = recon(tiers <= k) - recon(tiers <= k-1)

and scores each one against the error it was supposed to correct,

    e_{k-1} = x - recon(tiers <= k-1)

Three numbers per tier, all on the mean-removed field (with stats
conditioning the mean is transmitted, so it is not the pyramid's work):

  ENERGY     ||D_k||^2 as a fraction of the source's spatial energy --
             how much variation this tier injects.
  ALIGNMENT  cos(D_k, e_{k-1}) -- whether that injection points at the
             error. A tier can add plenty of energy in the wrong
             direction, which looks like progress in an energy-only view
             and is actively harmful.
  SCALE      autocorrelation of D_k. A healthy cascade has each tier
             operating at a shorter correlation length than the last;
             if two tiers show the same profile they are duplicating work.

The baseline rung is the transmitted mean alone (all tiers at depth 0), so
tier 0's increment is measured against a mean-only reconstruction rather
than against nothing.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import ChipStore
from georvq.factory import from_checkpoint


@torch.no_grad()
def deviation(x, valid):
    vm = valid.unsqueeze(1).to(x.dtype)
    mu = (x * vm).sum((2, 3), keepdim=True) / vm.sum((2, 3), keepdim=True) \
        .clamp_min(1e-6)
    return (x - mu) * vm


@torch.no_grad()
def autocorr(t, lags, valid):
    out = {}
    den = t.pow(2).sum(1)[valid].mean().clamp_min(1e-12)
    for d in lags:
        h = (t[:, :, :, :-d] * t[:, :, :, d:]).sum(1)
        hv = valid[:, :, :-d] & valid[:, :, d:]
        v = (t[:, :, :-d, :] * t[:, :, d:, :]).sum(1)
        vv = valid[:, :-d, :] & valid[:, d:, :]
        out[d] = float(((h[hv].sum() + v[vv].sum())
                        / (hv.sum() + vv.sum()).clamp_min(1)) / den)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    store = ChipStore(cfg.data.store_dir)
    _, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val = fixed_val_tokens(store, va, token=cfg.model.window,
                           max_tokens=args.tokens)
    specs = m.specs
    full = [sp.depth for sp in specs]
    # cascade rungs: mean only, then each tier switched on at full depth
    rungs = [tuple([0] * len(specs))]
    for i in range(len(specs)):
        rungs.append(tuple(full[:i + 1] + [0] * (len(specs) - i - 1)))
    lags = [1, 2, 4, 8, 16]
    print(f"{args.ckpt}\n  {val.shape[0]} val windows, rungs "
          f"{['-'.join(map(str,r)) for r in rungs]}\n")

    acc = [{"e": 0.0, "n": 0.0, "cs": 0.0, "cn": 0, "ac": {d: 0.0 for d in lags},
            "nb": 0} for _ in specs]
    src_e = 0.0
    for i in range(0, val.shape[0], args.chunk):
        x, v = m.source.prep(val[i:i + args.chunk], dev)
        x = x.float()
        if not int(v.sum()):
            continue
        st = m._stats_of(x, v)
        zs = m.encode_levels(x, st, v)
        dx = deviation(x, v)
        src_e += float(dx.pow(2).sum(1)[v].sum())
        prev = None
        for k, r in enumerate(rungs):
            Ls = torch.tensor(r, device=dev)[None].expand(x.shape[0], -1)
            zq, _, _ = m.quantize(zs, Ls)
            o = deviation(m.decode(zq, Ls, st, v).float(), v)
            if prev is not None:
                a = acc[k - 1]
                D = o - prev                      # what this tier added
                err = dx - prev                   # what was wrong before it
                a["e"] += float(D.pow(2).sum(1)[v].sum())
                c = F.cosine_similarity(D, err, dim=1, eps=1e-8)[v]
                a["cs"] += float(c.sum()); a["cn"] += int(c.numel())
                ac = autocorr(D, lags, v)
                for d in lags:
                    a["ac"][d] += ac[d]
                a["nb"] += 1
            prev = o
        del x, zs, zq

    print(f"{'tier':>5} {'bits':>7} {'b/px':>7} {'energy':>8} {'align':>7}   "
          + " ".join(f"{'r'+str(d):>6}" for d in lags))
    W = cfg.model.window
    for k, sp in enumerate(specs):
        a = acc[k]
        bits = sp.cells * sp.depth * math.log2(sp.codes)
        print(f"{k:>5} {int(bits):>7} {bits/W**2:>7.3f} "
              f"{a['e']/max(src_e,1e-9):>8.4f} {a['cs']/max(a['cn'],1):>7.4f}   "
              + " ".join(f"{a['ac'][d]/max(a['nb'],1):>6.3f}" for d in lags))
    print("\nenergy = this tier's increment as a fraction of the SOURCE's "
          "spatial energy\nalign  = cosine between the increment and the "
          "error it was correcting\nr(d)   = autocorrelation of the "
          "increment: the scale this tier works at")


if __name__ == "__main__":
    main()
