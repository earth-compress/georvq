"""How much spatial variation does a depth tuple actually encode?

`tier0_moments.py` showed the coarse tier recovers the per-window STD
badly (rel 0.835 at 20-0-0, worse than a constant predictor). That is
consistent with two very different failures, and the fix differs:

  TOO LITTLE  the reconstruction is nearly a flat patch, so it has almost
              no spatial variation to be right or wrong about;
  WRONG       it has plenty of variation, pointed in the wrong direction.

So decompose each reconstruction into its own patch mean plus a spatial
deviation, and report both the MAGNITUDE of that deviation relative to
the truth's, and its ALIGNMENT with the truth's. A flat patch scores
~0 magnitude; a busy but uncorrelated one scores magnitude ~1 and
alignment ~0.

Also reports the deviation's spatial autocorrelation against the source's,
which says what SCALE of structure survives -- a coarse tier that encodes
only very smooth shape should show correlation far above the source's at
short lags.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import from_checkpoint


@torch.no_grad()
def deviation(x, valid):
    """x minus its per-window mean over valid pixels; nodata left at zero."""
    vm = valid.unsqueeze(1).to(x.dtype)
    mu = (x * vm).sum((2, 3), keepdim=True) / vm.sum((2, 3), keepdim=True) \
        .clamp_min(1e-6)
    return (x - mu) * vm


@torch.no_grad()
def autocorr(t, lags, valid):
    """Spatial autocorrelation of a (already band-centred) field, averaged
    over bands. Pairs are only counted where BOTH pixels are valid."""
    out = {}
    den = t.pow(2).sum(1)[valid].mean().clamp_min(1e-12)
    for d in lags:
        h = (t[:, :, :, :-d] * t[:, :, :, d:]).sum(1)
        hv = valid[:, :, :-d] & valid[:, :, d:]
        v = (t[:, :, :-d, :] * t[:, :, d:, :]).sum(1)
        vv = valid[:, :-d, :] & valid[:, d:, :]
        num = (h[hv].sum() + v[vv].sum()) / (hv.sum() + vv.sum()).clamp_min(1)
        out[d] = float(num / den)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="checkpoints/tessera_vq8_k256/model_best.pt")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--tuples", default="4-0-0,10-0-0,20-0-0,20-1-0,20-20-3,"
                                        "20-20-20")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    store = open_store(cfg.data.store_dir)
    _, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val = fixed_val_tokens(store, va, token=cfg.model.window,
                           max_tokens=args.tokens)
    W = cfg.model.window
    print(f"{args.ckpt}\n  {val.shape[0]} held-out {W}px windows")
    lags = [1, 2, 4, 8, 16]
    print(f"\n{'tuple':>10} {'bits':>7} {'spatial':>8} {'align':>7}   "
          + " ".join(f"{'r'+str(d):>6}" for d in lags))
    print(f"{'':>10} {'':>7} {'energy':>8} {'(ccos)':>7}")

    tuples = [tuple(int(v) for v in t.split("-"))
              for t in args.tuples.split(",")]
    src = {"e": 0.0, "n": 0, "ac": None}
    for tup in tuples:
        num = den = 0.0
        cs, cn = 0.0, 0
        acc = {d: 0.0 for d in lags}
        nb = 0
        for i in range(0, val.shape[0], 8):
            x, v = m.source.prep(val[i:i + 8], dev)
            x = x.float()
            if not int(v.sum()):
                continue
            st = m._stats_of(x, v)      # None unless stats_cond is on
            zs = m.encode_levels(x, st, v)
            Ls = torch.tensor(tup, device=dev)[None].expand(x.shape[0], -1)
            zq, _, _ = m.quantize(zs, Ls)
            out = m.decode(zq, Ls, st, v).float()
            dx, do = deviation(x, v), deviation(out, v)
            num += float(do.pow(2).sum(1)[v].sum())
            den += float(dx.pow(2).sum(1)[v].sum())
            c = F.cosine_similarity(do, dx, dim=1, eps=1e-8)[v]
            cs += float(c.sum()); cn += int(c.numel())
            a = autocorr(do, lags, v)
            for d in lags:
                acc[d] += a[d]
            if src["ac"] is None:
                pass
            nb += 1
            if tup == tuples[0]:      # source stats, computed once
                sa = autocorr(dx, lags, v)
                src["e"] += float(dx.pow(2).sum(1)[v].sum())
                src["n"] += 1
                src["ac"] = ({d: 0.0 for d in lags} if src["ac"] is None
                             else src["ac"])
                for d in lags:
                    src["ac"][d] += sa[d]
            del x, zs, zq, out
        bits = sum(sp.cells * L * math.log2(sp.codes)
                   for sp, L in zip(m.specs, tup))
        print(f"{'-'.join(map(str,tup)):>10} {int(bits):>7} "
              f"{num/max(den,1e-9):>8.4f} {cs/max(cn,1):>7.4f}   "
              + " ".join(f"{acc[d]/max(nb,1):>6.3f}" for d in lags),
              flush=True)
    print(f"{'SOURCE':>10} {'-':>7} {1.0:>8.4f} {1.0:>7.4f}   "
          + " ".join(f"{src['ac'][d]/max(src['n'],1):>6.3f}" for d in lags))
    print("\nspatial energy = ||recon - its own patch mean||^2 relative to "
          "the source's\nalign (ccos)  = cosine between the two deviations; "
          "0 = variation that is\n                present but uncorrelated "
          "with the truth")


if __name__ == "__main__":
    main()
