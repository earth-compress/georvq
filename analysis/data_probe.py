"""First look at a new data source, before any model exists.

Answers the questions that decide how to build a codec for it, none of
which need a checkpoint:

  VALIDITY    what fraction of pixels are nodata, and by what convention?
              (Getting this wrong silently poisons every later statistic:
              nodata is exactly zero in some stores and -128 in others, and
              counting it drags every mean toward the fill value.)
  MAGNITUDE   how much does per-pixel vector length vary? Near-constant
              magnitude means direction carries the information and a
              cosine objective is defensible; wide spread means an
              absolute loss is measuring something real.
  LOSSLESS    what rate does simple spatial prediction already achieve?
              Per-band entropy of the raw values, of the left-neighbour
              residual, and of the (left+up)/2 residual gives the
              rANS-achievable bits/pixel. This is the number a lossy codec
              has to beat by enough to justify its distortion -- and it is
              measured on the STORED integers, which is what a real
              lossless coder would see.
  STRUCTURE   spatial autocorrelation of the source at increasing lags,
              i.e. how much redundancy there is to exploit and at what
              scale.

Sources are handled by reusing the loaders in patch_intrinsic_dim.py
(tessera / ae / s2), so this runs on any of them unchanged.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_intrinsic_dim import _dequant                      # noqa: E402

from georvq.data.store import ChipStore                       # noqa: E402


def entropy_bits(counts):
    """Shannon entropy of a count vector, in bits. 0 for a dead symbol set."""
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum())


def predictive_entropy(q, valid, max_bands=None):
    """Per-band entropy of raw / left-residual / (left+up)/2-residual.

    q: (bands, H, W) integer array as STORED. Returns bits per pixel per
    band for each predictor, summed over bands = bits/pixel for the whole
    vector. Only positions whose three contributing pixels are all valid
    are counted, so nodata cannot manufacture a spuriously predictable
    residual.
    """
    B = q.shape[0] if max_bands is None else min(q.shape[0], max_bands)
    v = valid[1:, 1:] & valid[1:, :-1] & valid[:-1, 1:]
    if not v.any():
        return None
    cur = q[:B, 1:, 1:].astype(np.int32)
    left = q[:B, 1:, :-1].astype(np.int32)
    up = q[:B, :-1, 1:].astype(np.int32)
    preds = {"raw": cur,
             "left": cur - left,
             "avg": cur - ((left + up) >> 1)}
    out = {}
    for k, t in preds.items():
        tot = 0.0
        for b in range(B):
            d = t[b][v]
            lo = int(d.min())
            cnt = np.bincount((d - lo).ravel())
            tot += entropy_bits(cnt.astype(np.float64))
        out[k] = tot * (q.shape[0] / B)      # extrapolate if bands sampled
    return out


def autocorr(x, valid, lags):
    """Band-centred spatial autocorrelation of the dequantized field."""
    t = torch.from_numpy(x).float()
    m = torch.from_numpy(valid)
    t = t - t.reshape(t.shape[0], -1)[:, m.reshape(-1)].mean(1)[:, None, None]
    t = t * m[None]
    den = t.pow(2).sum(0)[m].mean().clamp_min(1e-12)
    out = {}
    for d in lags:
        h = (t[:, :, :-d] * t[:, :, d:]).sum(0)
        hv = m[:, :-d] & m[:, d:]
        w = (t[:, :-d, :] * t[:, d:, :]).sum(0)
        wv = m[:-d, :] & m[d:, :]
        num = (h[hv].sum() + w[wv].sum()) / (hv.sum() + wv.sum()).clamp_min(1)
        out[d] = float(num / den)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--source", default="tessera", choices=["tessera", "ae"])
    ap.add_argument("--chips", type=int, default=40)
    ap.add_argument("--entropy-bands", type=int, default=16,
                    help="bands to histogram (extrapolated); the full set is "
                         "exact but slow")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    store = ChipStore(args.store)
    rows = store.index.sample(n=min(args.chips, len(store.index)),
                              random_state=0)
    lags = [1, 2, 4, 8, 16, 32]

    n_px = n_valid = 0
    mags, ent, ac = [], {"raw": [], "left": [], "avg": []}, []
    bmin, bmax = None, None
    for _, r in rows.iterrows():
        sh, slot = int(r["shard"]), int(r["slot"])
        x, valid = _dequant(store, sh, slot, args.source)
        q = np.asarray(store.open_shard(sh)[slot])
        n_px += valid.size
        n_valid += int(valid.sum())
        if not valid.any():
            continue
        v = x[:, valid]
        mags.append(np.linalg.norm(v, axis=0))
        bmin = v.min(1) if bmin is None else np.minimum(bmin, v.min(1))
        bmax = v.max(1) if bmax is None else np.maximum(bmax, v.max(1))
        e = predictive_entropy(q, valid, args.entropy_bands)
        if e:
            for k in ent:
                ent[k].append(e[k])
        ac.append(autocorr(x, valid, lags))

    mags = np.concatenate(mags)
    print(f"{args.store}  ({args.source})")
    print(f"  {len(rows)} chips, {n_px:,} pixels, "
          f"{100*n_valid/max(n_px,1):.2f}% valid\n")

    p = np.percentile(mags, [1, 10, 50, 90, 99])
    print("PER-PIXEL VECTOR MAGNITUDE")
    print(f"  p1 {p[0]:.4f}  p10 {p[1]:.4f}  p50 {p[2]:.4f}  "
          f"p90 {p[3]:.4f}  p99 {p[4]:.4f}   p99/p1 = {p[4]/max(p[0],1e-9):.2f}x")
    print(f"  -> {'near-constant: direction carries the information'
                 if p[4]/max(p[0],1e-9) < 1.6 else
                 'wide: magnitude is real signal, an absolute loss is justified'}")

    print(f"\nLOSSLESS RATE from spatial prediction (bits/pixel, all bands)")
    for k in ("raw", "left", "avg"):
        if ent[k]:
            m = float(np.mean(ent[k]))
            print(f"  {k:>5} predictor: {m:>8.2f} b/px")
    if ent["raw"] and ent["avg"]:
        r, a = float(np.mean(ent["raw"])), float(np.mean(ent["avg"]))
        print(f"  -> spatial prediction saves {100*(1-a/r):.1f}%; a lossy "
              f"codec must beat {a:.1f} b/px to be worth its distortion")

    print(f"\nSOURCE SPATIAL AUTOCORRELATION")
    mean_ac = {d: float(np.mean([a[d] for a in ac])) for d in lags}
    print("  " + "  ".join(f"lag{d}={mean_ac[d]:.3f}" for d in lags))
    half = next((d for d in lags if mean_ac[d] < 0.5), None)
    print(f"  -> correlation falls below 0.5 at lag "
          f"{half if half else '>%d' % lags[-1]} px")

    print(f"\nPER-BAND RANGE  min {bmin.min():.4f}..{bmin.max():.4f}, "
          f"max {bmax.min():.4f}..{bmax.max():.4f}")


if __name__ == "__main__":
    main()
