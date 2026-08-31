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
from patch_intrinsic_dim import _dequant, S2_10M, S2_20M, S2_60M   # noqa: E402

from georvq.data.store import ChipStore                       # noqa: E402

DEM_NODATA, DEM_VALID_MIN = -32767.0, -1000.0
S1_NODATA, S1_DB_STEP = -32768.0, 0.1
S1_BANDS = ["vv", "vh"]


def iter_chips(source, path, n_chips, seed=0):
    """Yield (x, valid, q) per chip: dequantized float (bands,H,W), a
    validity mask, and the INTEGER representation a lossless coder would
    actually see -- which differs per source and is easy to get wrong.

      tessera/ae  int8 as stored
      dem         float32 metres; the lossless unit is the integer metre
      s2          uint16 DN, three native grids upsampled onto the 10m
                  lattice (1068 = 2*534 = 6*178, so this is exact)
      era5:<var>  int16 as stored (matches the native GRIB 16-bit
                  packing); one chip is one hourly slice of one variable
                  with pressure levels as bands, so the time axis is
                  flattened into the chip draw. "era5:t+std" standardises
                  each level by moments estimated from a held-aside draw
                  of slices; the default is physical units.
    """
    if source.startswith("era5:"):
        import zarr
        var = source.split(":", 1)[1]
        std = var.endswith("+std")
        var = var[:-4] if std else var
        root = zarr.open(os.path.join(path, "cubes.zarr"), mode="r")
        a = root[var]
        sc = np.asarray(root[var + "_scale"], dtype=np.float64)
        off = np.asarray(root[var + "_offset"], dtype=np.float64)
        pl = a.ndim == 5
        n_cubes, n_t = a.shape[0], a.shape[-3]
        rng = np.random.default_rng(seed)

        def slice_(c, t):
            q = np.asarray(a[c, :, t]) if pl else np.asarray(a[c, t])[None]
            s = sc[c] if pl else sc[c:c + 1]
            o = off[c] if pl else off[c:c + 1]
            x = ((q.astype(np.int64) + 32768) * s[:, None, None]
                 + o[:, None, None])
            return x, q

        mu = sig = None
        if std:
            xs = [slice_(int(c), int(t))[0] for c, t in zip(
                rng.integers(0, n_cubes, 16), rng.integers(0, n_t, 16))]
            flat = np.stack(xs).transpose(1, 0, 2, 3).reshape(len(xs[0]), -1)
            mu = flat.mean(1)[:, None, None]
            sig = np.maximum(flat.std(1), 1e-12)[:, None, None]
        for c, t in zip(rng.integers(0, n_cubes, n_chips),
                        rng.integers(0, n_t, n_chips)):
            x, q = slice_(int(c), int(t))
            if std:
                x = (x - mu) / sig
            valid = np.ones(x.shape[1:], dtype=bool)
            yield x.astype(np.float32), valid, q.astype(np.int32)
        return

    if source == "s1":
        # SAR gamma0 is stored as LINEAR power, where speckle is
        # multiplicative and the distribution spans orders of magnitude.
        # Both the statistics and any real codec belong in dB, where
        # speckle is additive -- so the field is analysed in dB and the
        # "lossless" integer is dB at 0.1 dB precision, which is the
        # precision a practitioner archives. Quoting linear-power entropy
        # instead would be measuring the storage format, not the data.
        import glob
        import rasterio
        scenes = sorted(glob.glob(f"{path}/*/*/*/vv.tif"))
        rng = np.random.default_rng(seed)
        for i in rng.choice(len(scenes), min(n_chips, len(scenes)),
                            replace=False):
            d = scenes[int(i)].rsplit("/", 1)[0]
            planes = []
            for b in S1_BANDS:
                with rasterio.open(f"{d}/{b}.tif") as r:
                    planes.append(r.read(1).astype(np.float32))
            lin = np.stack(planes)
            valid = np.isfinite(lin).all(0) & (lin > 0).all(0) & \
                (lin != S1_NODATA).all(0)
            db = np.zeros_like(lin)
            np.log10(lin, out=db, where=valid[None])
            db = 10.0 * db * valid[None]
            yield db, valid, np.round(db / S1_DB_STEP).astype(np.int32)
        return

    if source == "s2":
        import glob
        import rasterio
        scenes = sorted(glob.glob(f"{path}/*/*/*/B02.tif"))
        rng = np.random.default_rng(seed)
        for i in rng.choice(len(scenes), min(n_chips, len(scenes)),
                            replace=False):
            d = scenes[int(i)].rsplit("/", 1)[0]
            planes, ref = [], None
            for names, up in ((S2_10M, 1), (S2_20M, 2), (S2_60M, 6)):
                for b in names:
                    with rasterio.open(f"{d}/{b}.tif") as r:
                        a = r.read(1)
                    if up > 1:
                        a = np.repeat(np.repeat(a, up, 0), up, 1)
                    ref = a.shape[0] if ref is None else ref
                    planes.append(a[:ref, :ref])
            q = np.stack(planes).astype(np.int32)
            with rasterio.open(f"{d}/cloud_mask.tif") as r:
                cm = r.read(1)[:ref, :ref]
            valid = (cm == 0) & (q > 0).all(0)
            yield q.astype(np.float32) / 10000.0, valid, q
        return

    store = ChipStore(path)
    rows = store.index.sample(n=min(n_chips, len(store.index)),
                              random_state=seed)
    for _, r in rows.iterrows():
        sh, slot = int(r["shard"]), int(r["slot"])
        if source == "dem":
            a = np.asarray(store.open_shard(sh)[slot]).astype(np.float32)
            valid = (a[0] > DEM_VALID_MIN) & (a[0] != DEM_NODATA)
            yield a * valid[None], valid, np.round(a).astype(np.int32)
        else:
            x, valid = _dequant(store, sh, slot, source)
            yield x, valid, np.asarray(store.open_shard(sh)[slot])


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
    ap.add_argument("--source", default="tessera",
                    help="tessera | ae | dem | s2 | s1 | era5:<var>[+std]")
    ap.add_argument("--chips", type=int, default=40)
    ap.add_argument("--entropy-bands", type=int, default=16,
                    help="bands to histogram (extrapolated); the full set is "
                         "exact but slow")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    lags = [1, 2, 4, 8, 16, 32]

    n_px = n_valid = 0
    mags, ent, ac = [], {"raw": [], "left": [], "avg": []}, []
    bmin, bmax = None, None
    nchip = 0
    for x, valid, q in iter_chips(args.source, args.store, args.chips):
        nchip += 1
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
    nb = bmin.shape[0]
    print(f"{args.store}  ({args.source}, {nb} band"
          f"{'s' if nb > 1 else ''})")
    print(f"  {nchip} chips, {n_px:,} pixels, "
          f"{100*n_valid/max(n_px,1):.2f}% valid\n")

    p = np.percentile(mags, [1, 10, 50, 90, 99])
    print("PER-PIXEL VALUE" if nb == 1 else "PER-PIXEL VECTOR MAGNITUDE")
    # p99/p1 is uninformative when p1 sits at a physical floor (DEM sea
    # level is exactly 0), so report the interdecile ratio too
    print(f"  p1 {p[0]:.4f}  p10 {p[1]:.4f}  p50 {p[2]:.4f}  "
          f"p90 {p[3]:.4f}  p99 {p[4]:.4f}   p90/p10 = "
          f"{p[3]/max(p[1],1e-9):.2f}x")
    if nb == 1 or p[0] <= 0:
        print("  -> single band: direction is not a concept here; only the "
              "value distribution and its spatial structure matter")
    elif p[4] / max(p[0], 1e-9) < 1.6:
        print("  -> near-constant: direction carries the information")
    else:
        print("  -> wide: magnitude is real signal, an absolute loss is "
              "justified")

    unit = {"s1": "dB @0.1dB", "dem": "integer metre",
            "s2": "uint16 DN"}.get(args.source, "int8 as stored")
    if args.source.startswith("era5:"):
        unit = "int16 as stored (per-cube affine, GRIB-matched)"
    print(f"\nLOSSLESS RATE from spatial prediction (bits/pixel, all bands)"
          f"\n  [lossless unit: {unit}]")
    for k in ("raw", "left", "avg"):
        if ent[k]:
            m = float(np.mean(ent[k]))
            print(f"  {k:>5} predictor: {m:>8.2f} b/px")
    if ent["raw"]:
        r = float(np.mean(ent["raw"]))
        # take the BEST predictor, not a favoured one: which wins is
        # source-dependent. On Sentinel-2 `left` beats `avg`, the reverse
        # of DEM and Tessera, because the 20m/60m bands are nearest-
        # upsampled onto the 10m lattice so the horizontal neighbour is
        # often an exact duplicate while the vertical average is not.
        best_k = min((k for k in ("left", "avg") if ent[k]),
                     key=lambda k: float(np.mean(ent[k])))
        bv = float(np.mean(ent[best_k]))
        print(f"  -> best predictor is '{best_k}': saves {100*(1-bv/r):.1f}%; "
              f"a lossy codec must beat {bv:.1f} b/px to justify distortion")

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
