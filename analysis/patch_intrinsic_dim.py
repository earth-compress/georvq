"""How many kinds of pixel live inside ONE patch?

`patch_moments_pca.py` asked how the patch MEANS are distributed across the
dataset. This asks the complementary question, and the one that decides
whether a per-patch palette is a sensible code: within a single 64px
window, how much of the pixel-to-pixel variation is captured by a few
directions (PCA) or a few prototypes (k-means)?

Two different notions of "simple", and they imply different codecs:

  LINEAR RANK   if d components carry the variance, a patch is a
                d-dimensional affine subspace and transform coding wins:
                send a d x 128 basis plus d coefficients per pixel.
  SPECIES       if K prototypes carry it, a patch is a palette and indexed
                coding wins: send K x 128 codewords plus log2(K) bits per
                pixel. This is the cheaper structure by far at small K --
                8 species is 3 bits/pixel.

Both are scored as fraction of WITHIN-patch variance explained, i.e.
against the patch's own mean, so the transmitted-mean baseline scores 0 by
construction and everything here is structure the mean does not already
carry (~14.5% of total energy; the mean is the other 85.5%).

The contrast that matters is LOCAL vs GLOBAL. A shared codebook fitted
across all patches -- which is what the pyramid's VQ does -- has to cover
every pixel species in the dataset. If a single patch needs far fewer, the
per-patch palette is buying a genuinely easier problem, and that gap is
the argument for spending side-channel bits on it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from georvq.data.store import ChipStore

SCALE_MAX = 1.0
AE_NODATA = -128


def _dequant(store, sh, slot, source):
    """(x, valid) for one chip: (bands, px, px) float32 and (px, px) bool.

    tessera: int8 * per-pixel float16 scale, nodata = all-bands-zero.
    ae:      signed-square dequant of int8, then per-pixel L2 NORMALISATION
             -- AlphaEarth pixels are unit vectors, so they lie on a sphere.
             That is a real difference for this measurement: the radial
             direction carries no information, so one of the 64 dimensions
             is spent by construction and the within-patch variation is
             purely angular.
    """
    a = np.asarray(store.open_shard(sh)[slot])
    if source == "ae":
        valid = a[0] != AE_NODATA
        x = (a.astype(np.float32) / 127.5)
        x = np.square(x) * np.sign(x)
        x = x * valid[None]
        n = np.linalg.norm(x, axis=0, keepdims=True)
        x = x / np.maximum(n, 1e-6)
        return x, valid
    sc = np.asarray(np.load(f"{store.dir}/scales_{int(sh):04d}.npy",
                            mmap_mode="r")[slot]).astype(np.float32)
    valid = (a != 0).any(0) & (sc > 0) & (sc < SCALE_MAX)
    return a.astype(np.float32) * np.where(valid, sc, 0.0)[None], valid


def chip_patches(store, sh, slot, win, min_valid, source="tessera"):
    """Valid-enough windows of one chip as (n, npix, bands) float32."""
    x, valid = _dequant(store, sh, slot, source)
    B, px, _ = x.shape
    out = []
    for r in range(0, px - win + 1, win):
        for c in range(0, px - win + 1, win):
            v = valid[r:r + win, c:c + win].reshape(-1)
            if v.sum() < min_valid * win * win:
                continue
            w = x[:, r:r + win, c:c + win].reshape(B, -1)[:, v]
            if w.shape[1] >= 512:            # keep the covariance sane
                out.append(w.T[:4096].copy())     # (npix, bands)
    return out


def collect(store, rows, n_chips, win, min_valid, workers, max_patches,
            source="tessera"):
    sel = rows.sample(n=min(n_chips, len(rows)), random_state=0)
    jobs = [(int(r["shard"]), int(r["slot"])) for _, r in sel.iterrows()]
    out = []
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(lambda j: chip_patches(store, j[0], j[1], win,
                                                 min_valid, source), jobs):
            out.extend(res)
            if len(out) >= max_patches:
                break
    return out[:max_patches]


@torch.no_grad()
def kmeans_ve(X, ks, iters=15, seed=0):
    """Fraction of within-patch variance explained by k prototypes.

    X: (npix, bands). k-means++ style seeding is overkill here -- with
    thousands of points and small k, furthest-point init converges fine and
    is deterministic, which matters when comparing across patches.
    """
    g = torch.Generator().manual_seed(seed)
    mu = X.mean(0)
    tss = float((X - mu).pow(2).sum())
    out = {}
    for k in ks:
        if k >= X.shape[0]:
            out[k] = 1.0
            continue
        # furthest-point init: deterministic and spreads the seeds
        idx = [int(torch.randint(X.shape[0], (1,), generator=g))]
        d = (X - X[idx[0]]).pow(2).sum(1)
        for _ in range(k - 1):
            idx.append(int(d.argmax()))
            d = torch.minimum(d, (X - X[idx[-1]]).pow(2).sum(1))
        C = X[idx].clone()
        for _ in range(iters):
            a = torch.cdist(X, C).argmin(1)
            for j in range(k):
                m = a == j
                if m.any():
                    C[j] = X[m].mean(0)
        sse = float((X - C[torch.cdist(X, C).argmin(1)]).pow(2).sum())
        out[k] = 1.0 - sse / max(tss, 1e-12)
    return out


@torch.no_grad()
def pca_ve(X, ks):
    """Fraction of within-patch variance in the top-k principal directions."""
    Xc = X - X.mean(0)
    lam = torch.linalg.eigvalsh(torch.cov(Xc.T)).flip(0).clamp_min(0)
    cum = torch.cumsum(lam, 0) / lam.sum().clamp_min(1e-12)
    ve = {k: float(cum[min(k, len(cum)) - 1]) for k in ks}
    d = {q: int((cum < q).sum()) + 1 for q in (0.90, 0.95, 0.99)}
    pr = float(lam.sum() ** 2 / lam.pow(2).sum().clamp_min(1e-12))
    return ve, d, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/data/tessera-samples/chips_512")
    ap.add_argument("--source", default="tessera", choices=["tessera", "ae"])
    ap.add_argument("--chips", type=int, default=120)
    ap.add_argument("--patches", type=int, default=384)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--min-valid", type=float, default=0.95)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    store = ChipStore(args.store)
    tr, _ = store.split(0.04, 0)
    pats = collect(store, tr, args.chips, args.win, args.min_valid,
                   args.workers, args.patches, args.source)
    print(f"{len(pats)} patches of {args.win}px "
          f"({pats[0].shape[0]} valid px x {pats[0].shape[1]} bands each)\n")

    ks = [1, 2, 4, 8, 16, 32, 64]
    P = [torch.from_numpy(p) for p in pats]

    # ---- per patch ----
    pve, kve, d90, d95, d99, prs = [], [], [], [], [], []
    for i, X in enumerate(P):
        v, d, pr = pca_ve(X, ks)
        pve.append([v[k] for k in ks]); prs.append(pr)
        d90.append(d[0.90]); d95.append(d[0.95]); d99.append(d[0.99])
        kve.append([kmeans_ve(X, ks)[k] for k in ks])
        if (i + 1) % 64 == 0:
            print(f"  {i+1}/{len(P)} patches", flush=True)
    pve = np.array(pve); kve = np.array(kve)

    print(f"\nWITHIN-PATCH linear rank: components for 90/95/99% of the "
          f"patch's own variance")
    for name, arr in (("median", np.median), ("p10", lambda a: np.percentile(a, 10)),
                      ("p90", lambda a: np.percentile(a, 90))):
        print(f"  {name:>6}: {arr(d90):>5.0f} / {arr(d95):>5.0f} / "
              f"{arr(d99):>5.0f}   (participation ratio "
              f"{arr(prs):>5.1f}/{P[0].shape[1]})")

    print(f"\nfraction of WITHIN-patch variance explained, per patch "
          f"(median across patches)")
    print(f"{'k':>4} {'PCA dims':>10} {'k-means species':>16}")
    for j, k in enumerate(ks):
        print(f"{k:>4} {np.median(pve[:, j]):>10.4f} "
              f"{np.median(kve[:, j]):>16.4f}")

    # ---- pooled across patches: the shared-codebook comparison ----
    allpix = torch.cat([X - X.mean(0) for X in P])   # mean-removed pixels
    sub = allpix[torch.randperm(allpix.shape[0])[:200000]]
    gv, gd, gpr = pca_ve(sub, ks)
    gk = kmeans_ve(sub, ks, iters=12)
    print(f"\nSAME MEASUREMENT ON POOLED mean-removed pixels from ALL "
          f"patches\n(what a SHARED codebook has to cover):")
    print(f"  90/95/99% at {gd[0.90]} / {gd[0.95]} / {gd[0.99]} dims, "
          f"participation ratio {gpr:.1f}")
    print(f"{'k':>4} {'PCA dims':>10} {'k-means species':>16}")
    for k in ks:
        print(f"{k:>4} {gv[k]:>10.4f} {gk[k]:>16.4f}")

    print(f"\nLOCAL ADVANTAGE (median per-patch minus pooled), k-means:")
    for j, k in enumerate(ks):
        print(f"  k={k:<3} per-patch {np.median(kve[:, j]):.4f} vs pooled "
              f"{gk[k]:.4f}   (+{np.median(kve[:, j]) - gk[k]:.4f})")


if __name__ == "__main__":
    main()
