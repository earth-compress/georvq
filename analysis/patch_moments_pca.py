"""How low-rank are a patch's first two moments?

For each 64px window the codec must, at minimum, convey where the patch
sits in embedding space (its MEAN vector) and how much each dimension
varies inside it (its per-dimension STD vector). Both are 128-d. If either
lives on a much lower-dimensional linear subspace, that is free rate: the
coarse tier can spend k coefficients instead of 128, and the residual
coder only has to handle what is left.

This measures that directly. PCA is fitted on half the windows and scored
on the OTHER half, so "99% of variance" cannot be an artefact of fitting
128 directions to the same points we score.

Reported per quantity:
  * cumulative explained variance vs k (the usual spectrum read);
  * held-out relative L2 error of the RECONSTRUCTED VECTOR, which is what
    actually matters -- variance explained is measured after removing the
    global mean, and for a quantity with a large mean offset (both of
    these) it flatters the approximation badly;
  * held-out cosine, for the mean vector, since the codec's own metric is
    cosine.

The std vector is also fitted in LOG space. Standard deviations are
positive and vary multiplicatively -- a patch that is 2x more varied is
2x in every band -- so a linear subspace in log space is the more natural
model, and the comparison says which one a coder should use.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from georvq.data.store import ChipStore

SCALE_MAX = 1.0        # georvq.sources.tessera.TesseraSource.SCALE_MAX


def chip_windows(store, sh, slot, win, min_valid):
    """All non-overlapping `win`-px windows of one chip, dequantized.

    Returns (means, stds) as (n, bands) float32, keeping only windows whose
    valid fraction clears `min_valid` -- a window that is half nodata has a
    mean and a std, but they describe a different thing than a full one.
    """
    a = np.asarray(store.open_shard(sh)[slot])                  # (B, px, px)
    sc = np.asarray(store._aux_shards.setdefault(
        sh, np.load(f"{store.dir}/scales_{int(sh):04d}.npy", mmap_mode="r")
    )[slot]).astype(np.float32)                                 # (px, px)
    valid = (a != 0).any(0) & (sc > 0) & (sc < SCALE_MAX)
    x = a.astype(np.float32) * np.where(valid, sc, 0.0)[None]
    B, px, _ = x.shape
    ms, ss = [], []
    for r in range(0, px - win + 1, win):
        for c in range(0, px - win + 1, win):
            v = valid[r:r + win, c:c + win]
            n = int(v.sum())
            if n < min_valid * win * win:
                continue
            w = x[:, r:r + win, c:c + win].reshape(B, -1)[:, v.reshape(-1)]
            ms.append(w.mean(1))
            # ddof=1: an unbiased spread estimate, n is a few thousand so
            # the correction is cosmetic, but the quantity is a population
            # spread not a sample of one
            ss.append(w.std(1, ddof=1))
    if not ms:
        return None
    return np.stack(ms), np.stack(ss)


def collect(store, rows, n_chips, win, min_valid, workers):
    sel = rows.sample(n=min(n_chips, len(rows)), random_state=0)
    jobs = [(int(r["shard"]), int(r["slot"])) for _, r in sel.iterrows()]
    out = []
    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(
                lambda j: chip_windows(store, j[0], j[1], win, min_valid),
                jobs)):
            if res is not None:
                out.append(res)
            if (i + 1) % 100 == 0:
                got = sum(o[0].shape[0] for o in out)
                print(f"  {i+1}/{len(jobs)} chips -> {got:,} windows",
                      flush=True)
    M = np.concatenate([o[0] for o in out])
    S = np.concatenate([o[1] for o in out])
    return M, S


def load_or_collect(store_dir, cache, n_chips, n_val_chips, win, min_valid,
                    workers):
    """(M, S, Mv, Sv) — train moments and GEOGRAPHICALLY held-out val ones.

    The val windows come from the store's own held-out geographic cells,
    not from a random split of the same windows: windows cut from one chip
    are spatially correlated, so a random split would leak and every
    held-out number would read too well.
    """
    import os
    if cache and os.path.exists(cache):
        z = np.load(cache)
        print(f"loaded {z['M'].shape[0]:,} train / {z['Mv'].shape[0]:,} val "
              f"windows from {cache}")
        return z["M"], z["S"], z["Mv"], z["Sv"]
    store = ChipStore(store_dir)
    tr, va = store.split(0.04, 0)
    print(f"{store_dir}: {len(store.index):,} chips, sampling {n_chips} of "
          f"{len(tr):,} train and {n_val_chips} of {len(va):,} val chips")
    M, S = collect(store, tr, n_chips, win, min_valid, workers)
    print("  -- val split --", flush=True)
    Mv, Sv = collect(store, va, n_val_chips, win, min_valid, workers)
    if cache:
        np.savez(cache, M=M, S=S, Mv=Mv, Sv=Sv)
        print(f"cached to {cache}")
    return M, S, Mv, Sv


def pca_report(name, A, ks, cosine=False, back=None, ref=None):
    """Fit on half, score on the other half. A: (N, D) float64 torch.

    `back`/`ref`: score the reconstruction in a DIFFERENT space than the
    one PCA is fitted in -- used for the log-space std fit, whose relative
    error must be read after exp() to be comparable with the linear fit.
    Without this the log fit looks far worse purely because log-stds have a
    small mean offset and so a large relative deviation.
    """
    N, D = A.shape
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    fit, ev = A[perm[:N // 2]], A[perm[N // 2:]]
    ev_ref = ref[perm[N // 2:]] if ref is not None else None

    mu = fit.mean(0)
    C = torch.cov((fit - mu).T)
    lam, V = torch.linalg.eigh(C)
    lam, V = lam.flip(0).clamp_min(0), V.flip(1)          # descending
    cum = torch.cumsum(lam, 0) / lam.sum()

    e = ev - mu
    proj = e @ V                                           # (n, D) coeffs
    # energy of the held-out vector itself, not of its deviation: the
    # denominator for a relative error a coder would actually see
    den = ev.pow(2).sum(1).sqrt()

    if ev_ref is not None:
        den = ev_ref.pow(2).sum(1).sqrt()

    # participation ratio: the number of directions the energy actually
    # occupies. (sum lam)^2 / sum lam^2 is D for a flat spectrum and 1 for
    # a rank-1 one, so it says "low-rank or not" without picking a cutoff.
    pr = float(lam.sum() ** 2 / lam.pow(2).sum())
    d = {q: int((cum < q).sum()) + 1 for q in (0.90, 0.95, 0.99)}
    print(f"\n=== {name}  (D={D}, fit {N//2:,} / held-out {N-N//2:,}) ===")
    print(f"  ||global mean|| = {mu.norm():.4f}   "
          f"mean ||vector|| = {den.mean():.4f}   "
          f"total variance = {lam.sum():.5f}")
    print(f"  participation ratio {pr:.1f}/{D}   "
          f"components for 90/95/99% var: {d[0.90]}/{d[0.95]}/{d[0.99]}")
    hdr = f"{'k':>4} {'cum var %':>10} {'rel L2 %':>9}"
    print(hdr + (f" {'cos':>8}" if cosine else ""))
    # k=0 is the constant predictor (every window gets the global mean):
    # the baseline any PCA claim has to beat
    for k in [0] + list(ks):
        if k > D:
            continue
        rec = mu + (proj[:, :k] @ V[:, :k].T if k else 0.0)
        if back is not None:
            rec = back(rec)
        r = (ev_ref if ev_ref is not None else ev) - rec
        rel = 100 * float((r.pow(2).sum(1).sqrt() / den).mean())
        cv = 100 * float(cum[k - 1]) if k else 0.0
        line = f"{k:>4} {cv:>10.3f} {rel:>9.3f}"
        if cosine:
            cs = torch.nn.functional.cosine_similarity(
                rec, ev_ref if ev_ref is not None else ev, dim=1)
            line += f" {float(cs.mean()):>8.5f}"
        print(line)
    return lam, V, mu


def rd_curve(name, A, win=64):
    """Gaussian rate-distortion of the quantity, by reverse water-filling.

    Turns the spectrum into the number a codec designer wants: the bits per
    window needed to convey this quantity to a given relative error, if it
    were Gaussian with the measured covariance. Since the Gaussian MAXIMISES
    R(D) over all sources with a given covariance, this is an UPPER bound on
    the rate real data needs -- a coder exploiting non-Gaussian structure
    does better. Useful for "is this quantity worth a dedicated allocation",
    since a small upper bound settles the question outright.

    R(D) = 0.5 * sum_i log2(lam_i / theta) over lam_i > theta, with
    D = sum_i min(theta, lam_i); the water level theta is solved by
    bisection for each target.
    """
    mu = A.mean(0)
    lam = torch.linalg.eigvalsh(torch.cov((A - mu).T)).clamp_min(1e-12)
    energy = float(A.pow(2).sum(1).mean())     # ||vector||^2, incl. the mean
    print(f"\n--- {name}: bits/window to convey it (Gaussian bound) ---")
    print(f"{'rel L2 %':>9} {'bits/window':>12} {'b/px @64':>10}")
    for rel in (0.20, 0.10, 0.05, 0.02, 0.01):
        target = (rel ** 2) * energy           # allowed squared error
        if target >= float(lam.sum()):
            continue
        lo, hi = 1e-14, float(lam.max())
        for _ in range(200):
            th = 0.5 * (lo + hi)
            if float(torch.clamp(lam, max=th).sum()) > target:
                hi = th
            else:
                lo = th
        th = 0.5 * (lo + hi)
        bits = 0.5 * float(torch.log2(lam[lam > th] / th).sum())
        print(f"{100*rel:>9.0f} {bits:>12.1f} {bits/win**2:>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/data/tessera-samples/chips_512")
    ap.add_argument("--chips", type=int, default=750)
    ap.add_argument("--val-chips", type=int, default=150)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--min-valid", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="",
                    help="npz to cache the collected moments in (collection "
                         "reads ~25 GiB; the analysis is instant)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    M, S, _, _ = load_or_collect(args.store, args.cache, args.chips,
                                 args.val_chips, args.win, args.min_valid,
                                 args.workers)
    print(f"\ncollected {M.shape[0]:,} windows of {args.win}px "
          f"({M.shape[1]} bands)")

    Mt = torch.from_numpy(M).double()
    St = torch.from_numpy(S).double()
    ks = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    pca_report("MEAN vector", Mt, ks, cosine=True)
    _, Vs, mus = pca_report("STD vector", St, ks)
    # log space: std is positive and scales multiplicatively. Scored after
    # exp(), against the SAME linear stds, so the two tables compare.
    pca_report("STD vector, log space (scored in linear space)",
               St.clamp_min(1e-8).log(), ks, back=torch.exp, ref=St)

    # What IS the leading std direction? If it is close to the global mean
    # std vector, PC1 is just "how varied is this patch overall" -- one
    # scalar gain -- and the remaining components carry the actual shape.
    v1 = Vs[:, 0]
    v1 = v1 * torch.sign(v1.sum())          # sign is arbitrary; fix it
    print(f"\nstd PC1 vs the global mean std vector: "
          f"cos = {float(v1 @ mus / (v1.norm() * mus.norm())):.4f}, "
          f"{int((v1 > 0).sum())}/{v1.numel()} loadings positive")
    amp = St.pow(2).sum(1).sqrt()
    c1 = (St - mus) @ v1
    print(f"  corr(PC1 coefficient, ||std vector||) = "
          f"{float(torch.corrcoef(torch.stack([c1, amp]))[0, 1]):.4f}")

    rd_curve("MEAN vector", Mt)
    rd_curve("STD vector", St)

    # How much of a patch's total energy do the two moments account for?
    # Context for the numbers above: if the mean carries most of it, the
    # mean's subspace is the one that matters.
    em = float(Mt.pow(2).sum(1).mean())
    es = float(St.pow(2).sum(1).mean())
    print(f"\nper-window energy split: ||mean||^2 = {em:.4f}, "
          f"sum of per-band variances = {es:.4f} "
          f"({100*em/(em+es):.1f}% of E||x||^2 is the mean)")


if __name__ == "__main__":
    main()
