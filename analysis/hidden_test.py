"""HIDDEN cross-tier evaluation. The challenge agent never sees this.

Takes a *method* (not codebooks) and re-fits it independently on all four
pyramid tiers at production K and depth, then scores it on things the
challenge never measured:

  - per-tier index entropy and residual at full 8-level depth
  - END-TO-END decoded reconstruction cosine through the real decoder,
    which is the only number that actually matters

Methods must expose fit(X, K, n_levels) -> list of (K, D) codebooks.
"""
import numpy as np
import torch

DEV = torch.device("cuda:1")
CKPT = "/home/ali/projects/ring-space/checkpoints/tessera_uf_vq8_s2/model.pt"
N_WINDOWS = 12000
MAX_FIT = 400_000
LEVELS = 8


# ---------------- methods ----------------
def _assign(R, C, chunk=8192):
    cn = C.pow(2).sum(1)
    return torch.cat([(ch.pow(2).sum(1, keepdim=True) - 2 * ch @ C.t()
                       + cn[None]).argmin(1) for ch in R.split(chunk)])


def fit_kmeans(X, K, n_levels, iters=8, seed=0):
    """Production-style: sequential Lloyd per residual level."""
    assert X.shape[0] >= K, f"need >= K vectors, got {X.shape[0]} for K={K}"
    R = X.clone()
    g = torch.Generator(device=X.device).manual_seed(seed)
    books = []
    N = R.shape[0]
    ones = torch.ones(N, device=X.device)
    for _ in range(n_levels):
        C = R[torch.randperm(N, device=X.device, generator=g)[:K]].clone()
        for _ in range(iters):
            a = _assign(R, C)
            s = torch.zeros_like(C).index_add_(0, a, R)
            c = torch.zeros(K, device=X.device).index_add_(0, a, ones)
            live = c > 0
            C[live] = s[live] / c[live, None]
            nd = int((~live).sum())
            if nd:
                C[~live] = R[torch.randint(0, N, (nd,), device=X.device)]
        books.append(C)
        R = R - C[_assign(R, C)]
    return books


def _recon_from_boundaries(b):
    """Reconstruction points whose midpoints are exactly the boundaries b.
    c_{i+1} = 2 b_i - c_i; sweep the free start offset for feasibility."""
    L = len(b) + 1
    lo, hi = b[0] - 10 * (b[-1] - b[0] + 1e-9), b[0]
    for _ in range(200):
        c0 = 0.5 * (lo + hi)
        c = [c0]
        for i in range(L - 1):
            c.append(2 * b[i] - c[-1])
        ok = all(c[i + 1] > c[i] for i in range(L - 1))
        if ok:
            return np.array(c)
        hi = c0 if c[1] <= c[0] else hi
        lo = c0 if c[1] > c[0] else lo
    return None


def _factor3(K):
    """K -> three near-equal power-of-two factors."""
    bits = int(round(np.log2(K)))
    a = bits // 3
    f = [a, a, bits - 2 * a]
    return [2 ** x for x in sorted(f)]


def fit_bpq(X, K, n_levels, depth_frac=0.66, seed=0):
    """The challenge agent's winning construction, generalised: per level,
    a balanced product code on consecutive mid-tail PCA directions with
    quantile-placed decision boundaries."""
    R = X.clone()
    D = X.shape[1]
    books = []
    for lv in range(n_levels):
        Xc = R - R.mean(0)
        cov = (Xc.T @ Xc) / Xc.shape[0]
        ev, U = torch.linalg.eigh(cov.double())
        U = U.flip(1).float()                       # descending
        Ls = _factor3(K)
        start = int(D * depth_frac) + 4 * lv
        tri = [min(start + i, D - 1) for i in range(3)]
        P = (R @ U).double().cpu().numpy()
        cv = []
        for j, i in enumerate(tri):
            L = Ls[j]
            b = np.maximum.accumulate(
                np.quantile(P[:, i], np.arange(1, L) / L))
            c = _recon_from_boundaries(b)
            if c is None:
                c = np.quantile(P[:, i], (np.arange(L) + 0.5) / L)
            cv.append(c)
        dig = np.stack(np.unravel_index(np.arange(K), tuple(Ls)), 1)
        coef = np.stack([cv[j][dig[:, j]] for j in range(3)], 1)
        C = (torch.from_numpy(coef).float().to(X.device)
             @ U[:, tri].t()).contiguous()
        books.append(C)
        R = R - C[_assign(R, C)]
    return books


def fit_pq_top(X, K, n_levels, seed=0):
    """Product quantizer on the HIGH-variance directions: PCA-rotate,
    split the leading dims into 3 contiguous blocks (equal variance mass),
    k-means each block, Cartesian product -> explicit codebook."""
    R = X.clone(); D = X.shape[1]
    Ls = _factor3(K); books = []
    for lv in range(n_levels):
        Xc = R - R.mean(0)
        ev, U = torch.linalg.eigh(((Xc.T @ Xc) / Xc.shape[0]).double())
        U = U.flip(1).float(); evd = ev.flip(0).float()
        P = R @ U
        # choose leading dims covering 90% of variance, split into 3 blocks
        cum = evd.cumsum(0) / evd.sum()
        nd = int((cum < 0.90).sum()) + 1
        nd = max(3, min(nd, D))
        edges = [0, nd // 3, 2 * (nd // 3), nd]
        subs = []
        for j in range(3):
            sl = slice(edges[j], edges[j + 1])
            B = P[:, sl].contiguous(); k = Ls[j]
            g = torch.Generator(device=X.device).manual_seed(seed + 10 * lv + j)
            C = B[torch.randperm(B.shape[0], device=X.device, generator=g)[:k]].clone()
            ones = torch.ones(B.shape[0], device=X.device)
            for _ in range(8):
                a = _assign(B, C)
                sm = torch.zeros_like(C).index_add_(0, a, B)
                c = torch.zeros(k, device=X.device).index_add_(0, a, ones)
                live = c > 0
                C[live] = sm[live] / c[live, None]
                ndz = int((~live).sum())
                if ndz:
                    C[~live] = B[torch.randint(0, B.shape[0], (ndz,), device=X.device)]
            subs.append((sl, C))
        dig = np.stack(np.unravel_index(np.arange(K), tuple(Ls)), 1)
        Cp = torch.zeros(K, D, device=X.device)
        for j, (sl, C) in enumerate(subs):
            Cp[:, sl] = C[torch.from_numpy(dig[:, j]).to(X.device)]
        Cw = (Cp @ U.t()).contiguous()
        books.append(Cw)
        R = R - Cw[_assign(R, Cw)]
    return books


METHODS = {"kmeans": fit_kmeans, "bpq": fit_bpq, "pq_top": fit_pq_top}


# ---------------- harness ----------------
def main(method_names):
    from georvq.factory import from_checkpoint
    from georvq.data.store import ChipStore
    from georvq.data.pool import fixed_val_tokens

    m = from_checkpoint(CKPT, DEV)
    cfg = m.full_config
    store = ChipStore(cfg.data.store_dir)
    _, vr = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val_s = fixed_val_tokens(store, vr, token=cfg.model.window,
                             max_tokens=N_WINDOWS)
    NW = val_s.shape[0]
    is_ho = (torch.arange(NW) % 4 == 0)             # window-disjoint split
    MAXC = 64                                        # cells kept per fit window
    with torch.no_grad():
        xs = []
        Ztr = [[] for _ in m.specs]; Zho = [[] for _ in m.specs]
        for i in range(0, NW, 32):
            x, _ = m.source.prep(val_s[i:i + 32], DEV)
            hm = is_ho[i:i + 32].to(DEV)
            if hm.any():
                xs.append(x[hm].cpu())
            for t, z in enumerate(m.encode_levels(x)):
                z = z.float()
                if hm.any():
                    Zho[t].append(z[hm])
                if (~hm).any():
                    zf = z[~hm]
                    if zf.shape[1] > MAXC:
                        sel = torch.randperm(zf.shape[1], device=DEV)[:MAXC]
                        zf = zf[:, sel]
                    Ztr[t].append(zf.reshape(-1, zf.shape[-1]))
        X_img = torch.cat(xs)
        Ztr = [torch.cat(z) for z in Ztr]
        Zho = [torch.cat(z) for z in Zho]
    n_ho = X_img.shape[0]
    print(f"{NW} windows -> {NW - n_ho} fit / {n_ho} holdout "
          f"(window-disjoint); fit vectors/tier: "
          f"{[z.shape[0] for z in Ztr]}")

    for name in method_names:
        fit = METHODS[name]
        zq_ho, rows = [], []
        for t, sp in enumerate(m.specs):
            Xtr = Ztr[t]
            Xho = Zho[t].reshape(-1, sp.dim)
            if Xtr.shape[0] > MAX_FIT:
                sel = torch.randperm(Xtr.shape[0], device=DEV)[:MAX_FIT]
                Xtr = Xtr[sel]
            # a codebook cannot exceed the number of fitting vectors;
            # cap at N/4 so every codeword sees >=4 samples
            Keff = min(sp.codes, max(64, 1 << int(np.log2(
                max(Xtr.shape[0] // 4, 64)))))
            books = fit(Xtr, Keff, LEVELS)
            R = Xho.clone()
            e0 = float(R.pow(2).sum(1).mean())
            Hs, q = [], torch.zeros_like(Xho)
            for C in books:
                a = _assign(R, C)
                cnt = torch.bincount(a, minlength=Keff).float()
                p = cnt / cnt.sum()
                Hs.append(float(-(p[p > 0] * p[p > 0].log2()).sum()))
                q = q + C[a]
                R = R - C[a]
            rows.append((t, Keff, float(np.mean(Hs)),
                         float(R.pow(2).sum(1).mean()) / e0))
            zq_ho.append(q.reshape(n_ho, sp.cells, sp.dim))
        with torch.no_grad():
            Ls = torch.tensor([[s.depth for s in m.specs]] * n_ho,
                              device=DEV)
            out = m.decoder(zq_ho, Ls)
            xh = X_img.to(DEV)
            valid = xh.abs().sum(1) > 0          # exclude nodata pixels
            _, st = m.source.recon_stats(out.float(), xh, valid)
        print(f"\n=== {name} ===")
        print(f"{'tier':>5} {'K':>7} {'meanH':>7} {'H/nom':>7} {'residual':>9}")
        for t, K, H, res in rows:
            print(f"{t:>5} {K:>7} {H:>7.2f} {H/np.log2(K)*100:>6.1f}% "
                  f"{res:>9.4f}")
        print(f"  END-TO-END decoded cos {float(st['cos']):.4f}  "
              f"vec_rel_l2 {float(st['vec_rel_l2']):.4f}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or ["kmeans", "bpq"])
