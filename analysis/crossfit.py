exec(open('/data/tessera-samples/cb_private/fit_sweep.py').read().split("print(\"=== 1.")[0])
import time
K, L, F = 16384, 8, 4
print(f"cross-fit: K={K} L={L} folds={F}; baseline (plain seq) = 0.2262\n", flush=True)
t0 = time.time()
R = FIT.clone(); books = []
n = R.shape[0]; perm = torch.randperm(n, device=DEV)
for lv in range(L):
    C = km(R, K, iters=8)                      # shipped book: all data
    books.append(C)
    if lv < L - 1:                             # honest residuals for next lvl
        newR = torch.empty_like(R)
        for f in range(F):
            hold = perm[f::F]
            keep = torch.cat([perm[j::F] for j in range(F) if j != f])
            Cf = km(R[keep], K, iters=6, seed=100 + f)
            a = _chunked_assign(R[hold], Cf)
            newR[hold] = R[hold] - Cf[a]
        R = newR
    else:
        R = R - C[_chunked_assign(R, C)]
    # running eval on holdout
    Re = EV.clone()
    for Cb in books:
        Re = Re - Cb[_chunked_assign(Re, Cb)]
    print(f"  level {lv}: holdout resid {float(Re.pow(2).sum(-1).mean())/e0:.4f}"
          f"   ({time.time()-t0:.0f}s)", flush=True)
Re = EV.clone(); H = 0.0
for Cb in books:
    a = _chunked_assign(Re, Cb)
    p = torch.bincount(a, minlength=K).float(); p = p/p.sum()
    H += float(-(p[p>0]*p[p>0].log2()).sum()); Re = Re - Cb[a]
r = float(Re.pow(2).sum(-1).mean())/e0
print(f"\nCROSS-FIT final: resid {r:.4f}  ({H:.1f} emp bits)  "
      f"vs plain 0.2262 -> {(r/0.2262-1)*100:+.1f}%", flush=True)
