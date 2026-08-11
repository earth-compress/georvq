"""Rate-allocation + gain-shape sweep on w32hd's healthy fine tier.
One-shot fits (no EMA), window-disjoint split, matched nominal rate."""
import numpy as np, torch
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
from georvq.model.vq_gainshape import GainShapeResidualVQ

DEV = torch.device("cuda:1"); TIER = 2
m = from_checkpoint("/tmp/w32hd_snap.pt", DEV); cfg = m.full_config
store = ChipStore(cfg.data.store_dir)
_, vr = store.split(cfg.data.val_frac, cfg.data.split_seed)
vs = fixed_val_tokens(store, vr, token=cfg.model.window, max_tokens=8000)
Z = []
with torch.no_grad():
    for i in range(0, vs.shape[0], 128):
        x, _ = m.source.prep(vs[i:i+128], DEV)
        Z.append(m.encode_levels(x)[TIER].float())
Z = torch.cat(Z); D = Z.shape[-1]
wp = torch.randperm(Z.shape[0], device=DEV); nf = int(Z.shape[0]*0.8)
FIT = Z[wp[:nf]].reshape(-1, D); EV = Z[wp[nf:]].reshape(-1, D)
if FIT.shape[0] > 500_000:
    FIT = FIT[torch.randperm(FIT.shape[0], device=DEV)[:500_000]]
e0 = float(EV.pow(2).sum(-1).mean())
print(f"w32hd fine tier: dim {D}, fit {FIT.shape[0]}, eval {EV.shape[0]}\n")

def km(R, K, iters=8, seed=0):
    N = R.shape[0]; g = torch.Generator(device=DEV).manual_seed(seed)
    C = R[torch.randperm(N, device=DEV, generator=g)[:K]].clone()
    ones = torch.ones(N, device=DEV)
    for _ in range(iters):
        a = _chunked_assign(R, C)
        s = torch.zeros_like(C).index_add_(0, a, R)
        c = torch.zeros(K, device=DEV).index_add_(0, a, ones)
        live = c > 0; C[live] = s[live]/c[live, None]
        nd = int((~live).sum())
        if nd: C[~live] = R[torch.randint(0, N, (nd,), device=DEV)]
    return C

def fit_plain(K, L):
    R = FIT.clone(); books = []
    for _ in range(L):
        C = km(R, K); books.append(C); R = R - C[_chunked_assign(R, C)]
    return books

def ev_plain(books):
    R = EV.clone(); H = 0.0
    for C in books:
        a = _chunked_assign(R, C)
        cnt = torch.bincount(a, minlength=C.shape[0]).float(); p = cnt/cnt.sum()
        H += float(-(p[p>0]*p[p>0].log2()).sum()); R = R - C[a]
    return float(R.pow(2).sum(-1).mean())/e0, H

def ev_gs(rvq):
    R = EV.clone(); H = 0.0
    for lev in rvq.levels:
        E = lev.codebook.weight.float()
        a = (R @ E.t()).argmax(-1); u = E[a]
        gh, gi = lev.quantize_gain((R*u).sum(-1))
        for idx, n in ((a, lev.num_codes), (gi, lev.gain_book.numel())):
            c = torch.bincount(idx, minlength=n).float(); p = c/c.sum()
            H += float(-(p[p>0]*p[p>0].log2()).sum())
        R = R - gh.unsqueeze(-1)*u
    return float(R.pow(2).sum(-1).mean())/e0, H

print("=== 1. RATE ALLOCATION (plain VQ, ~112 nominal bits/cell) ===")
print(f"{'K':>7} {'levels':>7} {'nominal b':>10} {'emp bits':>9} {'residual':>10}")
for K, L in ((16384,8),(4096,9),(4096,10),(1024,11),(256,14)):
    b = fit_plain(K, L); r, H = ev_plain(b)
    print(f"{K:>7} {L:>7} {L*np.log2(K):>10.0f} {H:>9.1f} {r:>10.4f}")

print("\n=== 2. GAIN-SHAPE vs PLAIN VQ at matched nominal rate ===")
print(f"{'method':>28} {'nominal b':>10} {'emp bits':>9} {'residual':>10}")
b = fit_plain(16384, 8); r, H = ev_plain(b)
print(f"{'plain VQ K=16384 x8':>28} {8*14:>10} {H:>9.1f} {r:>10.4f}")
for Kg, gb in ((1024,4),(4096,2)):
    rvq = GainShapeResidualVQ(num_levels=8, num_codes=Kg, dim=D,
                              bf16_scores=False, gain_bits=gb).to(DEV)
    rvq.spherize_books(FIT, iters=8)
    r, H = ev_gs(rvq)
    print(f"{f'gain-shape K={Kg} +{gb}b x8':>28} "
          f"{8*(np.log2(Kg)+gb):>10.0f} {H:>9.1f} {r:>10.4f}")
