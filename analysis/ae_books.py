import numpy as np, torch, pandas as pd
from georvq.compat import from_legacy_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV = torch.device("cuda:1")
m, cfg = from_legacy_checkpoint(
    "/home/ali/projects/ring-space/checkpoints/ae_pyr64_divfull/model_best.pt", DEV)
store = ChipStore(cfg.data.store_dir)
vr = pd.read_parquet("/home/ali/projects/ring-space/checkpoints/ae_pyr64_divfull/val_chips.parquet")
vs = fixed_val_tokens(store, vr, token=cfg.model.window, max_tokens=3000)
X, Z = [], [[] for _ in m.specs]
with torch.no_grad():
    for i in range(0, vs.shape[0], 64):
        x, _ = m.source.prep(vs[i:i+64].to(DEV), DEV)
        X.append(x.cpu())
        for t, z in enumerate(m.encode_levels(x)): Z[t].append(z.float())
X = torch.cat(X); Z = [torch.cat(z) for z in Z]
nw = Z[0].shape[0]; wp = torch.randperm(nw, device=DEV); nf = int(nw*0.75)
def km(R, K, iters=8, seed=0):
    N=R.shape[0]; g=torch.Generator(device=DEV).manual_seed(seed)
    C=R[torch.randperm(N,device=DEV,generator=g)[:K]].clone(); ones=torch.ones(N,device=DEV)
    for _ in range(iters):
        a=_chunked_assign(R,C); s=torch.zeros_like(C).index_add_(0,a,R)
        c=torch.zeros(K,device=DEV).index_add_(0,a,ones); live=c>0
        C[live]=s[live]/c[live,None]; nd=int((~live).sum())
        if nd: C[~live]=R[torch.randint(0,N,(nd,),device=DEV)]
    return C
print(f"AE ae_pyr64_divfull: {nw} windows, {nf} fit / {nw-nf} holdout\n")
print(f"{'tier':>4} {'K':>6} {'trained resid':>14} {'fresh resid':>12} {'delta':>8}")
res = {}
for name in ("trained", "fresh"):
    zq = []
    for t, sp in enumerate(m.specs):
        Ztr = Z[t][wp[:nf]].reshape(-1, sp.dim); Zho = Z[t][wp[nf:]].reshape(-1, sp.dim)
        if Ztr.shape[0] > 500_000: Ztr = Ztr[torch.randperm(Ztr.shape[0],device=DEV)[:500_000]]
        if name == "trained":
            books = [l.codebook.weight.float() for l in m.rvqs[t].levels]
        else:
            R = Ztr.clone(); books = []
            for _ in range(sp.depth):
                C = km(R, sp.codes); books.append(C); R = R - C[_chunked_assign(R, C)]
        R = Zho.clone(); q = torch.zeros_like(Zho)
        for C in books:
            a = _chunked_assign(R, C); q = q + C[a]; R = R - C[a]
        res.setdefault(name, []).append(float(R.pow(2).sum(-1).mean())
                                        / float(Zho.pow(2).sum(-1).mean()))
        zq.append(q.reshape(nw-nf, sp.cells, sp.dim))
    with torch.no_grad():
        Ls = torch.tensor([[s.depth for s in m.specs]]*(nw-nf), device=DEV)
        out = m.decoder(zq, Ls); xh = X[wp[nf:].cpu()].to(DEV)
        v = xh.abs().sum(1) > 0
        _, st = m.source.recon_stats(out.float(), xh, v)
    res[name+"_cos"] = float(st["cos"])
for t, sp in enumerate(m.specs):
    d = res["fresh"][t] - res["trained"][t]
    print(f"{t:>4} {sp.codes:>6} {res['trained'][t]:>14.4f} {res['fresh'][t]:>12.4f} {d:>+8.4f}")
print(f"\nEND-TO-END decoded cos: trained {res['trained_cos']:.4f}  "
      f"fresh {res['fresh_cos']:.4f}  ({res['fresh_cos']-res['trained_cos']:+.4f})")
