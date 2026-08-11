"""Fit RVQ on w32hd stage-1 (NORMALIZED) latents, then test the
UNFINETUNED stage-1 decoder across matryoshka truncations."""
import numpy as np, torch
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:1")
m=from_checkpoint("/tmp/w32_now.pt",DEV); cfg=m.full_config
KS=[2048,8192,16384]; L=8
store=ChipStore(cfg.data.store_dir); _,vr=store.split(cfg.data.val_frac,cfg.data.split_seed)
vs=fixed_val_tokens(store,vr,token=cfg.model.window,max_tokens=6000)
X=[];Z=[[] for _ in m.specs]
with torch.no_grad():
    for i in range(0,vs.shape[0],128):
        x,_=m.source.prep(vs[i:i+128],DEV); X.append(x.cpu())
        for t,z in enumerate(m.encode_stage1(x)): Z[t].append(z.float())
X=torch.cat(X); Z=[torch.cat(z) for z in Z]
nw=Z[0].shape[0]; nf=int(nw*0.75)
print(f"w32hd @step {torch.load('/tmp/w32_now.pt',map_location='cpu',weights_only=False).get('step')}: "
      f"{nw} windows, {nf} fit",flush=True)
def km(R,K,it=8,sd=0):
    N=R.shape[0];g=torch.Generator(device=DEV).manual_seed(sd)
    C=R[torch.randperm(N,device=DEV,generator=g)[:K]].clone();o=torch.ones(N,device=DEV)
    for _ in range(it):
        a=_chunked_assign(R,C);s=torch.zeros_like(C).index_add_(0,a,R)
        c=torch.zeros(K,device=DEV).index_add_(0,a,o);lv=c>0
        C[lv]=s[lv]/c[lv,None];nd=int((~lv).sum())
        if nd: C[~lv]=R[torch.randint(0,N,(nd,),device=DEV)]
    return C
BK=[]
for t,sp in enumerate(m.specs):
    R=Z[t][:nf].reshape(-1,sp.dim)
    if R.shape[0]>400_000: R=R[torch.randperm(R.shape[0],device=DEV)[:400_000]]
    bs=[]
    for _ in range(L):
        C=km(R,KS[t]);bs.append(C);R=R-C[_chunked_assign(R,C)]
    BK.append(bs); print(f"  tier {t} fitted (K={KS[t]})",flush=True)
PX=cfg.model.window**2
print(f"\n{'tuple':>10} {'bits/px':>8} {'cos':>8}  (stage-1 decoder, NO finetune)",flush=True)
for tup in ((8,1,0),(8,4,0),(8,8,0),(8,8,2),(8,8,4),(8,8,8)):
    zq=[];sig=[]
    for t,sp in enumerate(m.specs):
        H=Z[t][nf:].reshape(-1,sp.dim); R=H.clone(); q=torch.zeros_like(H)
        for l in range(tup[t]):
            a=_chunked_assign(R,BK[t][l]); q=q+BK[t][l][a]; R=R-BK[t][l][a]
        rel=float(R.pow(2).sum(-1).mean())/float(H.pow(2).sum(-1).mean())
        zq.append(q.reshape(nw-nf,sp.cells,sp.dim)); sig.append(min((12*rel)**0.5,4.0))
    with torch.no_grad():
        S=torch.tensor([sig]*(nw-nf),device=DEV,dtype=torch.float)
        out=m.decoder(zq,S); xh=X[nf:].to(DEV); v=xh.abs().sum(1)>0
        _,st=m.source.recon_stats(out.float(),xh,v)
    bits=sum(m.specs[t].cells*tup[t]*np.log2(KS[t]) for t in range(3))/PX
    print(f"{'-'.join(map(str,tup)):>10} {bits:>8.1f} {float(st['cos']):>8.4f}",flush=True)
