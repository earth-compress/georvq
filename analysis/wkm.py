"""Sensitivity-weighted vs plain k-means, judged on DECODED cosine."""
import numpy as np, torch
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:0"); TIER=2; K=16384; L=8; EPS=0.25
m=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt",DEV)
cfg=m.full_config; store=ChipStore(cfg.data.store_dir)
_,vr=store.split(cfg.data.val_frac,cfg.data.split_seed)
vs=fixed_val_tokens(store,vr,token=cfg.model.window,max_tokens=3000)
X=[];Z=[[] for _ in m.specs]
with torch.no_grad():
    for i in range(0,vs.shape[0],64):
        x,_=m.source.prep(vs[i:i+64],DEV); X.append(x.cpu())
        for t,z in enumerate(m.encode_stage1(x)): Z[t].append(z.float())
X=torch.cat(X); Z=[torch.cat(z) for z in Z]; D=Z[TIER].shape[-1]
nw=Z[0].shape[0]; nf=int(nw*0.75)
# sensitivity on a small batch
with torch.no_grad():
    zs=[z[:64].clone() for z in Z]; z0=torch.zeros(64,len(m.specs),device=DEV)
    base=m.decoder(zs,z0).float(); sv=torch.zeros(D,device=DEV)
    for d in range(D):
        zp=[z.clone() for z in zs]; zp[TIER][...,d]+=EPS
        sv[d]=float((m.decoder(zp,z0).float()-base).pow(2).mean())
w=(sv/sv.mean()).sqrt()                        # scale dims by sqrt(sensitivity)
def km(R,K,it=8,sd=0):
    N=R.shape[0];g=torch.Generator(device=DEV).manual_seed(sd)
    C=R[torch.randperm(N,device=DEV,generator=g)[:K]].clone();o=torch.ones(N,device=DEV)
    for _ in range(it):
        a=_chunked_assign(R,C);s=torch.zeros_like(C).index_add_(0,a,R)
        c=torch.zeros(K,device=DEV).index_add_(0,a,o);lv=c>0
        C[lv]=s[lv]/c[lv,None];nd=int((~lv).sum())
        if nd: C[~lv]=R[torch.randint(0,N,(nd,),device=DEV)]
    return C
FIT=Z[TIER][:nf].reshape(-1,D); HO=Z[TIER][nf:].reshape(-1,D)
if FIT.shape[0]>400_000: FIT=FIT[torch.randperm(FIT.shape[0],device=DEV)[:400_000]]
print(f"sens weight range {float(w.min()):.2f}-{float(w.max()):.2f}",flush=True)
for tag,ww in (("plain",torch.ones_like(w)),("sens-weighted",w)):
    R=(FIT*ww).clone(); bs=[]
    for _ in range(L): C=km(R,K); bs.append(C); R=R-C[_chunked_assign(R,C)]
    Rh=(HO*ww).clone(); q=torch.zeros_like(Rh)
    for C in bs:
        a=_chunked_assign(Rh,C); q=q+C[a]; Rh=Rh-C[a]
    qz=(q/ww).reshape(nw-nf,-1,D)
    with torch.no_grad():
        zq=[Z[t][nf:] for t in range(len(m.specs))]; zq[TIER]=qz
        out=m.decoder(zq,torch.zeros(nw-nf,len(m.specs),device=DEV))
        xh=X[nf:].to(DEV); v=xh.abs().sum(1)>0
        _,st=m.source.recon_stats(out.float(),xh,v)
    print(f"{tag:>14}: latent resid {float(Rh.pow(2).sum(-1).mean())/float((HO*ww).pow(2).sum(-1).mean()):.4f}"
          f"   DECODED cos {float(st['cos']):.5f}",flush=True)
