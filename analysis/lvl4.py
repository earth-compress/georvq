"""4-level RVQ, per-level eigenbasis computed in that level's RESIDUAL
space. Per-level latent residual + decoded cosine, full vs truncated."""
import numpy as np, torch
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:0"); TIER=2; K=16384; L=4
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
FIT=Z[TIER][:nf].reshape(-1,D); HO=Z[TIER][nf:].reshape(-1,D)
if FIT.shape[0]>400_000: FIT=FIT[torch.randperm(FIT.shape[0],device=DEV)[:400_000]]
e0=float(HO.pow(2).sum(-1).mean())
def km(R,K,it=8,sd=0):
    N=R.shape[0];g=torch.Generator(device=DEV).manual_seed(sd)
    C=R[torch.randperm(N,device=DEV,generator=g)[:K]].clone();o=torch.ones(N,device=DEV)
    for _ in range(it):
        a=_chunked_assign(R,C);s=torch.zeros_like(C).index_add_(0,a,R)
        c=torch.zeros(K,device=DEV).index_add_(0,a,o);lv=c>0
        C[lv]=s[lv]/c[lv,None];nd=int((~lv).sum())
        if nd: C[~lv]=R[torch.randint(0,N,(nd,),device=DEV)]
    return C
def decode(rec):
    with torch.no_grad():
        zq=[Z[t][nf:] for t in range(len(m.specs))]; zq[TIER]=rec.reshape(nw-nf,-1,D)
        out=m.decoder(zq,torch.zeros(nw-nf,len(m.specs),device=DEV))
        xh=X[nf:].to(DEV); v=xh.abs().sum(1)>0
        return float(m.source.recon_stats(out.float(),xh,v)[1]["cos"])
for frac in (1.0,0.5,0.25):
    P=int(D*frac)
    Rf=FIT.clone(); Rh=HO.clone(); rec=torch.zeros_like(HO)
    print(f"\n=== eigen top {int(frac*100)}% ({P} of {D} dims/level) ===",flush=True)
    print(f"{'lvl':>4} {'resid':>8} {'this lvl':>9} {'decoded cos':>12} {'cum storage':>12}")
    for l in range(L):
        if frac<1.0:                       # per-level eigenbasis on RESIDUAL
            mu=Rf.mean(0); Fc=Rf-mu
            _,U=torch.linalg.eigh(((Fc.T@Fc)/Fc.shape[0]).double())
            Up=U.flip(1).float()[:,:P]
            C=km((Rf-mu)@Up,K,sd=l)
            a=_chunked_assign((Rf-mu)@Up,C); Rf=Rf-(C[a]@Up.t()+mu)
            ah=_chunked_assign((Rh-mu)@Up,C); step=C[ah]@Up.t()+mu
        else:
            C=km(Rf,K,sd=l); Rf=Rf-C[_chunked_assign(Rf,C)]
            step=C[_chunked_assign(Rh,C)]
        prev=float(Rh.pow(2).sum(-1).mean())/e0
        Rh=Rh-step; rec=rec+step
        cur=float(Rh.pow(2).sum(-1).mean())/e0
        print(f"{l:>4} {cur:>8.4f} {cur/prev:>9.3f} {decode(rec):>12.5f} "
              f"{(l+1)*K*P/2**20:>10.0f}M",flush=True)
