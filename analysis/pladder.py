"""Transform-code upper bound: rotate (PCA = MSE-optimal linear), scalar
quantize with per-dim bit allocation, invert. Rate matched to 8x14=112b."""
import numpy as np, torch
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:0"); TIER=2; BITS=112
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
FIT=Z[TIER][:nf].reshape(-1,D); HO=Z[TIER][nf:].reshape(-1,D); e0=float(HO.pow(2).sum(-1).mean())
mu=FIT.mean(0); Fc=FIT-mu
ev,U=torch.linalg.eigh(((Fc.T@Fc)/Fc.shape[0]).double()); ev=ev.flip(0).float(); U=U.flip(1).float()
def decode(rec):
    with torch.no_grad():
        zq=[Z[t][nf:] for t in range(len(m.specs))]; zq[TIER]=rec.reshape(nw-nf,-1,D)
        out=m.decoder(zq,torch.zeros(nw-nf,len(m.specs),device=DEV))
        xh=X[nf:].to(DEV); v=xh.abs().sum(1)>0
        return float(m.source.recon_stats(out.float(),xh,v)[1]["cos"])
print(f"D={D}, rate budget {BITS} bits/cell (= 8 levels x 14 b)",flush=True)
print(f"{'D_code':>7} {'b/dim':>6} {'var kept':>9} {'resid':>8} {'decoded cos':>12} {'storage':>9}")
for P,b in ((14,1),(28,1),(56,1),(112,1),(112,2)):
    Up=U[:,:P]; sd=ev[:P].clamp_min(1e-12).sqrt()
    Ptr=(FIT-mu)@Up; Pho=(HO-mu)@Up
    lv=2**b
    # per-dim LLOYD-MAX: 1-D k-means on the fit projections (== optimal
    # companding + uniform grid), applied to holdout
    q=torch.empty_like(Pho)
    for d in range(P):
        col=Ptr[:,d]
        c=torch.quantile(col,(torch.arange(lv,device=DEV).float()+0.5)/lv)
        for _ in range(25):
            a=(col[:,None]-c[None,:]).abs().argmin(1)
            s_=torch.zeros(lv,device=DEV).index_add_(0,a,col)
            n_=torch.zeros(lv,device=DEV).index_add_(0,a,torch.ones_like(col))
            live=n_>0; c[live]=s_[live]/n_[live]
        ah=(Pho[:,d][:,None]-c[None,:]).abs().argmin(1)
        q[:,d]=c[ah]
    rec=q@Up.t()+mu
    kept=float(ev[:P].sum()/ev.sum())*100
    print(f"{P:>7} {b:>6} {kept:>8.1f}% "
          f"{float((HO-rec).pow(2).sum(-1).mean())/e0:>8.4f} {decode(rec):>12.5f} "
          f"{P*D/2**20:>7.2f}M",flush=True)
# reference: current 8-level RVQ
R=FIT.clone(); bs=[]
def km(R,K,it=8,sd_=0):
    N=R.shape[0];g=torch.Generator(device=DEV).manual_seed(sd_)
    C=R[torch.randperm(N,device=DEV,generator=g)[:K]].clone();o=torch.ones(N,device=DEV)
    for _ in range(it):
        a=_chunked_assign(R,C);s=torch.zeros_like(C).index_add_(0,a,R)
        c=torch.zeros(K,device=DEV).index_add_(0,a,o);l2=c>0
        C[l2]=s[l2]/c[l2,None];nd=int((~l2).sum())
        if nd: C[~l2]=R[torch.randint(0,N,(nd,),device=DEV)]
    return C
for l in range(8): C=km(R,16384,sd_=l); bs.append(C); R=R-C[_chunked_assign(R,C)]
Rh=HO.clone(); rec=torch.zeros_like(HO)
for C in bs:
    a=_chunked_assign(Rh,C); rec=rec+C[a]; Rh=Rh-C[a]
print(f"{'RVQ 8x16384':>14}                  {float(Rh.pow(2).sum(-1).mean())/e0:>8.4f} "
      f"{decode(rec):>12.5f} {8*16384*D/2**20:>7.0f}M",flush=True)
