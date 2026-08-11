"""RVQ for N levels, then transform-code the residual at 1 bit/direction.
Rate matched to 8x14 = 112 bits/cell."""
import numpy as np, torch
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:0"); TIER=2; K=16384; BITS=112
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
if FIT.shape[0]>400_000: FIT=FIT[torch.randperm(FIT.shape[0],device=DEV)[:400_000]]
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
print(f"{'scheme':>28} {'resid':>8} {'decoded cos':>12} {'cb storage':>11}",flush=True)
for NL in (0,2,4,6,8):
    Rf=FIT.clone(); Rh=HO.clone(); rec=torch.zeros_like(HO)
    for l in range(NL):
        C=km(Rf,K,sd=l); Rf=Rf-C[_chunked_assign(Rf,C)]
        a=_chunked_assign(Rh,C); rec=rec+C[a]; Rh=Rh-C[a]
    P=BITS-NL*14                      # remaining bits -> that many 1-bit dirs
    if P>0:
        mu=Rf.mean(0); Fc=Rf-mu
        _,U=torch.linalg.eigh(((Fc.T@Fc)/Fc.shape[0]).double()); Up=U.flip(1).float()[:,:P]
        Ptr=(Rf-mu)@Up; Pho=(Rh-mu)@Up
        q=torch.empty_like(Pho)
        for d in range(P):
            col=Ptr[:,d]
            c=torch.quantile(col,torch.tensor([0.25,0.75],device=DEV))
            for _ in range(25):
                a2=(col[:,None]-c[None,:]).abs().argmin(1)
                s_=torch.zeros(2,device=DEV).index_add_(0,a2,col)
                n_=torch.zeros(2,device=DEV).index_add_(0,a2,torch.ones_like(col))
                lv2=n_>0; c[lv2]=s_[lv2]/n_[lv2]
            ah=(Pho[:,d][:,None]-c[None,:]).abs().argmin(1); q[:,d]=c[ah]
        rec=rec+(q@Up.t()+mu)
    mb=NL*K*D/1e6
    print(f"{f'RVQ x{NL} + {P} dirs x1b':>28} "
          f"{float((HO-rec).pow(2).sum(-1).mean())/e0:>8.4f} {decode(rec):>12.5f} "
          f"{mb:>9.1f}M",flush=True)
