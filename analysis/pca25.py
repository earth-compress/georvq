"""Naive: rotate to eigenbasis, codebooks covering only the leading 25%
of directions, zero-pad, rotate back. Decoded vs full codebooks."""
import numpy as np, torch
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:0"); TIER=2; K=16384; L=8
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
def km(R,K,it=8,sd=0):
    N=R.shape[0];g=torch.Generator(device=DEV).manual_seed(sd)
    C=R[torch.randperm(N,device=DEV,generator=g)[:K]].clone();o=torch.ones(N,device=DEV)
    for _ in range(it):
        a=_chunked_assign(R,C);s=torch.zeros_like(C).index_add_(0,a,R)
        c=torch.zeros(K,device=DEV).index_add_(0,a,o);lv=c>0
        C[lv]=s[lv]/c[lv,None];nd=int((~lv).sum())
        if nd: C[~lv]=R[torch.randint(0,N,(nd,),device=DEV)]
    return C
def decode(qz):
    with torch.no_grad():
        zq=[Z[t][nf:] for t in range(len(m.specs))]; zq[TIER]=qz.reshape(nw-nf,-1,D)
        out=m.decoder(zq,torch.zeros(nw-nf,len(m.specs),device=DEV))
        xh=X[nf:].to(DEV); v=xh.abs().sum(1)>0
        return float(m.source.recon_stats(out.float(),xh,v)[1]["cos"])
print(f"D={D}, fit {FIT.shape[0]}, holdout {HO.shape[0]}",flush=True)
# --- full codebooks (reference)
R=FIT.clone(); bs=[]
for _ in range(L): C=km(R,K); bs.append(C); R=R-C[_chunked_assign(R,C)]
Rh=HO.clone(); q=torch.zeros_like(HO)
for C in bs:
    a=_chunked_assign(Rh,C); q=q+C[a]; Rh=Rh-C[a]
print(f"{'full 384 dims':>22}: resid {float(Rh.pow(2).sum(-1).mean())/float(HO.pow(2).sum(-1).mean()):.4f}"
      f"  cos {decode(q):.5f}  storage {L*K*D/2**20:.0f}M",flush=True)
# --- eigenbasis, leading fraction only
mu=FIT.mean(0); Fc=FIT-mu
ev,U=torch.linalg.eigh(((Fc.T@Fc)/Fc.shape[0]).double()); U=U.flip(1).float()
for frac in (0.25,0.5):
    P=int(D*frac); Up=U[:,:P]
    Rp=(FIT-mu)@Up; bs2=[]
    for _ in range(L): C=km(Rp,K); bs2.append(C); Rp=Rp-C[_chunked_assign(Rp,C)]
    Hp=(HO-mu)@Up; qp=torch.zeros_like(Hp); Rr=Hp.clone()
    for C in bs2:
        a=_chunked_assign(Rr,C); qp=qp+C[a]; Rr=Rr-C[a]
    rec=qp@Up.t()+mu                       # zero-pad + inverse rotation
    resid=float((HO-rec).pow(2).sum(-1).mean())/float(HO.pow(2).sum(-1).mean())
    print(f"{f'eigen top {int(frac*100)}% ({P}d)':>22}: resid {resid:.4f}"
          f"  cos {decode(rec):.5f}  storage {L*K*P/2**20:.0f}M "
          f"({frac:.2f}x)",flush=True)
