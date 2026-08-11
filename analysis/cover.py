import numpy as np, torch
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:0"); K=16384; L=8
m=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt",DEV)
cfg=m.full_config; store=ChipStore(cfg.data.store_dir)
_,vr=store.split(cfg.data.val_frac,cfg.data.split_seed)
vs=fixed_val_tokens(store,vr,token=cfg.model.window,max_tokens=4000)
Z=[]
with torch.no_grad():
    for i in range(0,vs.shape[0],64):
        x,_=m.source.prep(vs[i:i+64],DEV); Z.append(m.encode_stage1(x)[2].float())
Z=torch.cat(Z); D=Z.shape[-1]
wp=torch.randperm(Z.shape[0],device=DEV); nf=int(Z.shape[0]*0.8)
FIT=Z[wp[:nf]].reshape(-1,D); EV=Z[wp[nf:]].reshape(-1,D)
if FIT.shape[0]>500_000: FIT=FIT[torch.randperm(FIT.shape[0],device=DEV)[:500_000]]
e0=float(EV.pow(2).sum(-1).mean())
def km(R,K,it=8,sd=0):
    N=R.shape[0];g=torch.Generator(device=DEV).manual_seed(sd)
    C=R[torch.randperm(N,device=DEV,generator=g)[:K]].clone();o=torch.ones(N,device=DEV)
    for _ in range(it):
        a=_chunked_assign(R,C);s=torch.zeros_like(C).index_add_(0,a,R)
        c=torch.zeros(K,device=DEV).index_add_(0,a,o);lv=c>0
        C[lv]=s[lv]/c[lv,None];nd=int((~lv).sum())
        if nd: C[~lv]=R[torch.randint(0,N,(nd,),device=DEV)]
    return C
def masks(D,frac):
    i=torch.arange(D,device=DEV); out=[]
    for l in range(L):
        per=int(1/frac); pat=l%per
        mk=((i//int(D*frac))%per==pat)
        out.append(mk)
    return out
print(f"w32hd fine tier D={D}, fit {FIT.shape[0]}, eval {EV.shape[0]}",flush=True)
# standard
R=FIT.clone(); bs=[]
for _ in range(L): C=km(R,K); bs.append(C); R=R-C[_chunked_assign(R,C)]
Re=EV.clone()
for C in bs: Re=Re-C[_chunked_assign(Re,C)]
std=float(Re.pow(2).sum(-1).mean())/e0
print(f"standard : resid {std:.4f}   storage {L*K*D/2**20:.0f}M vals",flush=True)

for frac,name in ((0.5,"50%"),(0.25,"25%"),(0.125,"12.5%")):
    MK=masks(D,frac); R=FIT.clone(); bs2=[]
    for l in range(L):
        mk=MK[l]; C=km(R[:,mk],K,sd=l); bs2.append((mk,C))
        a=_chunked_assign(R[:,mk],C); R=R.clone(); R[:,mk]=R[:,mk]-C[a]
    Re=EV.clone()
    for mk,C in bs2:
        a=_chunked_assign(Re[:,mk],C); Re=Re.clone(); Re[:,mk]=Re[:,mk]-C[a]
    r=float(Re.pow(2).sum(-1).mean())/e0
    print(f"masked {name:>6}: resid {r:.4f}  ({r/std:.3f}x std)  "
          f"storage {L*K*int(D*frac)/2**20:.0f}M vals "
          f"({frac:.3f}x)  hits/dim {int(L*frac)}",flush=True)
