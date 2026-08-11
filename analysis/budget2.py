"""Same codebook storage budget2, three ways: fewer dims (eigenbasis
truncation) vs fewer bits per value (int4/int2 QAT) vs both."""
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
def qbook(C,bits):
    """per-codeword scale + shared Lloyd-Max level set (NF4-style)."""
    if bits>=8: return C
    lv=2**bits
    s=C.abs().amax(1,keepdim=True).clamp_min(1e-9)
    Cn=(C/s).reshape(-1)
    sub=Cn[torch.randperm(Cn.numel(),device=Cn.device)[:200000]]
    g=torch.quantile(sub,(torch.arange(lv,device=C.device).float()+0.5)/lv)
    for _ in range(30):
        a=(sub[:,None]-g[None,:]).abs().argmin(1)
        sm=torch.zeros(lv,device=C.device).index_add_(0,a,sub)
        n=torch.zeros(lv,device=C.device).index_add_(0,a,torch.ones_like(sub))
        live=n>0; g[live]=sm[live]/n[live]
    out=torch.empty_like(C)
    for i in range(0,C.shape[0],4096):
        blk=(C[i:i+4096]/s[i:i+4096])
        ai=(blk[...,None]-g).abs().argmin(-1)
        out[i:i+4096]=g[ai]*s[i:i+4096]
    return out
def decode(rec):
    with torch.no_grad():
        zq=[Z[t][nf:] for t in range(len(m.specs))]; zq[TIER]=rec.reshape(nw-nf,-1,D)
        out=m.decoder(zq,torch.zeros(nw-nf,len(m.specs),device=DEV))
        xh=X[nf:].to(DEV); v=xh.abs().sum(1)>0
        return float(m.source.recon_stats(out.float(),xh,v)[1]["cos"])
def run(frac,bits,tag):
    P=int(D*frac)
    if frac<1.0:
        mu=FIT.mean(0); Fc=FIT-mu
        _,U=torch.linalg.eigh(((Fc.T@Fc)/Fc.shape[0]).double()); Up=U.flip(1).float()[:,:P]
        Rf=(FIT-mu)@Up; Rh=(HO-mu)@Up
    else:
        Rf=FIT.clone(); Rh=HO.clone()
    rec=torch.zeros_like(Rh)
    for l in range(L):
        C=qbook(km(Rf,K,sd=l),bits)
        Rf=Rf-C[_chunked_assign(Rf,C)]
        a=_chunked_assign(Rh,C); rec=rec+C[a]; Rh=Rh-C[a]
    full=rec@Up.t()+mu if frac<1.0 else rec
    mb=L*K*P*(bits/8)/1e6
    print(f"{tag:>26} {float((HO-full).pow(2).sum(-1).mean())/e0:>8.4f} "
          f"{decode(full):>12.5f} {mb:>9.1f} MB",flush=True)
print(f"{'scheme':>26} {'resid':>8} {'decoded cos':>12} {'storage':>12}",flush=True)
run(1.0,8,"full 384d int8 (ref)")
print("--- 25 MB budget2 ---",flush=True)
run(0.5,8,"192d (50% dims) int8"); run(1.0,4,"384d int4")
print("--- 12.6 MB budget2 ---",flush=True)
run(0.25,8,"96d (25% dims) int8"); run(1.0,2,"384d int2"); run(0.5,4,"192d int4")
