"""Codebook bit-depth sweep on corpus-scale fit data, val-set reporting.
NF4-style: per-codeword scale + shared Lloyd-Max level set, applied
quantization-aware (each level's cascade sees the quantized book)."""
import numpy as np, torch, glob, json
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:0"); TIER=2; K=16384; L=8
m=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt",DEV)
cfg=m.full_config; store=ChipStore(cfg.data.store_dir)
man=json.load(open("/data/tessera-samples/latents_w32hd_full/manifest.json")); SC=man["scale"]
fs=sorted(glob.glob("/data/tessera-samples/latents_w32hd_full/t2_*.npy"))[:6]
A=np.concatenate([np.load(f,mmap_mode="r")[::9].reshape(-1,man["dim"]) for f in fs])
FIT=torch.from_numpy(np.ascontiguousarray(A)).to(DEV).float()*SC
if FIT.shape[0]>1_500_000: FIT=FIT[torch.randperm(FIT.shape[0],device=DEV)[:1_500_000]]
_,vr=store.split(cfg.data.val_frac,cfg.data.split_seed)
vs=fixed_val_tokens(store,vr,token=cfg.model.window,max_tokens=6000)
X=[];Zv=[[] for _ in m.specs]
with torch.no_grad():
    for i in range(0,vs.shape[0],64):
        x,_=m.source.prep(vs[i:i+64],DEV); X.append(x.cpu())
        for t,z in enumerate(m.encode_stage1(x)): Zv[t].append(z.float())
X=torch.cat(X); Zv=[torch.cat(z) for z in Zv]; D=FIT.shape[1]; NW=X.shape[0]
HO=Zv[TIER].reshape(-1,D); e0=float(HO.pow(2).sum(-1).mean())
print(f"FIT {FIT.shape[0]/1e6:.2f}M corpus vectors | VAL {NW} windows, "
      f"{HO.shape[0]/1e6:.2f}M cells | K={K} x {L} levels, dim {D}\n",flush=True)
def km(R,Kx,it=8,sd=0):
    N=R.shape[0];g=torch.Generator(device=DEV).manual_seed(sd)
    C=R[torch.randperm(N,device=DEV,generator=g)[:Kx]].clone();o=torch.ones(N,device=DEV)
    for _ in range(it):
        a=_chunked_assign(R,C);s=torch.zeros_like(C).index_add_(0,a,R)
        c=torch.zeros(Kx,device=DEV).index_add_(0,a,o);lv=c>0
        C[lv]=s[lv]/c[lv,None];nd=int((~lv).sum())
        if nd: C[~lv]=R[torch.randint(0,N,(nd,),device=DEV)]
    return C
def qbook(C,bits):
    if bits>=16: return C
    lv=2**bits
    s=C.abs().amax(1,keepdim=True).clamp_min(1e-9)
    Cn=(C/s).reshape(-1)
    sub=Cn[torch.randperm(Cn.numel(),device=DEV)[:300000]]
    g=torch.quantile(sub,(torch.arange(lv,device=DEV).float()+0.5)/lv)
    for _ in range(40):
        a=(sub[:,None]-g[None,:]).abs().argmin(1)
        sm=torch.zeros(lv,device=DEV).index_add_(0,a,sub)
        n=torch.zeros(lv,device=DEV).index_add_(0,a,torch.ones_like(sub))
        live=n>0; g[live]=sm[live]/n[live]
    out=torch.empty_like(C)
    for i in range(0,C.shape[0],4096):
        blk=C[i:i+4096]/s[i:i+4096]
        out[i:i+4096]=g[(blk[...,None]-g).abs().argmin(-1)]*s[i:i+4096]
    return out
def decode(rec):
    with torch.no_grad():
        zq=[Zv[t] for t in range(len(m.specs))]; zq[TIER]=rec.reshape(NW,-1,D)
        out=m.decoder(zq,torch.zeros(NW,len(m.specs),device=DEV))
        xh=X.to(DEV); v=xh.abs().sum(1)>0
        return float(m.source.recon_stats(out.float(),xh,v)[1]["cos"])
print(f"{'bits':>6} {'resid':>8} {'decoded cos':>12} {'dcos':>9} "
      f"{'this tier':>10} {'all 3 tiers':>12}",flush=True)
for b in (16,8,6,4,3,2):
    Rf=FIT.clone(); Rh=HO.clone(); rec=torch.zeros_like(HO)
    for l in range(L):
        C=qbook(km(Rf,K,sd=l),b)
        Rf=Rf-C[_chunked_assign(Rf,C)]
        a=_chunked_assign(Rh,C); rec=rec+C[a]; Rh=Rh-C[a]
    cos=decode(rec)
    mb=L*K*D*(b/8)/1e6 + L*K*2/1e6
    allmb=(2048+8192+16384)*L*D*(b/8)/1e6 + (2048+8192+16384)*L*2/1e6
    tag="fp16" if b==16 else f"int{b}"
    print(f"{tag:>6} {float(Rh.pow(2).sum(-1).mean())/e0:>8.4f} {cos:>12.5f} "
          f"{1-cos:>9.5f} {mb:>8.1f}MB {allmb:>10.1f}MB",flush=True)
