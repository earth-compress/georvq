"""Branching ladders with GREEDY WATER-FILLING allocation, traced bit-by-bit.
Per-direction Lloyd-Max distortion tables fitted empirically (0..6 bits),
sensitivity-weighted marginal gain, decode every 7 bits."""
import numpy as np, torch, glob, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:0"); TIER=2; K=16384; TOT=112; MAXB=6; DEC_EVERY=7
m=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt",DEV)
cfg=m.full_config; store=ChipStore(cfg.data.store_dir)
man=json.load(open("/data/tessera-samples/latents_w32hd_full/manifest.json")); SC=man["scale"]
fs=sorted(glob.glob("/data/tessera-samples/latents_w32hd_full/t2_*.npy"))[:4]
A=np.concatenate([np.load(f,mmap_mode="r")[::11].reshape(-1,man["dim"]) for f in fs])
FIT=torch.from_numpy(np.ascontiguousarray(A)).to(DEV).float()*SC
if FIT.shape[0]>1_200_000: FIT=FIT[torch.randperm(FIT.shape[0],device=DEV)[:1_200_000]]
_,vr=store.split(cfg.data.val_frac,cfg.data.split_seed)
vs=fixed_val_tokens(store,vr,token=cfg.model.window,max_tokens=6000)
X=[];Zv=[[] for _ in m.specs]
with torch.no_grad():
    for i in range(0,vs.shape[0],64):
        x,_=m.source.prep(vs[i:i+64],DEV); X.append(x.cpu())
        for t,z in enumerate(m.encode_stage1(x)): Zv[t].append(z.float())
X=torch.cat(X); Zv=[torch.cat(z) for z in Zv]; D=FIT.shape[1]; NW=X.shape[0]
HO=Zv[TIER].reshape(-1,D); e0=float(HO.pow(2).sum(-1).mean())
print(f"FIT {FIT.shape[0]/1e6:.2f}M | EVAL {NW} windows | D={D}",flush=True)
def km(R,Kx,it=8,sd=0):
    N=R.shape[0];g=torch.Generator(device=DEV).manual_seed(sd)
    C=R[torch.randperm(N,device=DEV,generator=g)[:Kx]].clone();o=torch.ones(N,device=DEV)
    for _ in range(it):
        a=_chunked_assign(R,C);s=torch.zeros_like(C).index_add_(0,a,R)
        c=torch.zeros(Kx,device=DEV).index_add_(0,a,o);lv=c>0
        C[lv]=s[lv]/c[lv,None];nd=int((~lv).sum())
        if nd: C[~lv]=R[torch.randint(0,DEV and N,(nd,),device=DEV)]
    return C
def lloyd(col,lv,it=20):
    c=torch.quantile(col,(torch.arange(lv,device=DEV).float()+0.5)/lv)
    for _ in range(it):
        a=(col[:,None]-c[None,:]).abs().argmin(1)
        s_=torch.zeros(lv,device=DEV).index_add_(0,a,col)
        n_=torch.zeros(lv,device=DEV).index_add_(0,a,torch.ones_like(col))
        m_=n_>0; c[m_]=s_[m_]/n_[m_]
    return c
def decode(rec):
    with torch.no_grad():
        zq=[Zv[t] for t in range(len(m.specs))]; zq[TIER]=rec.reshape(NW,-1,D)
        out=m.decoder(zq,torch.zeros(NW,len(m.specs),device=DEV))
        xh=X.to(DEV); v=xh.abs().sum(1)>0
        return float(m.source.recon_stats(out.float(),xh,v)[1]["cos"])
# per-direction decoder sensitivity (probe once, on the raw latent basis)
with torch.no_grad():
    zs=[z[:48].clone() for z in Zv]; z0=torch.zeros(48,len(m.specs),device=DEV)
    base=m.decoder(zs,z0).float()
res={}
for N in (0,2,4,8):
    Rf=FIT.clone(); Rh=HO.clone(); rec=torch.zeros_like(HO); bits=[];dc=[];rr=[]
    for l in range(N):
        C=km(Rf,K,sd=l); Rf=Rf-C[_chunked_assign(Rf,C)]
        a=_chunked_assign(Rh,C); rec=rec+C[a]; Rh=Rh-C[a]
        bits.append((l+1)*14); rr.append(float(Rh.pow(2).sum(-1).mean())/e0)
        dc.append(1-decode(rec))
    B=TOT-N*14
    if B>0:
        mu=Rf.mean(0); Fc=Rf-mu
        evl,U=torch.linalg.eigh(((Fc.T@Fc)/Fc.shape[0]).double())
        U=U.flip(1).float(); evl=evl.flip(0).float().clamp_min(1e-12)
        P=min(D,B)                      # never more directions than bits
        Up=U[:,:P]; Ptr=(Rf-mu)@Up; Pho=(Rh-mu)@Up
        # sensitivity along these directions
        sv=torch.ones(P,device=DEV)
        with torch.no_grad():
            for k in range(P):
                zp=[z.clone() for z in zs]; zp[TIER]=zp[TIER]+0.25*Up[:,k]
                sv[k]=float((m.decoder(zp,z0).float()-base).pow(2).mean())
        sv=sv/sv.mean()
        # empirical Lloyd-Max distortion table D_k(b)
        LV={}; Dk=torch.zeros(P,MAXB+1,device=DEV)
        var=Ptr.var(0)
        Dk[:,0]=var
        for b in range(1,MAXB+1):
            for k in range(P):
                c=lloyd(Ptr[:,k],2**b); LV[(k,b)]=c
                q=c[(Ptr[:,k][:,None]-c[None,:]).abs().argmin(1)]
                Dk[k,b]=(Ptr[:,k]-q).pow(2).mean()
        alloc=torch.zeros(P,dtype=torch.long,device=DEV)
        gain=sv*(Dk[:,0]-Dk[:,1])
        for step in range(B):
            k=int(gain.argmax())
            alloc[k]+=1
            b=int(alloc[k])
            gain[k]=sv[k]*(Dk[k,b]-Dk[k,b+1]) if b<MAXB else -1e30
            if (step+1)%DEC_EVERY==0 or step==B-1:
                q=torch.zeros_like(Pho)
                for k2 in range(P):
                    bb=int(alloc[k2])
                    if bb>0:
                        c=LV[(k2,bb)]
                        q[:,k2]=c[(Pho[:,k2][:,None]-c[None,:]).abs().argmin(1)]
                part=q@Up.t()+mu
                bits.append(N*14+step+1)
                rr.append(float((Rh-part).pow(2).sum(-1).mean())/e0)
                dc.append(1-decode(rec+part))
        sh=torch.bincount(alloc,minlength=MAXB+1).tolist()
        print(f"  N={N}: alloc counts by bits {sh} (max {int(alloc.max())}b) "
              f"final dcos {dc[-1]:.5f}",flush=True)
    else:
        print(f"  N={N}: pure RVQ, final dcos {dc[-1]:.5f}",flush=True)
    res[N]=(bits,dc,rr)
fig,ax=plt.subplots(1,2,figsize=(13,5.2),constrained_layout=True)
cols=plt.cm.viridis(np.linspace(0,.85,len(res)))
for (N,(b,d,r)),c in zip(sorted(res.items()),cols):
    lab=f"RVQ x{N}" + (f" + water-filled {TOT-N*14}b" if N*14<TOT else " (pure)")
    ax[0].plot(b,d,"o-",color=c,lw=2,ms=3,label=lab)
    ax[1].plot(b,r,"o-",color=c,lw=2,ms=3,label=lab)
ax[0].set(yscale="log",xlabel="bits / cell",ylabel="1 - decoded cos",title="Decoded error (log)")
ax[1].set(yscale="log",xlabel="bits / cell",ylabel="latent residual E/E0",title="Latent residual (log)")
for a_ in ax: a_.legend(frameon=False,fontsize=8); a_.grid(alpha=.25)
fig.savefig("/data/tessera-samples/reports/waterfill_ladders.png",dpi=140)
print("saved -> /data/tessera-samples/reports/waterfill_ladders.png",flush=True)
