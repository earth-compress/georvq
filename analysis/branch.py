"""Branching rate ladders: RVQ trunk + transform branches at each split.
FIT from the large corpus latent store; EVAL on held-out val windows."""
import numpy as np, torch, glob, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:0"); TIER=2; K=16384; STEP=14; NP=8
m=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt",DEV)
cfg=m.full_config; store=ChipStore(cfg.data.store_dir)
man=json.load(open("/data/tessera-samples/latents_w32hd_full/manifest.json")); SC=man["scale"]
fs=sorted(glob.glob("/data/tessera-samples/latents_w32hd_full/t2_*.npy"))[:4]
A=np.concatenate([np.load(f,mmap_mode="r")[::11].reshape(-1,man["dim"]) for f in fs])
FIT=(torch.from_numpy(np.ascontiguousarray(A)).to(DEV).float()*SC)
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
print(f"FIT {FIT.shape[0]/1e6:.2f}M vectors (corpus store) | EVAL {NW} val windows",flush=True)
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
        zq=[Zv[t] for t in range(len(m.specs))]; zq[TIER]=rec.reshape(NW,-1,D)
        out=m.decoder(zq,torch.zeros(NW,len(m.specs),device=DEV))
        xh=X.to(DEV); v=xh.abs().sum(1)>0
        return float(m.source.recon_stats(out.float(),xh,v)[1]["cos"])
def bits1(col_fit,col_ho):
    c=torch.quantile(col_fit,torch.tensor([0.25,0.75],device=DEV))
    for _ in range(25):
        a=(col_fit[:,None]-c[None,:]).abs().argmin(1)
        s_=torch.zeros(2,device=DEV).index_add_(0,a,col_fit)
        n_=torch.zeros(2,device=DEV).index_add_(0,a,torch.ones_like(col_fit))
        lv=n_>0; c[lv]=s_[lv]/n_[lv]
    return c[(col_ho[:,None]-c[None,:]).abs().argmin(1)]
res={}
for N in (0,2,4,6,8):
    Rf=FIT.clone(); Rh=HO.clone(); rec=torch.zeros_like(HO)
    bits=[]; dc=[]; rr=[]
    books=[]
    for l in range(N):
        C=km(Rf,K,sd=l); books.append(C)
        Rf=Rf-C[_chunked_assign(Rf,C)]
        a=_chunked_assign(Rh,C); rec=rec+C[a]; Rh=Rh-C[a]
        bits.append((l+1)*STEP); rr.append(float(Rh.pow(2).sum(-1).mean())/e0)
        dc.append(1-decode(rec))
    P=(NP-N)*STEP
    if P>0:
        mu=Rf.mean(0); Fc=Rf-mu
        _,U=torch.linalg.eigh(((Fc.T@Fc)/Fc.shape[0]).double()); Up=U.flip(1).float()[:,:P]
        Ptr=(Rf-mu)@Up; Pho=(Rh-mu)@Up; acc=torch.zeros_like(Pho)
        for j in range(P):
            acc[:,j]=bits1(Ptr[:,j],Pho[:,j])
            if (j+1)%STEP==0:
                part=acc[:,:j+1]@Up[:,:j+1].t()+mu
                bits.append(N*STEP+j+1)
                rr.append(float((Rh-part).pow(2).sum(-1).mean())/e0)
                dc.append(1-decode(rec+part))
    res[N]=(bits,dc,rr); print(f"  N={N} done, final dcos {dc[-1]:.5f}",flush=True)
fig,ax=plt.subplots(1,2,figsize=(13,5.2),constrained_layout=True)
cols=plt.cm.viridis(np.linspace(0,.85,len(res)))
for (N,(b,d,r)),c in zip(sorted(res.items()),cols):
    lab=f"RVQ x{N}" + (f" + {(NP-N)*STEP} dirs" if N<NP else " (pure)")
    for a_,y in ((ax[0],d),(ax[1],r)):
        a_.plot(b,y,"o-",color=c,lw=2,ms=5,label=lab)
ax[0].set(yscale="log",xlabel="bits / cell",ylabel="1 - decoded cos",
          title="Decoded error (log)")
ax[1].set(yscale="log",xlabel="bits / cell",ylabel="latent residual E/E0",
          title="Latent residual (log)")
for a_ in ax: a_.legend(frameon=False,fontsize=8); a_.grid(alpha=.25)
fig.savefig("/data/tessera-samples/reports/branch_ladders.png",dpi=140)
print("saved -> /data/tessera-samples/reports/branch_ladders.png",flush=True)
