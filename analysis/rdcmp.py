"""R-D: trained phase-1 stage-2 model vs new w32hd encoder with fitted
int4 books and an UNTUNED stage-1 decoder."""
import numpy as np, torch, glob, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import _chunked_assign
DEV=torch.device("cuda:0")
def val_windows(m,n=2048):
    cfg=m.full_config; st=ChipStore(cfg.data.store_dir)
    _,vr=st.split(cfg.data.val_frac,cfg.data.split_seed)
    return fixed_val_tokens(st,vr,token=cfg.model.window,max_tokens=n)
def qb(C,bits=4):
    lv=2**bits; s=C.abs().amax(1,keepdim=True).clamp_min(1e-9); Cn=(C/s).reshape(-1)
    sub=Cn[torch.randperm(Cn.numel(),device=DEV)[:200000]]
    g=torch.quantile(sub,(torch.arange(lv,device=DEV).float()+0.5)/lv)
    for _ in range(30):
        a=(sub[:,None]-g[None,:]).abs().argmin(1)
        sm=torch.zeros(lv,device=DEV).index_add_(0,a,sub)
        n=torch.zeros(lv,device=DEV).index_add_(0,a,torch.ones_like(sub))
        l=n>0; g[l]=sm[l]/n[l]
    o=torch.empty_like(C)
    for i in range(0,C.shape[0],4096):
        b=C[i:i+4096]/s[i:i+4096]; o[i:i+4096]=g[(b[...,None]-g).abs().argmin(-1)]*s[i:i+4096]
    return o
def km(R,K,it=8,sd=0):
    N=R.shape[0];g=torch.Generator(device=DEV).manual_seed(sd)
    C=R[torch.randperm(N,device=DEV,generator=g)[:K]].clone();o=torch.ones(N,device=DEV)
    for _ in range(it):
        a=_chunked_assign(R,C);s=torch.zeros_like(C).index_add_(0,a,R)
        c=torch.zeros(K,device=DEV).index_add_(0,a,o);l=c>0
        C[l]=s[l]/c[l,None];nd=int((~l).sum())
        if nd: C[~l]=R[torch.randint(0,N,(nd,),device=DEV)]
    return C
def bits_px(specs,tup,px):
    return sum(specs[t].cells*tup[t]*np.log2(specs[t].codes) for t in range(len(tup)))/px
# ---- A: trained phase-1 model
mA=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_vq8/model.pt",DEV)
vsA=val_windows(mA); pxA=mA.full_config.model.window**2
XA=[];ZA=[[] for _ in mA.specs]
with torch.no_grad():
    for i in range(0,vsA.shape[0],32):
        x,_=mA.source.prep(vsA[i:i+32],DEV); XA.append(x.cpu())
        for t,z in enumerate(mA.encode_levels(x)): ZA[t].append(z.float())
XA=torch.cat(XA); ZA=[torch.cat(z) for z in ZA]; NA=XA.shape[0]
perA=[]
with torch.no_grad():
    for t,rvq in enumerate(mA.rvqs):
        sp=mA.specs[t]; R=ZA[t].reshape(-1,sp.dim).clone()
        lv=[]
        for lev in rvq.levels:
            E=lev.codebook.weight.float()
            a=_chunked_assign(R,E,chunk=8192); q=E[a]; lv.append(q); R=R-q
        perA.append(torch.stack(lv,-2).reshape(*ZA[t].shape[:2],len(lv),sp.dim))
        print(f"A tier{t} quantised",flush=True)
TA=[(8,0,0),(8,1,0),(8,2,0),(8,4,0),(8,8,0),(8,8,1),(8,8,2),(8,8,4),(8,8,8)]
rA=[]
with torch.no_grad():
    for tup in TA:
        zq=[perA[t][...,:tup[t],:].sum(-2) if tup[t]>0 else torch.zeros_like(ZA[t])
            for t in range(len(tup))]
        Ls=torch.tensor([list(tup)]*NA,device=DEV)
        out=mA.decoder(zq,Ls); xh=XA.to(DEV); v=xh.abs().sum(1)>0
        c=float(mA.source.recon_stats(out.float(),xh,v)[1]["cos"])
        rA.append((bits_px(mA.specs,tup,pxA),c)); print(f"A {tup} {c:.4f}",flush=True)
del perA,ZA; torch.cuda.empty_cache()
# ---- B: new encoder, fitted int4 books, untuned decoder
mB=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt",DEV)
man=json.load(open("/data/tessera-samples/latents_w32hd_full/manifest.json")); SC=man["scale"]
vsB=val_windows(mB,4096); pxB=mB.full_config.model.window**2
XB=[];ZB=[[] for _ in mB.specs]
with torch.no_grad():
    for i in range(0,vsB.shape[0],64):
        x,_=mB.source.prep(vsB[i:i+64],DEV); XB.append(x.cpu())
        for t,z in enumerate(mB.encode_stage1(x)): ZB[t].append(z.float())
XB=torch.cat(XB); ZB=[torch.cat(z) for z in ZB]; NB=XB.shape[0]
KS=[s.codes for s in mB.specs]; BK=[]
for t,sp in enumerate(mB.specs):
    fs=sorted(glob.glob(f"/data/tessera-samples/latents_w32hd_full/t{t}_*.npy"))[:3]
    A_=np.concatenate([np.load(f,mmap_mode="r")[::7].reshape(-1,sp.dim) for f in fs])
    R=torch.from_numpy(np.ascontiguousarray(A_)).to(DEV).float()*SC
    if R.shape[0]>600_000: R=R[torch.randperm(R.shape[0],device=DEV)[:600_000]]
    bs=[]
    for l in range(8):
        C=qb(km(R,KS[t],sd=l)); bs.append(C); R=R-C[_chunked_assign(R,C)]
    BK.append(bs); print(f"B tier{t} fitted",flush=True)
TB=[(8,0,0),(8,1,0),(8,4,0),(8,8,0),(8,8,1),(8,8,2),(8,8,4),(8,8,8)]
rB=[]
with torch.no_grad():
    for tup in TB:
        zq=[];sg=[]
        for t,sp in enumerate(mB.specs):
            H=ZB[t].reshape(-1,sp.dim); R=H.clone(); q=torch.zeros_like(H)
            for l in range(tup[t]):
                a=_chunked_assign(R,BK[t][l]); q=q+BK[t][l][a]; R=R-BK[t][l][a]
            rel=float(R.pow(2).sum(-1).mean())/float(H.pow(2).sum(-1).mean())
            zq.append(q.reshape(NB,sp.cells,sp.dim)); sg.append(min((12*rel)**0.5,4.0))
        out=mB.decoder(zq,torch.tensor([sg]*NB,device=DEV,dtype=torch.float))
        xh=XB.to(DEV); v=xh.abs().sum(1)>0
        c=float(mB.source.recon_stats(out.float(),xh,v)[1]["cos"])
        rB.append((bits_px(mB.specs,tup,pxB),c)); print(f"B {tup} {c:.4f}",flush=True)
fig,ax=plt.subplots(figsize=(8,5.6),constrained_layout=True)
for (r,lab,c,ceil) in ((rA,"phase-1: tessera_vq8, end-to-end trained","#4269d0",0.9660),
                       (rB,"new: w32hd + int4 books, UNTUNED decoder","#e05c5c",0.9899)):
    b=[x[0] for x in r]; y=[1-x[1] for x in r]
    ax.plot(b,y,"o-",color=c,lw=2,ms=5,label=lab)
    ax.axhline(1-ceil,color=c,ls=":",lw=1)
ax.set(xscale="log",yscale="log",xlabel="bits / pixel",ylabel="1 - decoded cos",
       title="R-D: trained phase-1 codec vs new encoder before stage-2\n(dotted = each model's unquantized ceiling)")
ax.legend(frameon=False,fontsize=9); ax.grid(alpha=.25)
fig.savefig("/data/tessera-samples/reports/rd_compare.png",dpi=140)
print("saved -> /data/tessera-samples/reports/rd_compare.png",flush=True)
