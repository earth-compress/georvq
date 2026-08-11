import numpy as np, torch, glob, json
from georvq.factory import from_checkpoint
DEV=torch.device("cuda:0"); TIER=2; EPS=0.25
man=json.load(open("/data/tessera-samples/latents_w32hd_full/manifest.json"))
SC=man["scale"]
fs=sorted(glob.glob(f"/data/tessera-samples/latents_w32hd_full/t{TIER}_*.npy"))[:3]
A=np.concatenate([np.load(f,mmap_mode="r")[::37].reshape(-1,man["dim"]) for f in fs])
print(f"{len(fs)} shards -> {A.shape[0]/1e6:.2f}M vectors, dim {A.shape[1]}",flush=True)
Z=torch.from_numpy(np.ascontiguousarray(A)).to(DEV).float()*SC
Zc=Z-Z.mean(0); C=(Zc.T@Zc)/Zc.shape[0]
sd=C.diag().sqrt(); R=C/(sd[:,None]*sd[None,:])
off=R[~torch.eye(R.shape[0],dtype=torch.bool,device=DEV)]
ev,U=torch.linalg.eigh(C.double()); ev=ev.flip(0).float().clamp_min(0); U=U.flip(1).float()
cum=(ev.cumsum(0)/ev.sum()).cpu().numpy()
print(f"CORRELATION: mean|r| {float(off.abs().mean()):.4f}  p99|r| "
      f"{float(off.abs().quantile(0.99)):.3f}  max|r| {float(off.abs().max()):.3f}")
print(f"EIGENSPECTRUM: PR {float(ev.sum()**2/(ev**2).sum()):.1f}  d50 "
      f"{int((cum<.5).sum())+1}  d95 {int((cum<.95).sum())+1}  "
      f"top1 {float(ev[0]/ev.sum())*100:.1f}%  ev[0]/ev[-1] {float(ev[0]/ev[-1].clamp_min(1e-9)):.0f}x")
# decoder sensitivity ALONG PRINCIPAL DIRECTIONS
m=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt",DEV)
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
cfg=m.full_config; st=ChipStore(cfg.data.store_dir)
_,vr=st.split(cfg.data.val_frac,cfg.data.split_seed)
vs=fixed_val_tokens(st,vr,token=cfg.model.window,max_tokens=64)
with torch.no_grad():
    x,_=m.source.prep(vs[:64],DEV); zs=[z.float() for z in m.encode_stage1(x)]
    z0=torch.zeros(64,len(m.specs),device=DEV); base=m.decoder(zs,z0).float()
    s=torch.zeros(U.shape[1],device=DEV)
    for k in range(U.shape[1]):
        zp=[z.clone() for z in zs]; zp[TIER]=zp[TIER]+EPS*U[:,k]
        s[k]=float((m.decoder(zp,z0).float()-base).pow(2).mean())
sn=s.cpu().numpy(); evn=ev.cpu().numpy()
print(f"PC SENSITIVITY: CV {sn.std()/sn.mean():.3f}  top/bottom decile "
      f"{sn[:38].mean()/sn[-38:].mean():.2f}x  corr(sens, log eigenvalue) "
      f"{np.corrcoef(sn,np.log(evn+1e-12))[0,1]:+.3f}")
p=evn*sn; p/=p.sum()
print(f"IMPORTANCE (var x sens) in PC basis: top-10% of PCs hold {p[:38].sum()*100:.1f}%"
      f"  top-25% hold {p[:96].sum()*100:.1f}%   (uniform = 10 / 25%)")
