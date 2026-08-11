"""Large offline latent store for one-shot codebook fitting.
Per-tier sampling so each tier gets ~2M vectors (coarse needs 1 window
each; fine gets 256 per window), from TRAIN chips only."""
import numpy as np, torch, time
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
DEV=torch.device("cuda:1"); W=32
OUT="/data/tessera-samples/latents_w32hd"
m=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt",DEV)
cfg=m.full_config; store=ChipStore(cfg.data.store_dir)
tr,_=store.split(cfg.data.val_frac,cfg.data.split_seed)
rows=tr.sample(9000,random_state=0)          # ~2.3M windows of 32px
KEEP=[1,16,256]                              # keep 1/KEEP[t] of windows per tier
buf=[[] for _ in m.specs]; nwin=0; t0=time.time()
import os; os.makedirs(OUT,exist_ok=True)
for ci,(_,r) in enumerate(rows.iterrows()):
    sh=store.open_shard(int(r["shard"])); chip=np.asarray(sh[int(r["slot"])])
    sc=np.load(f"{store.dir}/scales_{int(r['shard']):04d}.npy",mmap_mode="r")[int(r["slot"])]
    sb=np.ascontiguousarray(sc).astype(np.float16).view(np.int8).reshape(512,512,2).transpose(2,0,1)
    full=np.concatenate([chip,sb])
    t=torch.from_numpy(full).to(DEV)
    tiles=t.unfold(1,W,W).unfold(2,W,W).permute(1,2,0,3,4).reshape(-1,130,W,W)
    with torch.no_grad():
        x,v=m.source.prep(tiles,DEV)
        ok=v.float().mean((1,2))>0.98
        if ok.sum()==0: continue
        x=x[ok]
        for ti,z in enumerate(m.encode_stage1(x)):
            step=KEEP[ti]
            buf[ti].append(z[::step].reshape(-1,z.shape[-1]).half().cpu())
    nwin+=int(ok.sum())
    if (ci+1)%1500==0:
        print(f"  {ci+1}/9000 chips, {nwin/1e6:.2f}M windows, "
              f"{[sum(b.shape[0] for b in bb)/1e6 for bb in buf]} M vec/tier "
              f"({time.time()-t0:.0f}s)",flush=True)
for ti in range(len(buf)):
    A=torch.cat(buf[ti]); np.save(f"{OUT}/tier{ti}.npy",A.numpy()); buf[ti]=None
    print(f"tier {ti}: {A.shape} -> {OUT}/tier{ti}.npy",flush=True)
print(f"done: {nwin} windows in {time.time()-t0:.0f}s",flush=True)
