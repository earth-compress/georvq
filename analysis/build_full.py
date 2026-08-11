"""Full-corpus int8 latent store from the finished w32hd encoder.
encode_stage1 latents are unit-norm*sqrt(D) => per-element std ~1, so a
single global scale suffices (no per-vector sidecar)."""
import numpy as np, torch, time, os, json
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
DEV=torch.device("cuda:1"); W=32; SCALE=6.0/127.0; PER_SHARD=128
OUT="/data/tessera-samples/latents_w32hd_full"; os.makedirs(OUT,exist_ok=True)
m=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt",DEV)
cfg=m.full_config; store=ChipStore(cfg.data.store_dir)
tr,va=store.split(cfg.data.val_frac,cfg.data.split_seed)
idx=store.index.reset_index(drop=True)
val_ids=set(va.chip_id.tolist())
CELLS=[s.cells for s in m.specs]; DIM=m.specs[0].dim
json.dump({"scale":SCALE,"window":W,"cells":CELLS,"dim":DIM,
           "tiers":[str(s) for s in m.specs],"per_shard":PER_SHARD,
           "ckpt":"tessera_s1_w32hd"},open(f"{OUT}/manifest.json","w"))
print(f"{len(idx)} chips, {256} win/chip, cells {CELLS}, dim {DIM}",flush=True)
t0=time.time(); buf=[[] for _ in CELLS]; meta=[]; sh=0
def flush(sh,buf,meta):
    for ti in range(len(CELLS)):
        np.save(f"{OUT}/t{ti}_{sh:04d}.npy",np.concatenate(buf[ti]))
    np.save(f"{OUT}/meta_{sh:04d}.npy",np.array(meta,dtype=np.int64))
for ci,r in idx.iterrows():
    chip=np.asarray(store.open_shard(int(r["shard"]))[int(r["slot"])])
    sc=np.load(f"{store.dir}/scales_{int(r['shard']):04d}.npy",mmap_mode="r")[int(r["slot"])]
    sb=np.ascontiguousarray(sc).astype(np.float16).view(np.int8).reshape(512,512,2).transpose(2,0,1)
    t=torch.from_numpy(np.concatenate([chip,sb])).to(DEV)
    tiles=t.unfold(1,W,W).unfold(2,W,W).permute(1,2,0,3,4).reshape(-1,130,W,W)
    with torch.no_grad():
        x,v=m.source.prep(tiles,DEV)
        ok=(v.float().mean((1,2))>0.98)
        zs=m.encode_stage1(x)
    for ti,z in enumerate(zs):
        q=(z/SCALE).round().clamp(-127,127).to(torch.int8).cpu().numpy()
        buf[ti].append(q)
    meta.append([int(r["chip_id"]), int(r["chip_id"] in val_ids)] +
                ok.cpu().numpy().astype(np.int64).tolist())
    if len(meta)==PER_SHARD:
        flush(sh,buf,meta); sh+=1; buf=[[] for _ in CELLS]; meta=[]
        if sh%10==0:
            gb=sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT))/2**30
            done=(ci+1)/len(idx)
            print(f"  shard {sh}, {ci+1}/{len(idx)} chips, {gb:.0f} GiB, "
                  f"{(time.time()-t0)/60:.0f}m elapsed, "
                  f"~{(time.time()-t0)/max(done,1e-9)/60*(1-done):.0f}m left",flush=True)
if meta: flush(sh,buf,meta)
print(f"DONE {len(idx)} chips in {(time.time()-t0)/3600:.2f}h",flush=True)
