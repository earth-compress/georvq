"""Per-dimension decoder sensitivity vs latent variance (w32hd)."""
import numpy as np, torch
from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
DEV=torch.device("cuda:0"); TIER=2; EPS=0.25
m=from_checkpoint("/home/ali/projects/ring-space/checkpoints/tessera_s1_w32hd/model.pt",DEV)
cfg=m.full_config; store=ChipStore(cfg.data.store_dir)
_,vr=store.split(cfg.data.val_frac,cfg.data.split_seed)
vs=fixed_val_tokens(store,vr,token=cfg.model.window,max_tokens=64)
with torch.no_grad():
    x,_=m.source.prep(vs[:64],DEV); zs=[z.float() for z in m.encode_stage1(x)]
    S=torch.zeros(len(m.specs),1,device=DEV)
    base=m.decoder(zs,torch.zeros(64,len(m.specs),device=DEV)).float()
    D=zs[TIER].shape[-1]
    sens=torch.zeros(D,device=DEV)
    for d in range(D):
        zp=[z.clone() for z in zs]; zp[TIER][...,d]+=EPS
        out=m.decoder(zp,torch.zeros(64,len(m.specs),device=DEV)).float()
        sens[d]=float((out-base).pow(2).mean())
    var=zs[TIER].reshape(-1,D).var(0)
s=sens.cpu().numpy(); v=var.cpu().numpy()
o=np.argsort(-s)
print(f"tier {TIER} decoder sensitivity (D={D}, eps={EPS})")
print(f"  sensitivity: max/median {s.max()/np.median(s):.2f}x  "
      f"p99/p1 {np.percentile(s,99)/np.percentile(s,1):.2f}x  "
      f"CV {s.std()/s.mean():.3f}")
print(f"  latent var : max/median {v.max()/np.median(v):.2f}x  CV {v.std()/v.mean():.3f}")
print(f"  corr(sensitivity, variance) = {np.corrcoef(s,v)[0,1]:+.3f}")
print(f"  top-10% sensitive dims hold {s[o[:D//10]].sum()/s.sum()*100:.1f}% of total sensitivity"
      f"  (uniform would be 10%)")
