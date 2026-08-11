"""Freeze AE encoder+decoder; optimise ONLY the codebooks against the
DECODED loss (not latent MSE). Codeword values are differentiable via
z_q = E[idx]; assignment stays argmin (non-diff, re-evaluated each step)."""
import torch, pandas as pd, time
from georvq.compat import from_legacy_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
DEV = torch.device("cuda:1")
m, cfg = from_legacy_checkpoint(
    "/home/ali/projects/ring-space/checkpoints/ae_pyr64_divfull/model_best.pt", DEV)
store = ChipStore(cfg.data.store_dir)
vr = pd.read_parquet("/home/ali/projects/ring-space/checkpoints/ae_pyr64_divfull/val_chips.parquet")
vs = fixed_val_tokens(store, vr, token=cfg.model.window, max_tokens=2048).to(DEV)
n = vs.shape[0]; nf = int(n*0.7)
FITW, HOW = vs[:nf], vs[nf:]
for p in m.parameters(): p.requires_grad_(False)
BOOKS = [l.codebook.weight for rvq in m.rvqs for l in rvq.levels]
for b in BOOKS: b.requires_grad_(True)
Ls = lambda B: torch.tensor([[s.depth for s in m.specs]]*B, device=DEV)

def run(xb, grad):
    with torch.set_grad_enabled(grad):
        with torch.no_grad():
            x, _ = m.source.prep(xb, DEV); zs = m.encode_levels(x)
        zq = []
        for t, (z, rvq) in enumerate(zip(zs, m.rvqs)):
            R = z.reshape(-1, z.shape[-1]).float(); acc = 0
            for lev in rvq.levels:
                E = lev.codebook.weight.float()
                with torch.no_grad():
                    d = (R.pow(2).sum(1,keepdim=True) - 2*R@E.t() + E.pow(2).sum(1)[None])
                    a = d.argmin(1)
                q = E[a]; acc = acc + q; R = R - q.detach()
            zq.append(acc.reshape(z.shape))
        out = m.decoder(zq, Ls(xb.shape[0]))
        v = x.abs().sum(1) > 0
        mse, st = m.source.recon_stats(out.float(), x, v)
    return mse, float(st["cos"])

@torch.no_grad()
def ho():
    cs, nb = 0.0, 0
    for i in range(0, HOW.shape[0], 32):
        _, c = run(HOW[i:i+32], False); cs += c*32; nb += 32
    return cs/nb
print(f"AE codebook-only optimisation ({nf} fit / {n-nf} holdout windows)", flush=True)
b0 = ho(); print(f"  baseline holdout cos {b0:.5f}", flush=True)
opt = torch.optim.Adam(BOOKS, lr=1e-4)
t0 = time.time(); g = torch.Generator(device=DEV).manual_seed(0)
for s in range(1, 401):
    idx = torch.randint(0, nf, (24,), device=DEV, generator=g)
    mse, _ = run(FITW[idx], True)
    opt.zero_grad(set_to_none=True); mse.backward(); opt.step()
    if s % 100 == 0:
        print(f"  step {s}: holdout cos {ho():.5f}  ({time.time()-t0:.0f}s)", flush=True)
print(f"FINAL: {b0:.5f} -> {ho():.5f}", flush=True)
