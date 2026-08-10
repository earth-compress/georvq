"""Bake-off on REAL latents: unconstrained RVQ vs gain-shape RVQ.

Fits both quantizers on the frozen uf encoder's fine-tier latents (the
tier that collapsed to 3.9% perplexity) with an identical budget, then
reports utilization, residual energy, and TRUE rate (gain indices are
entropy-coded, so the gain's cost is its empirical entropy, not 5 bits).
"""
import math
import numpy as np
import torch

from georvq.factory import from_checkpoint
from georvq.data.store import ChipStore
from georvq.data.pool import fixed_val_tokens
from georvq.model.vq import ResidualVQ, kmeans_init_rvq
from georvq.model.vq_gainshape import GainShapeResidualVQ

DEV = torch.device("cuda:1")
TIER = 2                      # fine tier, dim 256, K 16384
LEVELS, K = 8, 16384
STEPS, BATCH = 300, 4096

m = from_checkpoint(
    "/home/ali/projects/ring-space/checkpoints/tessera_uf_vq8_s2/model.pt", DEV)
cfg = m.full_config
store = ChipStore(cfg.data.store_dir)
_, vr = store.split(cfg.data.val_frac, cfg.data.split_seed)
val_s = fixed_val_tokens(store, vr, token=cfg.model.window, max_tokens=1536)

Z = []
with torch.no_grad():
    for i in range(0, val_s.shape[0], 64):
        x, _ = m.source.prep(val_s[i:i + 64], DEV)
        z = m.encode_levels(x)[TIER]
        Z.append(z.float())                      # keep (win, cells, D)
Z = torch.cat(Z)
# split by WINDOW: adjacent cells inside a window are near-duplicates,
# so a random per-cell split leaks the eval set into the fit set
nw = Z.shape[0]
wperm = torch.randperm(nw, device=DEV)
n_fit_w = int(nw * 0.75)
D = Z.shape[-1]
fit = Z[wperm[:n_fit_w]].reshape(-1, D)
ev = Z[wperm[n_fit_w:]].reshape(-1, D)
Z = Z.reshape(-1, D)
print(f"tier {TIER}: dim {D}, fit {fit.shape[0]}, eval {ev.shape[0]}, "
      f"K={K}, levels={LEVELS}")
print(f"latent norm: mean {Z.norm(dim=1).mean():.3f} "
      f"CV {float(Z.norm(dim=1).std()/Z.norm(dim=1).mean()):.3f}\n")


def train(rvq, name):
    rvq.to(DEV).train()
    kmeans_init_rvq(rvq, fit[:150_000], iters=6, expected_batch=BATCH)
    if hasattr(rvq, "spherize_books"):
        rvq.spherize_books(fit[:150_000])
    g = torch.Generator(device=DEV).manual_seed(0)
    for s in range(STEPS):
        idx = torch.randint(0, fit.shape[0], (BATCH,), device=DEV, generator=g)
        rvq(fit[idx])
    return rvq


@torch.no_grad()
def evaluate(rvq, name):
    rvq.eval()
    resid = ev.clone()
    e0 = float(resid.pow(2).sum(-1).mean())
    rows, gain_bits_tot = [], 0.0
    for li, lev in enumerate(rvq.levels):
        E = lev.codebook.weight.float()
        if hasattr(lev, "quantize_gain"):
            idx = (resid @ E.t()).argmax(-1)
            u = E[idx]
            g_hat, g_idx = lev.quantize_gain((resid * u).sum(-1))
            q = g_hat.unsqueeze(-1) * u
            gc = torch.bincount(g_idx, minlength=lev.gain_book.numel()).float()
            gp = gc / gc.sum()
            gain_bits_tot += float(-(gp * gp.clamp_min(1e-12).log2()).sum())
        else:
            d = (resid.pow(2).sum(1, keepdim=True) - 2 * resid @ E.t()
                 + E.pow(2).sum(1)[None])
            idx = d.argmin(-1)
            q = E[idx]
        cnt = torch.bincount(idx, minlength=K).float()
        p = cnt / cnt.sum()
        perp = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
        idx_ent = float(-(p * p.clamp_min(1e-12).log2()).sum())
        resid = resid - q
        rows.append((li, perp, perp / K * 100, idx_ent,
                     float(resid.pow(2).sum(-1).mean()) / e0))
    print(f"--- {name}")
    print(f"{'lvl':>3} {'perplex':>8} {'%K':>6} {'idx bits':>9} "
          f"{'resid E/E0':>11}")
    for r in rows:
        print(f"{r[0]:>3} {r[1]:>8.0f} {r[2]:>6.1f} {r[3]:>9.2f} {r[4]:>11.4f}")
    idx_bits = sum(r[3] for r in rows)
    tot = idx_bits + gain_bits_tot
    print(f"  final residual energy: {rows[-1][4]:.4f} of input")
    print(f"  rate: {idx_bits:.1f} index bits"
          + (f" + {gain_bits_tot:.1f} gain bits" if gain_bits_tot else "")
          + f" = {tot:.1f} bits/cell over {LEVELS} levels\n")
    return rows[-1][4], tot


base = train(ResidualVQ(num_levels=LEVELS, num_codes=K, dim=D, decay=0.99,
                        dead_threshold=0.000625, bf16_scores=True), "base")
b_res, b_bits = evaluate(base, "unconstrained RVQ (production path)")
del base; torch.cuda.empty_cache()

for gb in (5, 3):
    gs = train(GainShapeResidualVQ(num_levels=LEVELS, num_codes=K, dim=D,
                                   decay=0.99, dead_threshold=0.000625,
                                   bf16_scores=True, gain_bits=gb),
               f"gs{gb}")
    g_res, g_bits = evaluate(gs, f"gain-shape RVQ (gain_bits={gb})")
    print(f"  vs base: residual x{g_res/b_res:.3f} at "
          f"{g_bits - b_bits:+.1f} bits/cell\n")
    del gs; torch.cuda.empty_cache()
