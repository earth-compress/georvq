"""Is a tier's latent carrying anything, or has it collapsed to a constant?

Written for a specific symptom: on the stats-conditioning run, min(4-0-0)
and coarse_only(20-0-0) validated IDENTICAL to four decimals across
successive validations, despite differing by 16 of tier 0's 20 RVQ levels.
Exact equality is stronger than "the deep levels are weak" -- it says they
contribute nothing at all.

The suspected cause is the VQ-free warmup. The warmup branch decodes at
FULL depth every step, so a long warmup gives the model no reason to make
the coarse tier informative: the decoder always has the fine tier
available. A representation that ignores tier 0 is then a perfectly good
solution to the warmup objective, and the matryoshka ordering the pyramid
depends on never forms.

Distinguishes the possibilities by measuring, per tier:
  * ACROSS-WINDOW spread of the latent -- a collapsed tier has nearly the
    same latent for every window, so its across-window std is tiny
    relative to the latent's own magnitude;
  * effective rank of the latent across windows;
  * per-LEVEL RVQ contribution norms -- if levels 5..20 contribute ~0,
    the deep ladder is inert;
  * per-level codebook perplexity.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import ChipStore
from georvq.factory import from_checkpoint


def eff_rank(Z):
    """Participation ratio of the across-window covariance."""
    Zc = Z - Z.mean(0)
    lam = torch.linalg.eigvalsh(torch.cov(Zc.T.double())).clamp_min(0)
    return float(lam.sum() ** 2 / lam.pow(2).sum().clamp_min(1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    store = ChipStore(cfg.data.store_dir)
    _, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val = fixed_val_tokens(store, va, token=cfg.model.window,
                           max_tokens=args.tokens)
    print(f"{args.ckpt}\n  {val.shape[0]} val windows, tiers "
          f"{[str(s) for s in m.specs]}\n")

    Z = [[] for _ in m.specs]
    cum_keep = None
    for i in range(0, val.shape[0], args.chunk):
        x, v = m.source.prep(val[i:i + args.chunk], dev)
        x = x.float()
        st = m._stats_of(x, v)
        zs = m.encode_levels(x, st, v)
        for t, z in enumerate(zs):
            Z[t].append(z.float().cpu())
        if cum_keep is None:
            _, cum_keep = m.level_cumsums(x, valid=v)
            cum_keep = [c.float().cpu() for c in cum_keep]
        del x, zs
    Z = [torch.cat(z) for z in Z]

    print("LATENT SPREAD ACROSS WINDOWS (a collapsed tier varies little)")
    print(f"{'tier':>5} {'cells':>6} {'dim':>5} {'mean |z|':>10} "
          f"{'across-window std':>18} {'ratio':>8} {'eff rank':>9}")
    for t, (z, sp) in enumerate(zip(Z, m.specs)):
        flat = z.reshape(-1, z.shape[-1])
        # per-window latent, averaged over cells: the quantity tier 0 has
        # exactly one of
        per_win = z.mean(1)
        mag = float(flat.norm(dim=-1).mean())
        sd = float((per_win - per_win.mean(0)).norm(dim=-1).mean())
        print(f"{t:>5} {sp.cells:>6} {sp.dim:>5} {mag:>10.4f} "
              f"{sd:>18.4f} {sd/max(mag,1e-9):>8.4f} "
              f"{eff_rank(per_win.double()):>9.2f}")

    print("\nPER-LEVEL RVQ CONTRIBUTION (norm of what each level adds)")
    for t, (c, sp) in enumerate(zip(cum_keep, m.specs)):
        # c: (B, cells, depth+1, dim) cumulative sums
        contrib = (c[:, :, 1:] - c[:, :, :-1]).norm(dim=-1).mean((0, 1))
        tot = float(c[:, :, -1].norm(dim=-1).mean())
        print(f"  tier {t} (|zq| at full depth = {tot:.4f}):")
        print("    " + " ".join(f"L{j+1}:{float(v):.4f}"
                                for j, v in enumerate(contrib)))

    print("\nPER-LEVEL CODEBOOK PERPLEXITY (of K = "
          f"{[sp.codes for sp in m.specs]})")
    for t, (z, rvq, sp) in enumerate(zip(Z, m.rvqs, m.specs)):
        zz = z.to(dev).float()
        per, idx = rvq.all_levels(zz)[:2] if isinstance(
            rvq.all_levels(zz), tuple) else (None, None)
        res = zz.reshape(-1, sp.dim).clone()
        pp = []
        for lev in rvq.levels:
            C = lev.codebook.weight.data.float()
            a = torch.cdist(res, C).argmin(1)
            cnt = torch.bincount(a, minlength=sp.codes).float()
            p = cnt / cnt.sum()
            pp.append(float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())))
            res = res - C[a]
        print(f"  tier {t}: " + " ".join(f"{v:.0f}" for v in pp))


if __name__ == "__main__":
    main()
