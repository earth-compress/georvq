"""What does the pyramid's coarse tier actually do with the patch moments?

The moment experiments say a dedicated coder conveys a 64px window's mean
to cos ~0.985 for ~128 bits. Tier 0 of the pyramid spends 160 bits and its
grid is 1x1, so its cell IS the window -- it is nominally solving the same
problem. `cos_blockmean` (0.9223 for k256) already beats coarse_only
(0.9049), which hints the tier is losing to a plain mean, but that is a
PER-PIXEL cosine and does not isolate the moments.

This measures the moments directly: take the reconstruction at a given
depth tuple, compute ITS per-window mean and std, and score them against
the true ones -- the same quantities, the same metrics, as moment_rd, so
the two are directly comparable.

If tier 0 recovers the mean worse than a 128-bit moment coder does, the
coarse tier is spending its budget on something a cheaper, dedicated
component does better, and moving that job out is a strict win.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import from_checkpoint


@torch.no_grad()
def moments(x, valid):
    """(B,C,H,W) -> per-window mean and std over VALID pixels, (B,C) each."""
    vm = valid.unsqueeze(1).float()
    n = vm.sum((2, 3)).clamp_min(1)
    mu = (x * vm).sum((2, 3)) / n
    var = ((x - mu[:, :, None, None]) ** 2 * vm).sum((2, 3)) / n.clamp_min(1)
    return mu, var.clamp_min(0).sqrt()


def score(name, a, b, bits):
    """a = reconstructed quantity, b = truth."""
    rel = float(((a - b).pow(2).sum(1).sqrt()
                 / b.pow(2).sum(1).sqrt().clamp_min(1e-9)).mean())
    cos = float(F.cosine_similarity(a, b, dim=1, eps=1e-9).mean())
    print(f"{name:>14} {bits:>7} {rel:>10.4f} {cos:>10.5f}")
    return rel, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="checkpoints/tessera_vq8_k256/model_best.pt")
    ap.add_argument("--device", default="cuda",
                    help="torch device; cuda is the card Slurm allocated")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--tuples", default="4-0-0,10-0-0,20-0-0,20-20-0,"
                                        "20-20-20")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    store = open_store(cfg.data.store_dir)
    _, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val = fixed_val_tokens(store, va, token=cfg.model.window,
                           max_tokens=args.tokens)
    print(f"{args.ckpt}\n  {val.shape[0]} held-out {cfg.model.window}px "
          f"windows; tier grids {[sp.grid for sp in m.specs]}")

    import math
    tuples = [tuple(int(v) for v in t.split("-"))
              for t in args.tuples.split(",")]
    print(f"\n{'tuple':>14} {'bits':>7} {'rel':>10} {'cos':>10}"
          f"   (quantity: per-window MEAN)")
    store_mu, store_sd, store_x = [], [], []
    for tup in tuples:
        MU, SD, MUh, SDh = [], [], [], []
        for i in range(0, val.shape[0], 8):
            x, v = m.prep(val[i:i + 8], {}, dev)[:2]
            zs = m.encode_levels(x, m._stats_of(x, v), v)
            Ls = torch.tensor(tup, device=dev)[None].expand(x.shape[0], -1)
            zq, _, _ = m.quantize(zs, Ls)
            out = m.decode(zq, Ls, m._stats_of(x, v), v).float()
            a, b = moments(x.float(), v)
            c, d = moments(out, v)
            MU.append(a.cpu()); SD.append(b.cpu())
            MUh.append(c.cpu()); SDh.append(d.cpu())
            del x, zs, zq, out
        MU, SD = torch.cat(MU), torch.cat(SD)
        MUh, SDh = torch.cat(MUh), torch.cat(SDh)
        bits = sum(sp.cells * L * math.log2(sp.codes)
                   for sp, L in zip(m.specs, tup))
        score("-".join(map(str, tup)), MUh, MU, int(bits))
        store_mu.append((tup, bits, MUh, MU, SDh, SD))

    print(f"\n{'tuple':>14} {'bits':>7} {'rel':>10} {'cos':>10}"
          f"   (quantity: per-window STD)")
    for tup, bits, MUh, MU, SDh, SD in store_mu:
        score("-".join(map(str, tup)), SDh, SD, int(bits))

    print("\nfor comparison, the dedicated moment coder (moment_rd, full "
          "store, 24k steps):")
    print(f"{'dim 64':>14} {128:>7} {0.1617:>10.4f} {0.98529:>10.5f}"
          "   mean block  (std block: rel 0.1788)")
    print(f"{'dim 64':>14} {256:>7} {0.1039:>10.4f} {0.99379:>10.5f}"
          "   mean block  (std block: rel 0.1658)")


if __name__ == "__main__":
    main()
