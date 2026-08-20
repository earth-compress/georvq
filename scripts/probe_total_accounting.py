"""Total on-disk archive size per RVQ ladder rung, all overheads included.

Rebuilds the winning entropy model (saved teacher -> SVD rank-16 -> KL
distill -> int8), measures per-(tier, level) coding rates on the whole
corpus, then accounts for every shipped byte at each fill-coarse-first
ladder rung (d-0-0, 6-d-0, 6-6-d — on this ladder the model's
full-depth-parent context assumption is exact):

  stream    N x sum(cells x CE) conditional bits (tier0/L0: order-0)
  sidecar   20 bits/window (independent entropy coding of mean/std)
  ent.model int8 bytes for the shared per-tier projection + position
            embedding + only the per-level heads in use (+ tier0/L0 CDF)
  codebooks fp16, only the levels in use

  python scripts/probe_total_accounting.py \
      --ckpt checkpoints/dem128_overnight_resume/model_best.pt \
      --load-idx <corpus_indices.npz> --teacher <entropy_teacher_dem128.pt>
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_entropy_compress import (quantize_weights, svd_truncate,
                                    train_tier)  # noqa: E402
from probe_index_entropy import (SharedTierModel, accum_full, context_dim,
                                 entropy_bits, gather_context)  # noqa: E402

from georvq.factory import from_checkpoint  # noqa: E402

SIDECAR_BITS = 20


@torch.no_grad()
def level_ce(model, i, ell, lvls, idxs, levels, parents, args, device):
    N, cells = idxs[i].shape[0], idxs[i].shape[1]
    take = min(args.eval_samples, N * cells)
    perm = torch.randperm(N * cells)[:take]
    ce = 0.0
    for s in range(0, take, 1 << 16):
        sel = perm[s:s + (1 << 16)]
        w, c = sel // cells, sel % cells
        y = torch.from_numpy(
            idxs[i][w.numpy(), c.numpy(), ell].astype(np.int64)).to(device)
        feats = gather_context(args.context, i, ell, w, c, idxs, levels,
                               parents, device)
        ce += float(torch.nn.functional.cross_entropy(
            model(feats, c.to(device), lvls.index(ell)), y,
            reduction="sum"))
    return ce / take / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--load-idx", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--context", default="all")
    ap.add_argument("--proj-dim", type=int, default=96)
    ap.add_argument("--head-rank", type=int, default=128)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--distill-steps", type=int, default=500)
    ap.add_argument("--train-batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-samples", type=int, default=2_000_000)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.set_float32_matmul_precision("high")

    model = from_checkpoint(args.ckpt, device).eval()
    specs = model.specs
    levels = [list(rvq.levels) for rvq in model.rvqs]
    z = np.load(args.load_idx)
    idxs = [z[f"tier{i}"] for i in range(len(specs))]
    N = idxs[0].shape[0]
    parents = [accum_full(idxs[i], levels[i], device)
               for i in range(len(specs) - 1)]
    dims = [lv[0].codebook.weight.shape[1] for lv in levels]

    # rebuild teacher -> prune -> distill -> int8
    sds = torch.load(args.teacher, map_location="cpu", weights_only=False)
    finals, h00 = [], None
    for i, sp in enumerate(specs):
        lvls = [l for l in range(sp.depth) if not (i == 0 and l == 0)]
        t = SharedTierModel(context_dim(args.context, i, dims), sp.cells,
                            [sp.codes] * len(lvls),
                            [torch.zeros(sp.codes)] * len(lvls),
                            proj_dim=args.proj_dim, hidden=256,
                            rank=args.head_rank).to(device)
        t.load_state_dict(sds[i])
        t.eval()
        s = svd_truncate(t, args.rank)
        s = train_tier(i, sp, idxs, levels, parents, args, device,
                       distill_from=t, model=s, steps=args.distill_steps)
        finals.append(quantize_weights(s, 8).eval())
    h00 = entropy_bits(np.bincount(idxs[0][:, :, 0].ravel(),
                                   minlength=specs[0].codes))

    # per-(tier, level) rates and component byte counts (int8: 1 B/param)
    ce, head_b, shared_b, cb_b = {}, {}, {}, {}
    for i, sp in enumerate(specs):
        lvls = [l for l in range(sp.depth) if not (i == 0 and l == 0)]
        m = finals[i]
        shared_b[i] = sum(p.numel() for p in m.proj.parameters()) \
            + m.pos.weight.numel()
        for ell in range(sp.depth):
            cb_b[(i, ell)] = sp.codes * dims[i] * 2          # fp16 codebook
            if i == 0 and ell == 0:
                ce[(i, ell)] = h00
                head_b[(i, ell)] = sp.codes * 2               # CDF table
            else:
                ce[(i, ell)] = level_ce(m, i, ell, lvls, idxs, levels,
                                        parents, args, device)
                head_b[(i, ell)] = sum(
                    p.numel() for p in m.heads[lvls.index(ell)].parameters())
            print(f"tier{i} L{ell}: CE={ce[(i, ell)]:.2f} bits "
                  f"head={head_b[(i, ell)] / 1e6:.2f}MB "
                  f"cb={cb_b[(i, ell)] / 1e6:.2f}MB", flush=True)

    px = model.window ** 2
    raw_bits = N * px * 16                                    # int16 metres
    ladder = [(d, 0, 0) for d in range(1, 7)] + \
        [(6, d, 0) for d in range(1, 7)] + \
        [(6, 6, d) for d in range(1, 7)]
    print(f"\ncorpus: {N} windows, raw {raw_bits / 8 / 2**30:.2f} GiB "
          f"(16 bpp int16)\n")
    print("| tuple | stream GiB | sidecar MiB | model MiB | books MiB | "
          "total GiB | eff bpp | vs raw |")
    print("|---|---|---|---|---|---|---|---|")
    for t in ladder:
        used = [(i, l) for i, d in enumerate(t) for l in range(d)]
        tiers = {i for i, _ in used}
        stream = N * sum(specs[i].cells * ce[(i, l)] for i, l in used)
        side = N * SIDECAR_BITS
        mdl = sum(head_b[(i, l)] for i, l in used) \
            + sum(shared_b[i] for i in tiers if i in shared_b)
        books = sum(cb_b[(i, l)] for i, l in used)
        tot_bits = stream + side + 8 * (mdl + books)
        print(f"| {t[0]}-{t[1]}-{t[2]} | {stream / 8 / 2**30:.3f} "
              f"| {side / 8 / 2**20:.1f} | {mdl / 2**20:.2f} "
              f"| {books / 2**20:.2f} | {tot_bits / 8 / 2**30:.3f} "
              f"| {tot_bits / (N * px):.4f} | {raw_bits / tot_bits:,.0f}x |")


if __name__ == "__main__":
    main()
