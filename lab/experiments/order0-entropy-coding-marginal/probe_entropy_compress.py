"""Compress the shared-arch entropy model: SVD rank pruning + KL
distillation finetune + weight quantization, reporting the size/rate curve.

The shared model's per-level head is Linear(256->rank)->Linear(rank->K)
with no nonlinearity between, i.e. a rank-limited factorization of a
256->K map — so pruning = SVD-truncating the effective matrix to a lower
rank (exact, no retraining), distillation = finetuning the truncated model
against the teacher's soft logits, and quantization = per-output-channel
symmetric intN on every Linear weight (encoder and decoder share the same
quantized model, so probabilities stay in sync by construction).

  python scripts/probe_entropy_compress.py \
      --ckpt checkpoints/dem128_overnight_resume/model_best.pt \
      --load-idx <corpus_indices.npz>
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_index_entropy import (SharedTierModel, accum_full, context_dim,
                                 entropy_bits, gather_context)  # noqa: E402

from georvq.factory import from_checkpoint  # noqa: E402


def train_tier(i, sp, idxs, levels, parents, args, device, distill_from=None,
               model=None, steps=None):
    """Train (or KL-distill-finetune) one tier's shared model on all data."""
    N, cells = idxs[i].shape[0], sp.cells
    lvls = [l for l in range(sp.depth) if not (i == 0 and l == 0)]
    if model is None:
        counts = [np.bincount(idxs[i][:, :, l].ravel(), minlength=sp.codes)
                  for l in lvls]
        freqs = [torch.from_numpy(np.log(c + 0.5) - np.log(c.sum())).float()
                 for c in counts]
        dims = [lv[0].codebook.weight.shape[1] for lv in levels]
        model = SharedTierModel(context_dim(args.context, i, dims), cells,
                                [sp.codes] * len(lvls), freqs,
                                proj_dim=args.proj_dim, hidden=256,
                                rank=args.head_rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    for step in range((steps or args.train_steps) * len(lvls)):
        ell = lvls[step % len(lvls)]
        w = torch.randint(0, N, (args.train_batch,))
        c = torch.randint(0, cells, (args.train_batch,))
        y = torch.from_numpy(
            idxs[i][w.numpy(), c.numpy(), ell].astype(np.int64)).to(device)
        with torch.no_grad():
            feats = gather_context(args.context, i, ell, w, c, idxs, levels,
                                   parents, device)
        logits = model(feats, c.to(device), lvls.index(ell))
        loss = nn.functional.cross_entropy(logits, y)
        if distill_from is not None:
            with torch.no_grad():
                t = distill_from(feats, c.to(device), lvls.index(ell))
            loss = 0.5 * loss + 0.5 * nn.functional.kl_div(
                logits.log_softmax(1), t.log_softmax(1),
                log_target=True, reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def tier_bits(i, sp, model, idxs, levels, parents, args, device, h00=None):
    """Sum over levels of cells * CE bits (order-0 for tier0/L0)."""
    N, cells = idxs[i].shape[0], sp.cells
    lvls = [l for l in range(sp.depth) if not (i == 0 and l == 0)]
    total = 0.0 if i != 0 else h00 * 1.0     # tier0 L0 coded order-0
    for ell in lvls:
        take = min(args.eval_samples, N * cells)
        perm = torch.randperm(N * cells)[:take]
        ce = 0.0
        for s in range(0, take, 1 << 16):
            sel = perm[s:s + (1 << 16)]
            w, c = sel // cells, sel % cells
            y = torch.from_numpy(
                idxs[i][w.numpy(), c.numpy(), ell].astype(np.int64)
            ).to(device)
            feats = gather_context(args.context, i, ell, w, c, idxs, levels,
                                   parents, device)
            ce += float(nn.functional.cross_entropy(
                model(feats, c.to(device), lvls.index(ell)), y,
                reduction="sum"))
        total += cells * ce / take / math.log(2)
    return total


def svd_truncate(model, rank):
    """Rank-r copy: SVD of each head's effective (K x 256) linear map."""
    m = copy.deepcopy(model)
    for h, head in enumerate(m.heads):
        w2, b2 = head[2].weight.data, head[2].bias.data    # 256 -> rank0
        w3, b3 = head[3].weight.data, head[3].bias.data    # rank0 -> K
        M = w3 @ w2                                        # (K, 256)
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        A = U[:, :rank] * S[:rank]                         # (K, r)
        lo = nn.Linear(w2.shape[1], rank, bias=False)
        lo.weight.data.copy_(Vt[:rank])
        hi = nn.Linear(rank, w3.shape[0])
        hi.weight.data.copy_(A)
        hi.bias.data.copy_(w3 @ b2 + b3)
        m.heads[h] = nn.Sequential(head[0], head[1], lo, hi)
    return m.to(next(model.parameters()).device)


def quantize_weights(model, bits):
    """Per-output-channel symmetric fake-quantization of Linear weights."""
    m = copy.deepcopy(model)
    qmax = 2 ** (bits - 1) - 1
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data
            scale = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
            mod.weight.data = (w / scale).round().clamp(-qmax, qmax) * scale
    return m


def model_bytes(model, bits):
    n = sum(p.numel() for p in model.parameters())
    return n, n * bits / 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--load-idx", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--context", default="all")
    ap.add_argument("--proj-dim", type=int, default=96)
    ap.add_argument("--head-rank", type=int, default=128)
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--distill-steps", type=int, default=500)
    ap.add_argument("--train-batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--distill-lr", type=float, default=3e-4)
    ap.add_argument("--eval-samples", type=int, default=2_000_000)
    ap.add_argument("--ranks", default="64,32,16")
    ap.add_argument("--precisions", default="16,8,4")
    ap.add_argument("--save-teacher", default=None)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.set_float32_matmul_precision("high")

    model = from_checkpoint(args.ckpt, device).eval()
    specs = model.specs
    levels = [list(rvq.levels) for rvq in model.rvqs]
    z = np.load(args.load_idx)
    idxs = [z[f"tier{i}"] for i in range(len(specs))]
    print(f"{idxs[0].shape[0]} windows loaded", flush=True)
    parents = [accum_full(idxs[i], levels[i], device)
               for i in range(len(specs) - 1)]
    h00 = entropy_bits(np.bincount(idxs[0][:, :, 0].ravel(),
                                   minlength=specs[0].codes))

    def report(tag, models, bits_per_w):
        tot = sum(tier_bits(i, sp, m, idxs, levels, parents, args, device,
                            h00) for (i, sp), m in zip(enumerate(specs),
                                                       models))
        n, byt = zip(*(model_bytes(m, bits_per_w) for m in models))
        print(f"{tag}: {tot:,.0f} bits/window | {sum(n) / 1e6:.2f}M params "
              f"| {sum(byt) / 2 ** 20:.1f} MiB", flush=True)
        return tot

    teachers = [train_tier(i, sp, idxs, levels, parents, args, device)
                for i, sp in enumerate(specs)]
    if args.save_teacher:
        torch.save([m.state_dict() for m in teachers], args.save_teacher)
    report("teacher rank128 fp32", teachers, 32)
    for p in args.precisions.split(","):
        report(f"teacher rank128 int{p}" if int(p) < 16 else
               "teacher rank128 fp16",
               [quantize_weights(m, int(p)) for m in teachers], int(p))

    lr0 = args.lr
    for r in (int(x) for x in args.ranks.split(",")):
        pruned = [svd_truncate(m, r) for m in teachers]
        report(f"pruned rank{r} (no finetune) fp32", pruned, 32)
        args.lr = args.distill_lr
        distilled = [train_tier(i, sp, idxs, levels, parents, args, device,
                                distill_from=teachers[i], model=pruned[i],
                                steps=args.distill_steps)
                     for i, sp in enumerate(specs)]
        args.lr = lr0
        report(f"distilled rank{r} fp32", distilled, 32)
        for p in args.precisions.split(","):
            report(f"distilled rank{r} int{p}" if int(p) < 16 else
                   f"distilled rank{r} fp16",
                   [quantize_weights(m, int(p)) for m in distilled], int(p))


if __name__ == "__main__":
    main()
