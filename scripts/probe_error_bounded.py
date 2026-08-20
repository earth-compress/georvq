"""Error-bounded (vs bit-bounded) archiving: per window, pick the cheapest
ladder rung whose reconstruction MAE (metres) meets a bound X; windows that
miss X even at full depth ship full depth and count as violations.

Per-window stream bits are EXACT under the final entropy model (distilled
rank-16 int8) — average-CE accounting would flatter the result, since
expensive-index windows and deep-tuple windows are the same (rough) ones.
The per-window rung choice is itself entropy-coded side info. Global sizes
scale the sampled stream to the full corpus; tables/codebooks ship in full
(any window may use any level).

  python scripts/probe_error_bounded.py \
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
                                 gather_context)  # noqa: E402

from georvq.factory import from_checkpoint  # noqa: E402

SIDECAR_BITS = 20
OVERHEAD_BYTES = (3.28 + 36.0) * 2 ** 20     # int8 entropy model + fp16 books
LADDER = [(d, 0, 0) for d in range(1, 7)] + \
    [(6, d, 0) for d in range(1, 7)] + [(6, 6, d) for d in range(1, 7)]


def _tier_model(args, i, sp, dims, rank, device):
    lvls = [l for l in range(sp.depth) if not (i == 0 and l == 0)]
    return SharedTierModel(context_dim(args.context, i, dims), sp.cells,
                           [sp.codes] * len(lvls),
                           [torch.zeros(sp.codes)] * len(lvls),
                           proj_dim=args.proj_dim, hidden=256,
                           rank=rank).to(device)


def rebuild_final(args, specs, levels, idxs_full, parents_full, device):
    """The distilled rank-r int8 model — built once, cached to --final."""
    dims = [lv[0].codebook.weight.shape[1] for lv in levels]
    if args.final and os.path.exists(args.final):
        sds = torch.load(args.final, map_location="cpu", weights_only=False)
        finals = []
        for i, sp in enumerate(specs):
            m = _tier_model(args, i, sp, dims, args.rank, device)
            m = svd_truncate(m, args.rank)   # match saved head shapes
            m.load_state_dict(sds[i])
            finals.append(m.eval())
        print(f"final entropy model loaded from {args.final}", flush=True)
        return finals
    sds = torch.load(args.teacher, map_location="cpu", weights_only=False)
    finals = []
    for i, sp in enumerate(specs):
        t = _tier_model(args, i, sp, dims, args.head_rank, device)
        t.load_state_dict(sds[i])
        t.eval()
        s = svd_truncate(t, args.rank)
        s = train_tier(i, sp, idxs_full, levels, parents_full, args, device,
                       distill_from=t, model=s, steps=args.distill_steps)
        finals.append(quantize_weights(s, 8).eval())
    if args.final:
        torch.save([m.state_dict() for m in finals], args.final)
        print(f"final entropy model saved -> {args.final}", flush=True)
    return finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--load-idx", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--final", default=None,
                    help="cache path for the distilled int8 model (loaded "
                         "if present, else built and saved)")
    ap.add_argument("--dump", default=None,
                    help="save per-window arrays (mae, maxe, rung_bits, "
                         "lat, lon) to this npz for downstream plots")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sample-chips", type=int, default=7000)
    ap.add_argument("--bounds", default="20,15,10,7,5,3,2")
    ap.add_argument("--context", default="all")
    ap.add_argument("--proj-dim", type=int, default=96)
    ap.add_argument("--head-rank", type=int, default=128)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--distill-steps", type=int, default=500)
    ap.add_argument("--train-batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.set_float32_matmul_precision("high")

    model = from_checkpoint(args.ckpt, device).eval()
    specs, W = model.specs, model.window
    levels = [list(rvq.levels) for rvq in model.rvqs]

    z = np.load(args.load_idx)
    idxs_full = [z[f"tier{i}"] for i in range(len(specs))]
    N_full = idxs_full[0].shape[0]
    parents_full = [accum_full(idxs_full[i], levels[i], device)
                    for i in range(len(specs) - 1)]
    p0 = np.bincount(idxs_full[0][:, :, 0].ravel(),
                     minlength=specs[0].codes).astype(np.float64)
    logp0 = np.log2(np.maximum(p0, 0.5) / p0.sum())
    finals = rebuild_final(args, specs, levels, idxs_full, parents_full,
                           device)
    del parents_full
    print("entropy model rebuilt", flush=True)

    # ---- sample windows: raw pixels + per-rung MAE + per-tier indices ----
    from georvq.data.store import ChipStore
    store = ChipStore("/data/georvq-stores/copdem_356")
    rows = store.index.sample(args.sample_chips,
                              random_state=0).sort_values("shard")
    offs = np.linspace(0, store.chip_px - W, 3, dtype=int)
    wins, lls, shard = [], [], {}
    for _, r in rows.iterrows():
        sh = int(r["shard"])
        if sh not in shard:
            shard = {sh: store.open_shard(sh)}
        a = np.asarray(shard[sh][int(r["slot"])])
        for rr in offs:
            for cc in offs:
                t = a[:, rr:rr + W, cc:cc + W]
                if (t[0] > -32766.5).any():
                    wins.append(t.copy())
                    lls.append((float(r.lat), float(r.lon)))
    X = torch.from_numpy(np.stack(wins))
    N = X.shape[0]
    print(f"{N} sample windows", flush=True)

    mae = np.zeros((N, len(LADDER)), np.float32)
    maxe = np.zeros((N, len(LADDER)), np.float32)
    s_idx = [np.zeros((N, sp.cells, sp.depth), np.int16) for sp in specs]
    with torch.no_grad():
        for s in range(0, N, 256):
            raw = X[s:s + 256]
            x, valid = model.source.prep(raw, device)
            stats = model.source.last_stats()
            _, cums = model.level_cumsums(x)
            zs = model.encode_levels(x)
            for i, rvq in enumerate(model.rvqs):
                residual = zs[i].float()
                for j, lev in enumerate(rvq.levels):
                    q, idx, _ = lev(residual)
                    residual = residual - q
                    s_idx[i][s:s + raw.shape[0], :, j] = \
                        idx.view(raw.shape[0], -1).cpu().numpy()
            norm = model.source.normalizer(None)
            xm = norm.denormalize(x, stats)[:, 0]
            vm = valid.float()
            nv = vm.sum((1, 2)).clamp_min(1)
            for k, t in enumerate(LADDER):
                out = model.eval_tuple(cums, list(t), x)
                om = norm.denormalize(out, stats)[:, 0]
                err = (om - xm).abs() * vm       # invalid px zeroed
                mae[s:s + raw.shape[0], k] = \
                    (err.sum((1, 2)) / nv).cpu().numpy()
                maxe[s:s + raw.shape[0], k] = \
                    err.amax(dim=(1, 2)).cpu().numpy()
    print("MAE grid done", flush=True)

    # ---- exact per-window bits per (tier, level) under the final model ----
    parents_s = [accum_full(s_idx[i], levels[i], device)
                 for i in range(len(specs) - 1)]
    bits = {}
    bits[(0, 0)] = -logp0[s_idx[0][:, 0, 0]]
    with torch.no_grad():
        for i, sp in enumerate(specs):
            lvls = [l for l in range(sp.depth) if not (i == 0 and l == 0)]
            cells = sp.cells
            for ell in lvls:
                acc = np.zeros(N, np.float64)
                tot = N * cells
                for s in range(0, tot, 1 << 16):
                    sel = torch.arange(s, min(s + (1 << 16), tot))
                    w, c = sel // cells, sel % cells
                    y = torch.from_numpy(
                        s_idx[i][w.numpy(), c.numpy(), ell]
                        .astype(np.int64)).to(device)
                    feats = gather_context(args.context, i, ell, w, c,
                                           s_idx, levels, parents_s, device)
                    lp = finals[i](feats, c.to(device),
                                   lvls.index(ell)).log_softmax(1)
                    nll = -lp.gather(1, y[:, None])[:, 0] / math.log(2)
                    np.add.at(acc, w.numpy(), nll.cpu().numpy())
                bits[(i, ell)] = acc
            print(f"tier{i} bits done", flush=True)

    rung_bits = np.stack([sum(bits[(i, l)] for i, d in enumerate(t)
                              for l in range(d)) for t in LADDER], 1)
    if args.dump:
        ll = np.array(lls, np.float32)
        np.savez(args.dump, mae=mae, maxe=maxe,
                 rung_bits=rung_bits.astype(np.float32),
                 lat=ll[:, 0], lon=ll[:, 1])
        print(f"per-window arrays -> {args.dump}", flush=True)

    # ---- bounds sweep ----
    scale = N_full / N
    raw_bits_full = N_full * W * W * 16
    print(f"\nsample {N} windows -> scaled to {N_full}; raw "
          f"{raw_bits_full / 8 / 2**30:.2f} GiB")
    print("| bound | viol% | mean MAE | p95 MAE | mean maxE | p99 maxE | "
          "global maxE | mean bits/win | choice bits | total GiB | eff bpp "
          "| vs raw |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for X_m in (float(b) for b in args.bounds.split(",")):
        ok = mae <= X_m
        cost = np.where(ok, rung_bits, np.inf)
        pick = cost.argmin(1)
        viol = ~ok.any(1)
        pick[viol] = len(LADDER) - 1
        wb = rung_bits[np.arange(N), pick]
        amae = mae[np.arange(N), pick]
        amax = maxe[np.arange(N), pick]
        pc = np.bincount(pick, minlength=len(LADDER)).astype(np.float64)
        pnz = pc[pc > 0] / pc.sum()
        h_choice = float(-(pnz * np.log2(pnz)).sum())
        stream = wb.sum() * scale
        tot = stream + N_full * (SIDECAR_BITS + h_choice) \
            + 8 * OVERHEAD_BYTES
        print(f"| {X_m:g} m | {100 * viol.mean():.2f} | {amae.mean():.2f} m "
              f"| {np.percentile(amae, 95):.2f} m | {amax.mean():.1f} m "
              f"| {np.percentile(amax, 99):.1f} m | {amax.max():.1f} m "
              f"| {wb.mean():,.0f} | {h_choice:.2f} "
              f"| {tot / 8 / 2**30:.3f} | {tot / (N_full * W * W):.4f} "
              f"| {raw_bits_full / tot:,.0f}x |")

    print("\nfixed-rung reference (same sample):")
    for k in (5, 11, 13, 15, 17):
        wb = rung_bits[:, k]
        tot = wb.sum() * scale + N_full * SIDECAR_BITS + 8 * OVERHEAD_BYTES
        t = LADDER[k]
        print(f"  {t[0]}-{t[1]}-{t[2]}: mean MAE {mae[:, k].mean():.2f} m "
              f"p95 {np.percentile(mae[:, k], 95):.2f} m | "
              f"mean maxE {maxe[:, k].mean():.1f} m "
              f"global {maxe[:, k].max():.1f} m | "
              f"{tot / 8 / 2**30:.3f} GiB | {raw_bits_full / tot:,.0f}x")


if __name__ == "__main__":
    main()
