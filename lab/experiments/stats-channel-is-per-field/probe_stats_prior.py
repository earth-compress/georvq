"""Probe: location prior for the (mean, std) stats side-channel.

The DEM sidecar ships each window's masked (mean, std) in metres so decoding
can restore absolute elevation. Direct coding costs H(q(mean)) + H(q(log sd))
bits/window; a location->stats predictor lets us send the quantized residual
instead. This probe fits that predictor and reports order-0 empirical
entropies (same methodology as the codec-as-prior lossless probes) for
direct vs residual coding, on train chips (the corpus-fit number, since DEM
runs are closed-distribution) and held-out chips.

Targets are (mean_m, log2 sd); quantization grids are shared between direct
and residual coding so the comparison is exact: mean at --mean-step metres,
log2(sd) at --logsd-step (0.01 ~ 0.7% relative amplitude steps).

  python scripts/probe_stats_prior.py --store-dir /data/... \
      --encoder satclip --window 128
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import ChipStore
from georvq.model.location import (build_base, build_location_table,
                                   get_location_encoder)
from georvq.sources.dem import NODATA, DEMNormalizer


def window_stats(toks, batch=4096):
    """Masked (mean, log2 sd) per window, (N, 2) float32 metres/log2-metres."""
    norm = DEMNormalizer()
    out = []
    for i in range(0, toks.shape[0], batch):
        x = toks[i:i + batch].float()
        _, (mu, sd) = norm.normalize(x, x[:, 0] != NODATA)
        out.append(torch.cat([mu, torch.log2(sd)], 1))
    return torch.cat(out)


def entropy_bits(q: np.ndarray) -> float:
    """Order-0 empirical entropy (bits/symbol) of an integer array."""
    _, counts = np.unique(q, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def code_report(y, pred, steps):
    """Per-channel direct vs residual entropy at the shared grids."""
    rep = {}
    for c, (name, step) in enumerate(zip(["mean_m", "log2_sd"], steps)):
        direct = np.round(y[:, c] / step).astype(np.int64)
        resid = np.round((y[:, c] - pred[:, c]) / step).astype(np.int64)
        rep[name] = {"direct": entropy_bits(direct),
                     "residual": entropy_bits(resid),
                     "rmse": float(np.sqrt(((y[:, c] - pred[:, c]) ** 2)
                                           .mean()))}
    return rep


def collect(store, rows, enc, device, window, per_chip, max_chips):
    table = build_location_table(rows, device, enc)
    toks, locs = fixed_val_tokens(store, rows, token=window,
                                  per_chip=per_chip, max_chips=max_chips,
                                  max_tokens=10**9, sat_table=table.cpu())
    return window_stats(toks).to(device), locs.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--encoder", choices=["satclip", "sphharm"],
                    default="satclip")
    ap.add_argument("--degree", type=int, default=40)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--per-chip", type=int, default=16)
    ap.add_argument("--max-chips", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--bottleneck", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--mean-step", type=float, default=1.0)
    ap.add_argument("--logsd-step", type=float, default=0.01)
    ap.add_argument("--out", default=None,
                    help="optionally save the fitted predictor")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() and args.gpu >= 0
                          else "cpu")
    torch.set_float32_matmul_precision("high")

    store = ChipStore(args.store_dir)
    train_rows, val_rows = store.split(args.val_frac, seed=args.split_seed)
    enc = get_location_encoder(args.encoder, device, args.degree)
    Y, S = collect(store, train_rows, enc, device, args.window,
                   args.per_chip, args.max_chips)
    Yv, Sv = collect(store, val_rows, enc, device, args.window,
                     args.per_chip, max_chips=2000)
    print(f"train {Y.shape[0]} windows (loc {S.shape[1]}d); "
          f"val {Yv.shape[0]}", flush=True)

    # standardized targets for optimization; predictions mapped back for bits
    y_mu, y_sd = Y.mean(0), Y.std(0)
    Yn, Yvn = (Y - y_mu) / y_sd, (Yv - y_mu) / y_sd
    base = build_base(S.shape[1], 2, args.hidden, args.layers,
                      bottleneck=args.bottleneck,
                      dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(base.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    best, best_sd, since = float("inf"), None, 0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, Y.shape[0], (min(args.batch, Y.shape[0]),),
                            device=device)
        loss = torch.nn.functional.mse_loss(base(S[idx]), Yn[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.eval_every == 0:
            base.eval()
            with torch.no_grad():
                vmse = torch.nn.functional.mse_loss(base(Sv), Yvn).item()
            base.train()
            print(f"step {step}: train_mse={loss.item():.4f} "
                  f"val_mse={vmse:.4f} val_R2={1 - vmse:.3f}", flush=True)
            if vmse < best:
                best, since = vmse, 0
                best_sd = {k: v.detach().cpu().clone()
                           for k, v in base.state_dict().items()}
            else:
                since += 1
                if since >= args.patience:
                    print(f"early stop at step {step}", flush=True)
                    break

    base.load_state_dict(best_sd)
    base.eval()
    steps = (args.mean_step, args.logsd_step)
    with torch.no_grad():
        for split, y, s in [("train", Y, S), ("val", Yv, Sv)]:
            pred = (base(s) * y_sd + y_mu).cpu().numpy()
            rep = code_report(y.cpu().numpy(), pred, steps)
            tot_d = sum(v["direct"] for v in rep.values())
            tot_r = sum(v["residual"] for v in rep.values())
            print(f"\n[{split}] bits/window: direct {tot_d:.2f} "
                  f"-> residual {tot_r:.2f} "
                  f"(saving {tot_d - tot_r:.2f}, {100 * (1 - tot_r / tot_d):.0f}%)")
            for name, v in rep.items():
                print(f"  {name:8s} H_direct={v['direct']:.2f} "
                      f"H_resid={v['residual']:.2f} rmse={v['rmse']:.3g}")
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"base": best_sd, "sdim": S.shape[1],
                    "hidden": args.hidden, "layers": args.layers,
                    "bottleneck": args.bottleneck, "dropout": args.dropout,
                    "encoder": args.encoder, "degree": args.degree,
                    "y_mu": y_mu.cpu(), "y_sd": y_sd.cpu(),
                    "val_mse": best}, args.out)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
