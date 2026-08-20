"""Pretrain the location -> coarse-latent prior for predict-and-subtract.

The prior is independent of the codebooks and entropy model: it maps a
chip's location embedding to a prediction of the coarse tier's latent.
Training it standalone is orders of magnitude faster than jointly (tiny
MLP, huge batch) and gives every run a stationary, converged residual to
quantize. Objective MSE(z0, prior(loc)) -> prior(loc) = E[z0 | location],
exactly what predict-and-subtract wants.

  python scripts/pretrain_location_prior.py --store-dir /data/... \
      --encoder-ckpt checkpoints/stage1/model.pt --out checkpoints/location_prior/base.pt
"""
from __future__ import annotations

import argparse
import os

import torch

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import ChipStore
from georvq.model.location import (build_base, build_location_table,
                                   get_location_encoder)


def load_encoder(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if "config" in ck:
        from georvq.factory import from_checkpoint
        return from_checkpoint(path, device)
    from georvq.compat import from_legacy_checkpoint
    return from_legacy_checkpoint(path, device)[0]


@torch.no_grad()
def collect_pairs(store, rows, model, enc_loc, device, per_chip, max_chips,
                  batch=512):
    """(z0, loc) pairs: coarse-tier latents of grid windows + each window's
    chip location embedding."""
    table = build_location_table(rows, device, enc_loc)
    toks, locs = fixed_val_tokens(store, rows, token=model.window,
                                  per_chip=per_chip, max_chips=max_chips,
                                  max_tokens=10**9, sat_table=table.cpu())
    zs = []
    for i in range(0, toks.shape[0], batch):
        x, _ = model.source.prep(toks[i:i + batch], device)
        zs.append(model.encode_levels(x)[0][:, 0].float())   # coarse: 1 cell
    return torch.cat(zs), locs.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--encoder-ckpt", required=True)
    ap.add_argument("--out", default="checkpoints/location_prior/base.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--encoder", choices=["satclip", "sphharm"],
                    default="satclip")
    ap.add_argument("--degree", type=int, default=40,
                    help="sphharm only: SH band limit (degree^2 features)")
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
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() and args.gpu >= 0
                          else "cpu")
    torch.set_float32_matmul_precision("high")

    store = ChipStore(args.store_dir)
    train_rows, val_rows = store.split(args.val_frac, seed=args.split_seed)
    model = load_encoder(args.encoder_ckpt, device)
    enc_loc = get_location_encoder(args.encoder, device, args.degree)
    X, S = collect_pairs(store, train_rows, model, enc_loc, device,
                         args.per_chip, args.max_chips)
    Xv, Sv = collect_pairs(store, val_rows, model, enc_loc, device,
                           args.per_chip, max_chips=2000)
    sdim = S.shape[1]
    print(f"train {X.shape[0]} pairs (z dim {X.shape[1]}, loc {sdim}d); "
          f"val {Xv.shape[0]}", flush=True)

    base = build_base(sdim, X.shape[1], args.hidden, args.layers,
                      bottleneck=args.bottleneck,
                      dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(base.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    var_v = Xv.var(0).mean()
    best, best_sd, since = float("inf"), None, 0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, X.shape[0], (min(args.batch, X.shape[0]),),
                            device=device)
        loss = torch.nn.functional.mse_loss(base(S[idx]), X[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.eval_every == 0:
            base.eval()
            with torch.no_grad():
                vmse = torch.nn.functional.mse_loss(base(Sv), Xv).item()
            base.train()
            r2 = 1 - vmse / float(var_v)
            print(f"step {step}: train_mse={loss.item():.4f} "
                  f"val_mse={vmse:.4f} val_R2={r2:.3f}", flush=True)
            if vmse < best:
                best, since = vmse, 0
                best_sd = {k: v.detach().cpu().clone()
                           for k, v in base.state_dict().items()}
            else:
                since += 1
                if since >= args.patience:
                    print(f"early stop at step {step} "
                          f"(best val_mse={best:.4f})", flush=True)
                    break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"base": best_sd, "sdim": sdim, "hidden": args.hidden,
                "layers": args.layers, "bottleneck": args.bottleneck,
                "dropout": args.dropout,
                "encoder": args.encoder, "degree": args.degree,
                "val_mse": best, "val_r2": 1 - best / float(var_v),
                "encoder_ckpt": str(args.encoder_ckpt)}, args.out)
    print(f"saved -> {args.out} (val_R2={1 - best / float(var_v):.3f})")


if __name__ == "__main__":
    main()
