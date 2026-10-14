"""What is a transmitted moment channel WORTH, per pixel?

The rate-distortion work measured how cheaply the moments can be sent. It
never measured the thing that justifies sending them: whether forcing the
decoder's output to obey them makes every pixel better. 2048 bits (mean
and std at 8 bits per band) is 0.5 b/px against a 10.66 b/px codec -- 4.7%
-- and it constrains all 4096 pixels of the window at once, so the bar it
has to clear is low. But "low bar" is not "cleared".

Takes the trained pyramid's reconstruction and renormalises each window,
per band, to match the TRUE per-window moments:

    out' = (out - mu_out) / sd_out * sd_x + mu_x

and reports the change in the codec's own metric. Three variants, because
they cost very different numbers of bits and carry very different risk:

  mean only   additive; the mean is the well-coded block (rel ~0.10)
  std only    multiplicative; the std is the badly-coded one (rel ~0.13
              at best), so its errors scale every pixel
  both        the full 2048-bit channel

Using the TRUE moments makes every number here an UPPER BOUND on what a
real coded channel delivers -- the point is to find out whether the
ceiling is worth chasing before building the channel.

Statistics are computed over valid pixels only, and applied to all.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import ChipStore
from georvq.factory import from_checkpoint


@torch.no_grad()
def moments(x, valid, eps=1e-6):
    vm = valid.unsqueeze(1).float()
    n = vm.sum((2, 3)).clamp_min(1)
    mu = (x * vm).sum((2, 3)) / n
    var = ((x - mu[:, :, None, None]) ** 2 * vm).sum((2, 3)) / n
    return mu, var.clamp_min(0).sqrt().clamp_min(eps)


@torch.no_grad()
def cos_relL2(out, x, valid):
    """SUMS over valid pixels, not means: a chunk can contain windows that
    are entirely nodata, and averaging per chunk turns those into nan."""
    c = F.cosine_similarity(out, x, dim=1, eps=1e-6)[valid]
    se = (out - x).pow(2).sum(1)
    rel = (se.sqrt() / x.pow(2).sum(1).sqrt().clamp_min(1e-6))[valid]
    return float(c.sum()), float(rel.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="checkpoints/tessera_vq8_k256/model_best.pt")
    ap.add_argument("--device", default="cuda",
                    help="torch device; cuda is the card Slurm allocated")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--tuples", default="20-0-0,20-20-3,20-20-13,20-20-20")
    ap.add_argument("--mean-bits", default="2,4,6",
                    help="bits/band for the quantized-mean variants")
    args = ap.parse_args()
    args.mean_bits = [int(v) for v in args.mean_bits.split(",") if v]

    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    store = ChipStore(cfg.data.store_dir)
    _, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val = fixed_val_tokens(store, va, token=cfg.model.window,
                           max_tokens=args.tokens)
    W = cfg.model.window
    print(f"{args.ckpt}\n  {val.shape[0]} held-out {W}px windows\n")
    print(f"  a full 8-bit moment channel is {2*cfg.model.bands*8} bits/window"
          f" = {2*cfg.model.bands*8/W**2:.3f} b/px\n")

    # per-band spread of the window MEANS, which sets the quantizer step
    ms = []
    for i in range(0, val.shape[0], 8):
        x, v = m.prep(val[i:i + 8], {}, dev)[:2]
        mu, _ = moments(x.float(), v)
        ms.append(mu[v.any(dim=(1, 2))])
    band_sd = torch.cat(ms).std(0).clamp_min(1e-9)
    del ms

    tuples = [tuple(int(v) for v in t.split("-"))
              for t in args.tuples.split(",")]
    cols = ["as-is", "+mean", "+std", "+both"] + \
        [f"+mean@{b}b" for b in args.mean_bits]
    print(f"{'tuple':>10} {'b/px':>7} | " +
          " | ".join(f"{k:>16}" for k in cols))
    print(f"{'':>10} {'':>7} | " +
          " | ".join(f"{'cos':>7} {'rel':>8}" for _ in cols))
    for tup in tuples:
        names = ["as-is", "+mean", "+std", "+both"] + \
            [f"+mean@{b}b" for b in args.mean_bits]
        acc = {k: [0.0, 0.0, 0] for k in names}
        for i in range(0, val.shape[0], 8):
            x, v = m.prep(val[i:i + 8], {}, dev)[:2]
            x = x.float()
            zs = m.encode_levels(x)
            Ls = torch.tensor(tup, device=dev)[None].expand(x.shape[0], -1)
            zq, _, _ = m.quantize(zs, Ls)
            out = m.decode(zq, Ls).float()
            mx, sx = moments(x, v)
            mo, so = moments(out, v)
            b4 = lambda t: t[:, :, None, None]
            variants = {
                "as-is": out,
                "+mean": out - b4(mo) + b4(mx),
                "+std": (out - b4(mo)) * b4(sx / so) + b4(mo),
                "+both": (out - b4(mo)) * b4(sx / so) + b4(mx),
            }
            # how few bits does the MEAN channel actually need? Uniform
            # per-band quantization over +/-3 sigma of that band's spread
            # across windows, so the rate is exactly bands*b bits/window.
            for b in args.mean_bits:
                step = (6.0 * band_sd / (2 ** b)).clamp_min(1e-9)
                mq = torch.round(mx / step) * step
                variants[f"+mean@{b}b"] = out - b4(mo) + b4(mq)
            n = int(v.sum())
            if not n:
                continue
            for k, o in variants.items():
                c, r = cos_relL2(o, x, v)
                acc[k][0] += c; acc[k][1] += r; acc[k][2] += n
            del x, zs, zq, out
        bits = sum(sp.cells * L * math.log2(sp.codes)
                   for sp, L in zip(m.specs, tup))
        cells = " | ".join(
            f"{acc[k][0]/acc[k][2]:>7.4f} {acc[k][1]/acc[k][2]:>8.4f}"
            for k in names)
        print(f"{'-'.join(map(str,tup)):>10} {bits/W**2:>7.3f} | {cells}",
              flush=True)


if __name__ == "__main__":
    main()
