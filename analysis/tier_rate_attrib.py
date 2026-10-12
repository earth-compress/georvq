"""Where does the remaining quantization error live, per tier?

At the top of the baseline rate range (~25-30 b/px) there is enough budget
to stop quantizing some tiers altogether and send their latents as int8
instead. Which tiers that is worth doing for depends on how much of the
gap between the RVQ operating point and the unquantized ceiling each tier
is responsible for -- and the tiers cost wildly different amounts to send
raw, because their latent payloads differ by 128x:

  tier 0   0.125 dims/px ->   1 b/px at int8   (vs 0.039 b/px at 20 levels)
  tier 1   2.000 dims/px ->  16 b/px           (vs 0.625)
  tier 2  16.000 dims/px -> 128 b/px           (vs 10.0)

So this decodes every combination of {quantized, int8-passthrough} per
tier and reports the achieved quality against the rate it costs. An
"int8 passthrough" tier is scored with its latent round-tripped through
int8 rather than left in fp32, so the number is what a real bitstream
would deliver, not an unreachable bound.

The marginal value of MORE RVQ depth is reported alongside, from the
per-level residual decay, since that is the competing way to spend the
same bits (0.5 b/px per tier-2 level, 0.031 per tier-1 level).
"""
from __future__ import annotations

import argparse
import itertools
import math

import torch
import torch.nn.functional as F

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import open_store
from georvq.factory import from_checkpoint
from georvq.metrics import centered_cos


def int8_roundtrip(z):
    """Per-cell max-abs int8 quantization -- what sending a latent raw costs.

    Scored honestly: a passthrough tier still pays its own 8-bit error, so
    the comparison against extra RVQ depth is like-for-like."""
    s = z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 127.0
    return torch.round(z / s).clamp(-127, 127) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg = m.full_config
    W = cfg.model.window
    store = open_store(cfg.data.store_dir)
    _, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val = fixed_val_tokens(store, va, token=W, max_tokens=args.tokens)
    specs = m.specs
    nt = len(specs)
    print(f"{args.ckpt}\n  {val.shape[0]} val windows\n")

    def tier_rate(i, mode):
        sp = specs[i]
        if mode == "q":
            return sp.cells * sp.depth * math.log2(sp.codes) / W ** 2
        return sp.cells * sp.dim * 8 / W ** 2          # int8 passthrough

    combos = list(itertools.product(*[["q", "i8"]] * nt))
    acc = {c: [0.0, 0.0, 0] for c in combos}
    acc["ceil"] = [0.0, 0.0, 0]
    for i in range(0, val.shape[0], args.chunk):
        x, v = m.source.prep(val[i:i + args.chunk], dev)
        x = x.float()
        st = m._stats_of(x, v)
        zs = m.encode_levels(x, st, v)
        Ls = torch.tensor([sp.depth for sp in specs],
                          device=dev)[None].expand(x.shape[0], -1)
        zq, _, _ = m.quantize(zs, Ls)
        n = int(v.sum())
        for c in combos:
            mix = [zq[t] if c[t] == "q" else int8_roundtrip(zs[t])
                   for t in range(nt)]
            o = m.decode(mix, Ls, st, v).float()
            acc[c][0] += float(F.cosine_similarity(o, x, dim=1)[v].sum())
            acc[c][1] += float(centered_cos(o, x, v)) * n
            acc[c][2] += n
        o = m.decode(zs, Ls, st, v).float()
        acc["ceil"][0] += float(F.cosine_similarity(o, x, dim=1)[v].sum())
        acc["ceil"][1] += float(centered_cos(o, x, v)) * n
        acc["ceil"][2] += n
        del x, zs, zq

    sb = m.side_bits() / W ** 2
    rows = []
    for c in combos:
        r = sum(tier_rate(t, c[t]) for t in range(nt)) + sb
        rows.append((r, "-".join(c), acc[c][0] / acc[c][2],
                     acc[c][1] / acc[c][2]))
    rows.sort()
    print(f"{'tiers (0-1-2)':>16} {'b/px':>8} {'cos':>9} {'ccos':>8} "
          f"{'gap closed':>11}")
    base = next(r for r in rows if r[1] == "q-q-q")
    ceil_c = acc["ceil"][0] / acc["ceil"][2]
    ceil_cc = acc["ceil"][1] / acc["ceil"][2]
    for r, name, c, cc in rows:
        frac = ((cc - base[3]) / (ceil_cc - base[3])
                if ceil_cc > base[3] else float("nan"))
        print(f"{name:>16} {r:>8.2f} {c:>9.5f} {cc:>8.4f} {100*frac:>10.1f}%")
    print(f"{'CEILING':>16} {'-':>8} {ceil_c:>9.5f} {ceil_cc:>8.4f} "
          f"{100.0:>10.1f}%")
    print("\n'gap closed' = fraction of the ccos gap between the all-quantized "
          "point\nand the unquantized ceiling that this combination recovers.")
    print(f"stats side channel included in every rate: {sb:.3f} b/px")


if __name__ == "__main__":
    main()
