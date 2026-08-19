"""Does a rotation+Lloyd refinement of tier 2 actually decode better?

`turboquant.py` showed ~1 bit/dim on the post-RVQ residual halves the
LATENT error. That is not the question. The decoder was trained on
RVQ-quantized inputs, so a latent more accurate than any it saw in
training need not reconstruct proportionally better -- the model may
simply not use the extra precision.

So: refine tier 2's latent, decode through the real model, and report the
codec's own metrics against the two anchors that bound the answer --
the unrefined RVQ operating point, and the unquantized ceiling.

The refinement is codebook-free by construction: a fixed random rotation
(a seed) plus Lloyd-Max levels fitted once on held-out residuals, which on
rotated coordinates converge to the Gaussian quantizer. Rate is charged as
the empirical entropy of the symbols, i.e. what an arithmetic coder spends.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turboquant import entropy_of, lloyd_batch, quant_batch    # noqa: E402

from georvq.data.pool import fixed_val_tokens                  # noqa: E402
from georvq.data.store import ChipStore                        # noqa: E402
from georvq.factory import from_checkpoint                     # noqa: E402
from georvq.metrics import centered_cos                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--fit-tokens", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--levels", default="2,3,4")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{args.gpu}")
    m = from_checkpoint(args.ckpt, dev)
    cfg, W = m.full_config, m.full_config.model.window
    store = ChipStore(cfg.data.store_dir)
    _, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    val = fixed_val_tokens(store, va, token=W,
                           max_tokens=args.tokens + args.fit_tokens)
    fit_s, val_s = val[:args.fit_tokens], val[args.fit_tokens:]
    specs = m.specs
    sp = specs[args.tier]
    full = torch.tensor([s.depth for s in specs], device=dev)
    d = sp.dim

    def tier_resid(raw):
        """RVQ residual of `tier` at full depth, for a batch of raw tokens."""
        out = []
        for i in range(0, raw.shape[0], args.chunk):
            x, v = m.source.prep(raw[i:i + args.chunk], dev)
            x = x.float()
            st = m._stats_of(x, v)
            zs = m.encode_levels(x, st, v)
            Ls = full[None].expand(x.shape[0], -1)
            zq, _, _ = m.quantize(zs, Ls)
            r = (zs[args.tier] - zq[args.tier]).reshape(-1, d)
            out.append(r.float())
        return torch.cat(out)

    # fixed random rotation: the whole "codebook" is this seed
    g = torch.Generator(device=dev).manual_seed(0)
    R, _ = torch.linalg.qr(torch.randn(d, d, generator=g, device=dev))
    Rtr = tier_resid(fit_s) @ R
    levels = [int(v) for v in args.levels.split(",")]
    tables = {L: lloyd_batch(Rtr.T, L) for L in levels}
    del Rtr

    base_bits = sum(s.cells * s.depth * math.log2(s.codes) for s in specs)
    side = m.side_bits() / W ** 2   # side_bits() is bits/window
    acc = {k: [0.0, 0.0, 0] for k in ["rvq", "ceiling"] + [f"L{L}" for L in levels]}
    ent = {L: [0.0, 0] for L in levels}

    for i in range(0, val_s.shape[0], args.chunk):
        x, v = m.source.prep(val_s[i:i + args.chunk], dev)
        x = x.float()
        n = int(v.sum())
        if not n:
            continue
        st = m._stats_of(x, v)
        zs = m.encode_levels(x, st, v)
        Ls = full[None].expand(x.shape[0], -1)
        zq, _, _ = m.quantize(zs, Ls)

        def score(key, zlist):
            o = m.decode(zlist, Ls, st, v).float()
            a = acc[key]
            a[0] += float(F.cosine_similarity(o, x, dim=1)[v].sum())
            a[1] += float(centered_cos(o, x, v)) * n
            a[2] += n

        score("rvq", zq)
        score("ceiling", zs)
        r = (zs[args.tier] - zq[args.tier])
        shape = r.shape
        rf = r.reshape(-1, d) @ R
        for L in levels:
            I, Yq = quant_batch(rf.T, tables[L])
            fix = (Yq.T @ R.T).reshape(shape)
            mix = list(zq)
            mix[args.tier] = zq[args.tier] + fix
            score(f"L{L}", mix)
            ent[L][0] += entropy_of(I.T.cpu()) * shape[0] * shape[1]
            ent[L][1] += shape[0] * shape[1]
        del x, zs, zq

    print(f"{args.ckpt}\n  tier {args.tier}, {acc['rvq'][2]:,} valid px, "
          f"base {base_bits/W**2 + side:.2f} b/px\n")
    print(f"{'variant':>12} {'b/px':>8} {'cos':>9} {'ccos':>8} "
          f"{'cos gap closed':>15} {'ccos gap closed':>16}")
    cr = acc["rvq"][0] / acc["rvq"][2], acc["rvq"][1] / acc["rvq"][2]
    ce = acc["ceiling"][0] / acc["ceiling"][2], acc["ceiling"][1] / acc["ceiling"][2]
    rows = [("RVQ x20", base_bits / W ** 2 + side, cr[0], cr[1])]
    for L in levels:
        a = acc[f"L{L}"]
        bpd = ent[L][0] / max(ent[L][1], 1) / d
        add = bpd * d * sp.cells / W ** 2
        rows.append((f"+rot L={L}", base_bits / W ** 2 + side + add,
                     a[0] / a[2], a[1] / a[2]))
    rows.append(("CEILING", float("nan"), ce[0], ce[1]))
    for name, b, c, cc in rows:
        gc = "" if name == "RVQ x20" or math.isnan(c) else \
            f"{100*(c-cr[0])/max(ce[0]-cr[0],1e-9):>14.1f}%"
        gcc = "" if name == "RVQ x20" else \
            f"{100*(cc-cr[1])/max(ce[1]-cr[1],1e-9):>15.1f}%"
        bs = "     -  " if math.isnan(b) else f"{b:>8.2f}"
        print(f"{name:>12} {bs} {c:>9.5f} {cc:>8.4f} {gc:>15} {gcc:>16}")
    print("\n'gap closed' = fraction of the RVQ-to-ceiling gap recovered.")


if __name__ == "__main__":
    main()
