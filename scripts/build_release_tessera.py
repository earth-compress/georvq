"""Build the Tessera release bundle, then verify it end to end.

Fits the two pieces of shipped state that are not weights — the mean
channel's per-band constants and the refinement's Lloyd table — on
held-out windows encoded with the shipped checkpoint, writes the bundle,
reloads it from disk, and measures the rate ladder through the reloaded
release. The verification is the point: it is the only way to know that
the *quantized* mean the decoder receives is as good as the float mean
training used.

    python scripts/build_release_tessera.py \
        --ckpt .../statscond_step32500.pt \
        --out /data/georvq-releases/tessera-beta-v1/bundle --gpu 1
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from georvq.data.pool import fixed_val_tokens
from georvq.data.store import ChipStore
from georvq.factory import from_checkpoint
from georvq.metrics import centered_cos
from georvq.tessera import release as rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--fit-windows", type=int, default=512)
    ap.add_argument("--eval-windows", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--notes", default="")
    ap.add_argument("--fp16", action="store_true",
                    help="store the weights at fp16: half the file, and "
                         "bit-identical arithmetic, because load_state_dict "
                         "upcasts into the fp32 model. Measured cost to "
                         "full-depth cosine: -8.8e-07. Off by default so the "
                         "reference release is exactly what was validated.")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    model = from_checkpoint(args.ckpt, dev)
    cfg = model.full_config
    W, src = cfg.model.window, model.source
    step = torch.load(args.ckpt, map_location="cpu",
                      weights_only=False).get("step", 0)

    store = ChipStore(cfg.data.store_dir)
    _, va = store.split(cfg.data.val_frac, cfg.data.split_seed)
    tok = fixed_val_tokens(store, va, token=W,
                           max_tokens=args.fit_windows + args.eval_windows)
    fit, ev = tok[:args.fit_windows], tok[args.fit_windows:]
    print(f"fit on {fit.shape[0]} windows, evaluate on {ev.shape[0]}")

    # ---- gather what the shipped weights actually produce ----
    from georvq.inference.api import Codec
    codec = Codec(model)
    t = rel.REFINEMENT.tier
    means, resid = [], []
    for i in range(0, fit.shape[0], args.chunk):
        x, valid = src.prep(fit[i:i + args.chunk], dev)
        means.append(model.patch_mean(x, valid).cpu().numpy())
        pk = codec.encode(x, valid=valid, refine_tiers=(t,))
        resid.append(np.stack([p.refinements[t] for p in pk]))
    means = np.concatenate(means)
    resid = np.concatenate(resid)

    meta = rel.build_bundle(args.out, model, cfg, means, resid, step=step,
                            fp16=args.fp16,
                            notes=args.notes or f"{cfg.log.run_tag} step {step}")
    print(json.dumps(meta, indent=1))

    # ---- reload from disk and measure the ladder ----
    R = rel.load(args.out, dev)
    pts = [("20-0-0", [20, 0, 0]), ("20-20-0", [20, 20, 0]),
           ("20-20-1", [20, 20, 1]), ("20-20-5", [20, 20, 5]),
           ("20-20-13", [20, 20, 13]), ("20-20-20", [20, 20, 20])]
    acc = {k: [0.0, 0.0, 0] for k, _ in pts}
    acc["20-20-20 +refine"] = [0.0, 0.0, 0]
    acc["ceiling (float mean)"] = [0.0, 0.0, 0]
    acc["20-20-20 (float mean)"] = [0.0, 0.0, 0]

    def add(k, o, x, v):
        n = int(v.sum())
        a = acc[k]
        a[0] += float(src.recon_stats(o, x, v)[1]["cos"]) * n
        a[1] += float(centered_cos(o, x, v)) * n
        a[2] += n

    full = [sp.depth for sp in R.specs]
    for i in range(0, ev.shape[0], args.chunk):
        x, valid = src.prep(ev[i:i + args.chunk], dev)
        pk = R.encode(x, valid=valid, refine=False)
        for name, d in pts:
            add(name, R.decode(pk, depths=d, valid=valid), x, valid)
        pkr = R.encode(x, valid=valid, refine=True)
        add("20-20-20 +refine", R.decode(pkr, depths=full, valid=valid),
            x, valid)
        add("ceiling (float mean)", model.eval_ceiling(x, valid=valid),
            x, valid)
        pf = codec.encode(x, valid=valid)          # exact float mean
        add("20-20-20 (float mean)", codec.decode(pf, depths=full,
                                                  valid=valid), x, valid)

    print(f"\n{'point':>22} {'b/px':>8} {'cos':>9} {'ccos':>9}")
    for name, d in pts:
        a = acc[name]
        print(f"{name:>22} {R.bpp(d):8.3f} {a[0]/a[2]:9.5f} {a[1]/a[2]:9.5f}")
    a = acc["20-20-20 +refine"]
    print(f"{'20-20-20 +refine':>22} {R.bpp(full, refine=True):8.3f} "
          f"{a[0]/a[2]:9.5f} {a[1]/a[2]:9.5f}")
    for name in ("20-20-20 (float mean)", "ceiling (float mean)"):
        a = acc[name]
        print(f"{name:>22} {'-':>8} {a[0]/a[2]:9.5f} {a[1]/a[2]:9.5f}")

    q = acc["20-20-20"]
    f_ = acc["20-20-20 (float mean)"]
    print(f"\nmean channel at {R.mean_channel.bits} bits/band costs "
          f"{q[0]/q[2] - f_[0]/f_[2]:+.5f} cos vs the exact float mean "
          f"({R.side_bits:.0f} bits/window = {R.side_bits/W**2:.4f} b/px)")


if __name__ == "__main__":
    main()
