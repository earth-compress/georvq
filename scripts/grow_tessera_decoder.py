"""Grow the released Tessera decoder: deeper and wider, same function.

Takes the shipped bundle, builds a decoder with more blocks and wider
MLPs/head, and initialises it so it reproduces the release exactly (see
`georvq.train.grow`). The encoder, codebooks and therefore the BITSTREAM
are untouched -- `bitstream_id` must be unchanged, and this script asserts
it. Only `decoder_id` moves.

    python scripts/grow_tessera_decoder.py \
        --bundle /data/georvq-releases/tessera-beta-v1/bundle \
        --add-blocks 8 --ff-ratio 3.0 --head-feat 192 \
        --out checkpoints/tessera_grown_init.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import torch

from georvq.factory import build_model
from georvq.repro import save_checkpoint
from georvq.train.grow import grow_decoder, verify_growth


def load_bundle(bundle, device):
    from georvq.inference.release import load_release
    out = load_release(bundle, device=device)
    codec = out[0] if isinstance(out, tuple) else out
    return getattr(codec, "model", codec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle",
                    default="/data/georvq-releases/tessera-beta-v1/bundle")
    ap.add_argument("--add-blocks", type=int, default=8)
    ap.add_argument("--ff-ratio", type=float, default=3.0)
    ap.add_argument("--head-feat", type=int, default=192)
    ap.add_argument("--out", default="checkpoints/tessera_grown/init.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--atol", type=float, default=2e-3)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and
                       torch.cuda.is_available() else "cpu")
    src = load_bundle(args.bundle, dev).eval()
    scfg = src.full_config
    print(f"release: decoder dim={scfg.model.decoder.dim} "
          f"depth={scfg.model.decoder.depth} "
          f"head_feat={scfg.model.decoder.head_feat} "
          f"ff_ratio={scfg.model.ff_ratio}")

    cfg = copy.deepcopy(scfg)
    d = cfg.model.decoder
    d.depth = scfg.model.decoder.depth + args.add_blocks
    d.head_feat = args.head_feat
    d.ff_ratio = args.ff_ratio          # DECODER only; encoder keeps its own
    print(f"grown:   decoder dim={d.dim} depth={d.depth} "
          f"head_feat={d.head_feat} ff_ratio={d.ff_ratio}")

    dst = build_model(cfg, dev).eval()
    n_s = sum(p.numel() for p in src.decoder.parameters())
    n_d = sum(p.numel() for p in dst.decoder.parameters())
    print(f"decoder params {n_s/1e6:.1f}M -> {n_d/1e6:.1f}M "
          f"({n_d/n_s:.2f}x); total {sum(p.numel() for p in dst.parameters())/1e6:.1f}M")

    grow_decoder(dst, src.state_dict())

    # --- the bitstream must not have moved -------------------------------
    from georvq.inference.release import bitstream_id
    bid_s, bid_d = bitstream_id(src), bitstream_id(dst)
    assert bid_s == bid_d, (
        f"bitstream_id changed {bid_s} -> {bid_d}: the surgery touched the "
        f"encoder or codebooks, so cached latents and existing packets "
        f"would be invalid")
    print(f"bitstream_id unchanged: {bid_s}")

    # --- and the function must not have moved ----------------------------
    torch.manual_seed(0)
    raw = torch.randint(-127, 128,
                        (args.windows, 130, cfg.model.window, cfg.model.window),
                        dtype=torch.int8, device=dev)
    x, valid = src.source.prep(raw, dev)
    r = verify_growth(src, dst, x, valid=valid, atol=args.atol)
    print(f"function check: max|diff| {r['max_abs_diff']:.3e} "
          f"(output scale {r['output_scale']:.3f}, rel {r['rel']:.3e})")
    if not r["ok"]:
        raise SystemExit(
            f"grown decoder does NOT reproduce the release "
            f"(max|diff| {r['max_abs_diff']:.3e} > {args.atol}). Refusing to "
            f"write: the experiment would confound added capacity with lost "
            f"weights.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_checkpoint(args.out, dst, cfg, 0,
                    {"grown_from": args.bundle, "bitstream_id": bid_s,
                     "max_abs_diff_vs_release": r["max_abs_diff"]})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
