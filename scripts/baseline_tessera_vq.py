"""tessera-vq baselines on Tessera chips: per-chip k-means VQ and 2-stage RVQ
(github.com/sk818/tessera-vq, `pip install tessera-vq blockwise-kmeans@git`).

Fair protocol, mirroring baseline_classical.py: eval on held-out val chips
(same geographic split as training); distortion = mean per-pixel cosine in
the unit-norm model space (the metric the neural codecs report); rate =
actual payload bits/pixel — per-pixel indices at log2(k) bits plus the
per-chip codebook served as int8 (tessera-vq's wire format), amortized over
the chip. Codebooks are int8-round-tripped before reconstruction, exactly
what the tessera-vq server ships. Results -> json points for the R-D
overlay (eval.rd_refs: "<out>|tessera_vq|tessera-vq VQ").
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from georvq.data.store import ChipStore, geographic_split

VQ_KS = [4, 16, 64, 256, 1024, 4096]
# (20, 256) is the VQTessera client's default serving configuration at
# t=512 — "tessera-vq as deployed"
RVQ_KS = [(16, 16), (20, 256), (64, 64), (256, 256), (1024, 1024)]


def unit_field(chip):
    """int8 (128, px, px) -> (n_px, 128) float32 unit vectors."""
    x = chip.reshape(128, -1).T.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def full_valid_chips(store, rows, n, seed):
    out, order = [], np.random.default_rng(seed).permutation(len(rows))
    for i in order[:min(len(rows), 40 * n)]:
        r = rows.iloc[int(i)]
        a = np.asarray(store.open_shard(int(r.shard))[int(r.slot)])
        if (a != 0).any(axis=0).all():        # no all-zero (nodata) pixels
            out.append(a)
        if len(out) >= n:
            break
    return out


def cosine(x, recon):
    """Mean per-pixel cosine of unit-norm x vs (re-normalized) recon."""
    n = np.linalg.norm(recon, axis=1, keepdims=True)
    return float((x * (recon / np.where(n > 0, n, 1.0))).sum(1).mean())


def bits_per_px(ks, n_px):
    """log2(k) index bits per pixel per stage + int8 codebook amortized."""
    return sum(np.log2(k) + k * 128 * 8 / n_px for k in ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips-dir", default="/data/tessera-samples/chips_512")
    ap.add_argument("--out",
                    default="/data/tessera-samples/baselines/tessera_vq.json")
    ap.add_argument("--n-chips", type=int, default=12)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-chips", action="store_true",
                    help="skip the val split (tiny/smoke stores)")
    args = ap.parse_args()

    from blockwise_kmeans import assign_blocked, kmeans_fit
    from tessera_vq.codebook_codec import roundtrip_uint8
    from tessera_vq.rvq_large import rvq_reconstruct_flat

    store = ChipStore(args.chips_dir)
    rows = (store.index if args.all_chips
            else geographic_split(store.index, args.val_frac,
                                  seed=args.split_seed)[1])
    chips = full_valid_chips(store, rows, args.n_chips, args.seed)
    print(f"{len(chips)} fully-valid chips from {len(rows)} "
          f"{'store' if args.all_chips else 'val'} rows", flush=True)
    assert chips, "no fully-valid chips found"

    vq_pts = {k: [] for k in VQ_KS}
    rvq_pts = {ks: [] for ks in RVQ_KS}
    for ci, chip in enumerate(chips):
        x = unit_field(chip)
        for k in VQ_KS:
            cb = roundtrip_uint8(kmeans_fit(x, k, seed=args.seed))
            cos = cosine(x, cb[assign_blocked(x, cb)])
            vq_pts[k].append((bits_per_px([k], len(x)), cos))
        for k1, k2 in RVQ_KS:
            recon = rvq_reconstruct_flat(x, k1, k2, seed=args.seed,
                                         quantize_codebooks=True)
            rvq_pts[(k1, k2)].append((bits_per_px([k1, k2], len(x)),
                                      cosine(x, recon)))
        print(f"  chip {ci + 1}/{len(chips)} done", flush=True)

    mean = lambda pts: [float(np.mean([p[0] for p in pts])),  # noqa: E731
                        float(np.mean([p[1] for p in pts]))]
    result = {
        "tessera_vq": [mean(v) for v in vq_pts.values()],
        "tessera_vq_rvq": [mean(v) for v in rvq_pts.values()],
        "meta": {"n_chips": len(chips), "chips_dir": args.chips_dir,
                 "protocol": "unit-norm cosine on fully-valid val chips; "
                             "bits = log2(k) idx/px + int8 codebook/chip",
                 "vq_ks": VQ_KS, "rvq_ks": RVQ_KS},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    for name in ("tessera_vq", "tessera_vq_rvq"):
        print(name, [(round(b, 2), round(c, 4)) for b, c in result[name]])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
