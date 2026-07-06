"""Classical lossy baselines on AlphaEarth chips: PCA+JPEG2000, HEVC-spectral.

Fair protocol: PCA fit on train chips; eval on held-out val chips; rate =
actual file bytes / pixels; distortion = masked cosine in dequantized
unit-norm space (same metric as the neural codecs). Results -> json points
for the R-D overlay.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from georvq.data.store import geographic_split
from georvq.sources.ae import NODATA, dequant


def load_chip(chips_dir, rows, i):
    r = rows.iloc[i]
    sh = np.load(f"{chips_dir}/shard_{int(r['shard']):04d}.npy", mmap_mode="r")
    return np.asarray(sh[int(r["slot"])])


def full_valid_chips(chips_dir, rows, n, seed):
    out, tried = [], 0
    order = np.random.default_rng(seed).permutation(len(rows))
    for i in order:
        a = load_chip(chips_dir, rows, int(i))
        if (a[0] != NODATA).all():
            out.append(a)
        if len(out) >= n:
            break
        tried += 1
        if tried > 400:
            break
    return out


def deq(a):  # int8 chip -> float (64, H, W) unit-norm
    return dequant(torch.from_numpy(a.copy())).numpy()


def cosine(x, y):
    return float((x * y).sum(0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips-dir", default="/data/AE-samples/chips_1024")
    ap.add_argument("--out",
                    default="plots/ae_validation/classical_baselines.json")
    args = ap.parse_args()
    chips = args.chips_dir
    index = pd.read_parquet(f"{chips}/index.parquet")
    tr, va = geographic_split(index, 0.04, seed=0)

    print("loading chips...", flush=True)
    train_chips = full_valid_chips(chips, tr, 4, 1)
    val_chips = full_valid_chips(chips, va, 8, 2)
    print(f"{len(train_chips)} train, {len(val_chips)} val chips", flush=True)

    # PCA fit (train pixels, dequant space)
    X = np.concatenate([deq(c).reshape(64, -1).T[::16] for c in train_chips])
    mu = X.mean(0)
    U = np.linalg.svd((X - mu).T @ (X - mu), hermitian=True)[0]  # (64, 64)
    print("PCA fit done", flush=True)

    points = {"pca_j2k": [], "hevc": [], "pca_j2k_64px": []}

    # ---- PCA + JPEG2000 (equal-PSNR allocation across components) ----
    for psnr in (34, 38, 42, 46, 50, 54):
        tot_bits, coss = 0, []
        for chip in val_chips:
            x = deq(chip)
            comps = (U.T @ (x.reshape(64, -1) - mu[:, None])).reshape(64, 1024,
                                                                      1024)
            S = 32000.0 / max(np.percentile(np.abs(comps), 99.99), 1e-6)
            rec = np.empty_like(comps)
            with tempfile.TemporaryDirectory() as td:
                for ci in range(64):
                    q = (np.clip(comps[ci] * S, -32000, 32000)
                         + 32768).astype(np.uint16)
                    im = Image.fromarray(q)          # mode I;16
                    fp = os.path.join(td, f"{ci}.j2k")
                    im.save(fp, format="JPEG2000", irreversible=True,
                            quality_mode="dB", quality_layers=[psnr])
                    tot_bits += os.path.getsize(fp) * 8
                    rec[ci] = (np.asarray(Image.open(fp)).astype(np.float32)
                               - 32768.0) / S
            y = (U @ rec.reshape(64, -1) + mu[:, None]).reshape(64, 1024, 1024)
            y = y / np.linalg.norm(y, axis=0, keepdims=True).clip(1e-8)
            coss.append(cosine(x, y))
        bpp = tot_bits / (len(val_chips) * 1024 * 1024)
        points["pca_j2k"].append([bpp, float(np.mean(coss))])
        print(f"pca_j2k psnr{psnr}: {bpp:.2f} b/px cos {np.mean(coss):.4f}",
              flush=True)

    # ---- PCA+J2K on independent 64px tiles (context-matched to our codec) ----
    for psnr in (34, 38, 42, 46):
        tot_bits, coss = 0, []
        for chip in val_chips[:4]:
            x = deq(chip)
            comps = (U.T @ (x.reshape(64, -1) - mu[:, None])).reshape(64, 1024,
                                                                      1024)
            S = 32000.0 / max(np.percentile(np.abs(comps), 99.99), 1e-6)
            rec = np.empty_like(comps)
            with tempfile.TemporaryDirectory() as td:
                for ty in range(0, 1024, 64):
                    for tx in range(0, 1024, 64):
                        for ci in range(64):
                            q = (np.clip(comps[ci, ty:ty+64, tx:tx+64] * S,
                                         -32000, 32000) + 32768).astype(np.uint16)
                            fp = os.path.join(td, "t.j2k")
                            Image.fromarray(q).save(
                                fp, format="JPEG2000", irreversible=True,
                                quality_mode="dB", quality_layers=[psnr])
                            tot_bits += os.path.getsize(fp) * 8
                            rec[ci, ty:ty+64, tx:tx+64] = \
                                (np.asarray(Image.open(fp)).astype(np.float32)
                                 - 32768.0) / S
            y = (U @ rec.reshape(64, -1) + mu[:, None]).reshape(64, 1024, 1024)
            y = y / np.linalg.norm(y, axis=0, keepdims=True).clip(1e-8)
            coss.append(cosine(x, y))
        bpp = tot_bits / (4 * 1024 * 1024)
        points["pca_j2k_64px"].append([bpp, float(np.mean(coss))])
        print(f"pca_j2k_64px psnr{psnr}: {bpp:.2f} b/px cos {np.mean(coss):.4f}",
              flush=True)

    # ---- HEVC: 64 bands as 8-bit gray video ----
    for crf in (10, 16, 22, 28, 34, 40):
        tot_bits, coss = 0, []
        for chip in val_chips:
            vid = (chip.astype(np.int16) + 128).astype(np.uint8)  # (64,1024,1024)
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "b.hevc")
                p = subprocess.run(
                    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                     "-s", "1024x1024", "-r", "25", "-i", "-", "-c:v", "libx265",
                     "-crf", str(crf), "-preset", "medium",
                     "-x265-params", "log-level=error", f],
                    input=vid.tobytes(), capture_output=True)
                tot_bits += os.path.getsize(f) * 8
                q = subprocess.run(
                    ["ffmpeg", "-y", "-i", f, "-f", "rawvideo", "-pix_fmt",
                     "gray", "-"], capture_output=True)
                out = np.frombuffer(q.stdout, dtype=np.uint8)
                out = out.reshape(64, 1024, 1024)
            rec = out.astype(np.int16) - 128
            x = deq(chip)
            y = deq(rec.astype(np.int8))
            y = y / np.linalg.norm(y, axis=0, keepdims=True).clip(1e-8)
            coss.append(cosine(x, y))
        bpp = tot_bits / (len(val_chips) * 1024 * 1024)
        points["hevc"].append([bpp, float(np.mean(coss))])
        print(f"hevc crf{crf}: {bpp:.2f} b/px cos {np.mean(coss):.4f}",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(points, f, indent=1)
    print(f"\n-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
