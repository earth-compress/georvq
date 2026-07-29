"""Train lightweight decoder variants against the frozen release bitstream.

The heavy decoder (attn 384x12, ~30M params) serves all three tiers at
full 64px resolution. This trains the cheap end of a decoder family that
consumes a PREFIX of the same blobs (bitstream-unchanged, decoder-side
only — docs/BITSTREAM.md):

  light   tier 0 only        head at  4px -> bilinear x16 -> 64px
  medium  tiers 0+1          head at 16px -> bilinear x4  -> 64px

Each dropped tier drops the head resolution 4x; the pixel result comes
from interpolation + per-pixel renorm. Inputs come from a pre-encoded
window-index cache (encoded with the released weights), so the encoder
never runs — codebook lookup + tiny decoder per step.

Usage:
  python scripts/train_lite_decoder.py --variant light --gpu 1
  python scripts/train_lite_decoder.py --variant medium --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from georvq.factory import from_checkpoint
from georvq.metrics import masked_recon_stats
from georvq.model.codec import sample_monotone_depths
from georvq.model.decoder import DenseHead, PyramidAttnDecoder

WIN = 64

VARIANTS = {
    "light":  {"tiers": 1, "dim": 128, "depth": 2, "heads": 4,
               "mins": [6], "canon": [[6], [7], [8]]},
    "medium": {"tiers": 2, "dim": 256, "depth": 4, "heads": 8,
               "mins": [6, 1], "canon": [[8, 1], [8, 4], [8, 8]]},
}


class LiteDecoder(torch.nn.Module):
    """Attention decoder over the first n tiers, low-res head, bilinear
    up to the window, per-pixel renorm (AE fields are unit-norm)."""

    def __init__(self, specs, dim, depth, heads, head_feat=64, bands=64):
        super().__init__()
        self.out_res = WIN // 4 ** (3 - len(specs))
        fine = specs[-1]
        head = DenseHead(dim, grid=fine.grid, p=self.out_res // fine.grid,
                         bands=bands, head_feat=head_feat, normalize=False)
        self.dec = PyramidAttnDecoder(specs, head=head, dim=dim, depth=depth,
                                      heads=heads, cls_bridge=False,
                                      ff_ratio=2.0)

    def forward(self, zqs, Ls):
        low = self.dec(zqs, Ls)                      # (B, C, r, r)
        up = F.interpolate(low, size=WIN, mode="bilinear",
                           align_corners=False)
        return F.normalize(up, dim=1, eps=1e-6)


class WindowCache:
    """Pre-encoded windows: idx memmaps (chips, 16, 16, cells, depth) +
    raw int8 targets gathered from the chip store."""

    def __init__(self, cache_dir, n_tiers):
        self.man = json.load(open(f"{cache_dir}/manifest.json"))
        self.idx = [np.load(f"{cache_dir}/idx{t}.npy", mmap_mode="r")
                    for t in range(n_tiers)]
        prov = pd.read_parquet(f"{cache_dir}/index.parquet")
        self.shard = prov.shard.to_numpy()
        self.slot = prov.slot.to_numpy()
        self.chips_dir = self.man["chips_dir"]
        self.N, self.G = self.idx[0].shape[0], self.man["win_grid"]
        self._shards = {}

    def _chip(self, sh):
        if sh not in self._shards:
            self._shards[sh] = np.load(
                f"{self.chips_dir}/shard_{sh:04d}.npy", mmap_mode="r")
        return self._shards[sh]

    def read(self, rng, n):
        """n random windows in grid-row strips of 16 (one contiguous-ish
        4 MiB read per strip instead of 16 scattered window reads — the
        shard layout is band-major, so per-window reads are 64-byte
        scatter and IO-starve the GPU).
        -> (per-tier idx int64 (n, cells, depth), targets int8)."""
        assert n % self.G == 0, f"batch must be a multiple of {self.G}"
        strips = n // self.G
        ci = rng.integers(0, self.N, strips)
        wr = rng.integers(0, self.G, strips)
        idx = [np.concatenate([np.asarray(self.idx[t][c, r])
                               for c, r in zip(ci, wr)]).astype(np.int64)
               for t in range(len(self.idx))]
        tg = np.concatenate([
            np.asarray(self._chip(self.shard[c])
                       [self.slot[c], :, r * WIN:(r + 1) * WIN, :])
            .reshape(64, WIN, self.G, WIN).transpose(2, 0, 1, 3)
            for c, r in zip(ci, wr)])
        return idx, tg


class Prefetcher:
    def __init__(self, cache, batch, seed, threads=10, depth=4):
        self.q = queue.Queue(maxsize=depth)
        for t in range(threads):
            rng = np.random.default_rng(seed * 977 + t)
            threading.Thread(target=self._work, args=(cache, batch, rng),
                             daemon=True).start()

    def _work(self, cache, batch, rng):
        while True:
            self.q.put(cache.read(rng, batch))


def prep(tg_np, dev):
    s = torch.from_numpy(tg_np).to(dev, non_blocking=True)
    valid = s[:, 0] != -128
    x = s.float() / 127.5
    x = x.square() * torch.sign(x) * valid.unsqueeze(1)
    return F.normalize(x, dim=1, eps=1e-6), valid


def zq_lookup(books, idx, Ls, dev):
    """Frozen-codebook reconstruction of decoder inputs at depths Ls."""
    zqs = []
    for t, per_level in enumerate(books):
        it = torch.from_numpy(idx[t]).to(dev, non_blocking=True)
        depth = it.shape[-1]
        per = torch.stack([per_level[j][it[..., j]]
                           for j in range(depth)], dim=2)
        m = (torch.arange(depth, device=dev)[None, None, :, None]
             < Ls[:, t][:, None, None, None]).float()
        zqs.append((per * m).sum(2))
    return zqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--bundle",
                    default="checkpoints/ae_release_beta/bundle")
    ap.add_argument("--cache", default="/mnt/raid/data/ae_deblock_cache")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--full-frac", type=float, default=0.1)
    ap.add_argument("--val-windows", type=int, default=2048)
    ap.add_argument("--val-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=2500)
    ap.add_argument("--read-workers", type=int, default=10)
    ap.add_argument("--run-tag", default=None)
    args = ap.parse_args()
    v = VARIANTS[args.variant]
    tag = args.run_tag or f"ae_lite_{args.variant}"
    outdir = f"checkpoints/{tag}"
    os.makedirs(outdir, exist_ok=True)
    dev = torch.device(f"cuda:{args.gpu}")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    model = from_checkpoint(os.path.join(args.bundle, "model.pt"), dev)
    from georvq.inference.release import bitstream_id
    bid = bitstream_id(model)
    specs = model.specs[:v["tiers"]]
    books = [[model.rvqs[t].levels[j].codebook.weight.detach()
              for j in range(specs[t].depth)] for t in range(v["tiers"])]
    del model.decoder, model.tokenizer   # only the codebooks are needed
    torch.cuda.empty_cache()

    net = LiteDecoder(specs, v["dim"], v["depth"], v["heads"]).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"{args.variant}: {v['tiers']} tiers -> {net.out_res}px head, "
          f"{n_par/1e6:.2f}M params | bitstream {bid}", flush=True)

    tr = WindowCache(f"{args.cache}/train", v["tiers"])
    va = WindowCache(f"{args.cache}/val", v["tiers"])
    pf = Prefetcher(tr, args.batch, seed=1, threads=args.read_workers)

    # fixed val windows
    vrng = np.random.default_rng(7)
    vidx, vtg = va.read(vrng, args.val_windows)
    vx, vvalid = prep(vtg, dev)

    @torch.no_grad()
    def validate():
        net.eval()
        out = {}
        for tup in v["canon"]:
            Ls = torch.tensor([tup] * args.val_windows, device=dev)
            cs, B = 0.0, 256
            for i in range(0, args.val_windows, B):
                zqs = zq_lookup(books, [a[i:i + B] for a in vidx],
                                Ls[i:i + B], dev)
                y = net(zqs, Ls[i:i + B])
                cs += float((y * vx[i:i + B]).sum(1)[vvalid[i:i + B]].sum())
            out["-".join(map(str, tup))] = cs / float(vvalid.sum())
        net.train()
        return out

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup)
        * 0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps))))

    best, t0, run_mse = -1.0, time.time(), None
    for step in range(1, args.steps + 1):
        idx, tg = pf.q.get()
        x, valid = prep(tg, dev)
        Ls = sample_monotone_depths(args.batch, specs, dev, mins=v["mins"],
                                    full_frac=args.full_frac)
        with torch.no_grad():
            zqs = zq_lookup(books, idx, Ls, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(zqs, Ls)
        mse, _ = masked_recon_stats(out.float(), x, valid)
        opt.zero_grad(set_to_none=True)
        mse.backward()
        opt.step(); sched.step()
        run_mse = float(mse) if run_mse is None \
            else 0.98 * run_mse + 0.02 * float(mse)

        if step % args.val_interval == 0 or step == args.steps:
            vs = validate()
            top = vs["-".join(map(str, v["canon"][-1]))]
            line = " ".join(f"{k}={c:.5f}" for k, c in vs.items())
            print(f"step {step}: train_mse {run_mse:.5f} | val cos {line} | "
                  f"{(time.time()-t0)/step*1000:.0f} ms/step", flush=True)
            if top > best:
                best = top
                torch.save({"model": net.state_dict(), "step": step,
                            "variant": args.variant, "cfg": v,
                            "bitstream_id": bid, "args": vars(args),
                            "val": vs},
                           f"{outdir}/model_best.pt")
        if step % args.save_interval == 0:
            torch.save({"model": net.state_dict(), "step": step,
                        "variant": args.variant, "cfg": v,
                        "bitstream_id": bid, "args": vars(args)},
                       f"{outdir}/model.pt")
    print(f"done: best val cos {best:.5f} -> {outdir}/model_best.pt",
          flush=True)


if __name__ == "__main__":
    main()
