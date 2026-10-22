"""Train the seam deblocker (georvq.sources.s2_deblock) for a frozen S2 codec.

The protocol is that of the ring-space beta_v4 deblocker
(scripts/train_deblocker.py there), moved onto the codec's own window grid:

- data: 252 px crops (7 x 7 windows of 36 px, the frame of RECIPE with
  its halo) from the training split of the live Major-TOM reader, at
  origins on the 6 px grid, through the trainer's resident scene pool.
  Every window of a crop is coded independently by the frozen codec
  (encoder, books and decoder) and decoded with the stats as the mean
  channel decodes them, so window-mean steps from stats coding are in
  the input as they will be in a release;
- depth: the coarse tier in full, the fine tier drawn from 0..depth,
  two distinct depths per batch with each crop assigned one of them;
- loss: MSE of the zero-mean residual in the frame-standardised space,
  over the whole crop, summed over the three grids (`residual_loss`);
- optimiser: Adam at 3e-4 without weight decay, 200 linear warmup steps
  then constant, grad-norm clip 1.0, batch 128 crops, 20,000 steps,
  float32 for the deblocker. The codec runs with TF32 matmuls unless
  --no-tf32, which can flip a near-tied code here and there in the
  training inputs and does not touch the evaluation decodes.

The release's deblocker was produced by

    python scripts/train_deblocker_s2.py \\
        --ckpt checkpoints/beta_arch_w36/frozen_42500.pt \\
        --channel <ch176.npz> --mean-codec <release-format mean_codec.py> \\
        --out-dir checkpoints/deblock_w36

SIGUSR1 or SIGTERM saves `state.pt` and exits; `--resume` continues.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from georvq.data.live_s2 import LiveS2Store
from georvq.data.pool import ChipPool, fixed_val_tokens
from georvq.factory import from_checkpoint
from georvq.inference.release import bitstream_id, decoder_id
from georvq.repro import git_state, load_checkpoint
from georvq.sources.s2 import GROUPS, group_valid, pixel_valid
from georvq.sources.s2_deblock import (RECIPE, SCALES, S2BoundaryDeblocker,
                                       from_windows, residual_loss,
                                       to_windows)

_stop = False


def _on_signal(signum, frame):
    global _stop
    _stop = True


def load_mean_codec(path):
    """The mean_codec module to read the channel with: this checkout's,
    or one loaded from a file (a branch whose channel format is newer)."""
    if not path:
        from georvq.inference import mean_codec
        return mean_codec
    spec = importlib.util.spec_from_file_location("mean_codec_ext", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TorchRVQPCA:
    """`RVQPCAMeanChannel.round_trip` in torch on the GPU, for training
    inputs only. The arithmetic is the channel's, in float32; a near-tie
    in the RVQ search may pick another (equally valid) code than numpy
    would, which is why the evaluation tools use the channel itself."""

    def __init__(self, ch, device):
        t = lambda a: torch.as_tensor(np.asarray(a), device=device)  # noqa
        self.books = t(ch.books).float()
        self.bn = (self.books ** 2).sum(-1)
        self.mu0, self.V, self.s = t(ch.mu0), t(ch.V), t(ch.s)
        self.bits = [int(b) for b in ch.bits]
        self.clip = t(ch.clip)
        self.pre_scale = None if ch.pre_scale is None else t(ch.pre_scale)
        self.pre_log = None if ch.pre_log is None else t(ch.pre_log).bool()

    @torch.no_grad()
    def round_trip(self, x):
        r = x.float().clone()
        if self.pre_log is not None:
            r[:, self.pre_log] = r[:, self.pre_log].log()
        if self.pre_scale is not None:
            r = r * self.pre_scale
        out = torch.zeros_like(r)
        for C, cn in zip(self.books, self.bn):
            d = (r * r).sum(1, keepdim=True) - 2 * r @ C.T + cn[None]
            q = C[d.argmin(1)]
            r, out = r - q, out + q
        c = (r - self.mu0) @ self.V
        ch = torch.zeros_like(c)
        for k, b in enumerate(self.bits):
            if b:
                n = 1 << b
                lo, w = -self.clip[k] * self.s[k], 2 * self.clip[k] * self.s[k]
                q = ((c[:, k] - lo) / w * n).floor().clamp(0, n - 1)
                ch[:, k] = lo + (q + 0.5) * w / n
        y = out + self.mu0 + ch @ self.V.T
        if self.pre_scale is not None:
            y = y / self.pre_scale
        if self.pre_log is not None:
            y[:, self.pre_log] = y[:, self.pre_log].exp()
        return y


class Recon:
    """Frozen codec + stats channel: raw uint16 crops -> decoded crops."""

    def __init__(self, model, channel, window):
        self.model, self.channel, self.window = model, channel, window
        self.src = model.source
        self.norm = self.src.normalizer(model.full_config.model)
        self.depths = [sp.depth for sp in model.specs]

    @torch.no_grad()
    def __call__(self, raw, fine):
        """raw {g: (B, C, S/r, S/r)} uint16/int; fine (B,) fine-tier depth.
        Returns (recon, truth, valid) as {g: float (B, C, ...)} in DN / 1e4
        and {g: (B, 1, ...) bool}."""
        B, _, S, _ = raw["10m"].shape
        n = S // self.window
        win = to_windows({g: raw[g].to(torch.int32) for g in GROUPS},
                         self.window)
        x, valid = self.src.prep(win, raw["10m"].device)
        st = self.src.last_stats()
        vec = self.norm.stats_vector(st).float()
        if isinstance(self.channel, TorchRVQPCA):
            dec = self.channel.round_trip(vec)
        else:
            dec = torch.from_numpy(self.channel.round_trip(
                vec.cpu().numpy())).to(x.device)
        sd, i, nb = {}, 0, vec.shape[1] // 2
        for g in GROUPS:
            k = GROUPS[g][0]
            sd[g] = (dec[:, i:i + k, None, None], dec[:, nb + i:nb + i + k,
                                                      None, None])
            i += k
        m = self.model
        _, cums = m.level_cumsums(x, valid=valid)
        Lw = torch.tensor(self.depths, device=x.device)[None] \
            .repeat(B * n * n, 1)
        Lw[:, -1] = fine.repeat_interleave(n * n)
        rows = torch.arange(Lw.shape[0], device=x.device)
        zqs = [c[rows, :, Lw[:, j]] for j, c in enumerate(cums)]
        o = m.decode(zqs, Lw, m._stats_of(x, valid), valid)
        rec = from_windows({g: v.float() for g, v in
                            self.norm.denormalize(o, sd).items()}, n, n,
                           self.window)
        truth = {g: raw[g].float() / 1e4 for g in GROUPS}
        bv = {g: raw[g] != 0 for g in GROUPS}
        pv = pixel_valid(bv)
        return rec, truth, {g: group_valid(pv, g) for g in GROUPS}


def seam_sums(y, window):
    """Sums of |neighbour difference| at window seams and in window
    interiors of the 10 m planes, over both axes: (seam, n_seam, int,
    n_int)."""
    t = y["10m"]
    out = np.zeros(4)
    for d in (t[..., 1:] - t[..., :-1], (t[..., 1:, :] - t[..., :-1, :])
              .transpose(-1, -2)):
        q = (torch.arange(1, d.shape[-1] + 1, device=d.device) % window) == 0
        a = d.abs()
        out += [float(a[..., q].sum()), a[..., q].numel(),
                float(a[..., ~q].sum()), a[..., ~q].numel()]
    return out


@torch.no_grad()
def validate(net, recon, val, device, depth, batch=32):
    """Per fine depth: SSE in DN raw / deblocked per grid over valid
    pixels, and the 10 m seam/interior mean |grad| raw / deblocked."""
    net.eval()
    res = {}
    for L in range(depth + 1):
        sse = {k: np.zeros(3) for k in ("raw", "deb")}
        cnt = np.zeros(3)
        seam = {k: np.zeros(4) for k in ("raw", "deb")}
        for a in range(0, val["10m"].shape[0], batch):
            raw = {g: val[g][a:a + batch].to(device) for g in GROUPS}
            B = raw["10m"].shape[0]
            fine = torch.full((B,), L, device=device)
            rec, truth, valid = recon(raw, fine)
            Ls = torch.tensor([recon.depths[0], L], device=device)[None] \
                .repeat(B, 1)
            deb = net(rec, Ls)
            for k, y in (("raw", rec), ("deb", deb)):
                for j, g in enumerate(GROUPS):
                    e = ((y[g] - truth[g]) * 1e4) ** 2 * valid[g]
                    sse[k][j] += float(e.sum())
                    if k == "raw":
                        cnt[j] += float(valid[g].expand_as(e).sum())
                seam[k] += seam_sums(y, recon.window)
        r = {}
        for k in ("raw", "deb"):
            r[f"rmse_{k}"] = np.sqrt(sse[k] / np.maximum(cnt, 1)).tolist()
            s = seam[k]
            r[f"seam_{k}"] = float((s[0] / s[1]) / (s[2] / s[3]))
        r["mse_ratio"] = (sse["deb"] / np.maximum(sse["raw"], 1e-12)).tolist()
        res[L] = r
    net.train()
    return res


def save_release_file(path, net, args, step, model, channel, mc_path):
    """The deblocker in georvq.model.deblock.load_deblocker's layout."""
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({
        "arch": {"width": args.width, "cond_dim": args.cond_dim,
                 "depth": net.depth, "window": net.window,
                 "cell": net.cell},
        "model": sd,
        "recipe": dict(RECIPE),
        "trained_against": {
            "bitstream_id": bitstream_id(model, extra=channel.id_state()),
            "decoder_id": decoder_id(model)},
        "provenance": {"step": int(step), "codec": os.path.abspath(args.ckpt),
                       "channel": os.path.abspath(args.channel),
                       "mean_codec": mc_path or "georvq.inference.mean_codec",
                       "args": {k: v for k, v in vars(args).items()
                                if isinstance(v, (int, float, str, bool))},
                       "git": git_state()},
    }, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="frozen codec checkpoint")
    ap.add_argument("--channel", required=True, help="stats channel .npz")
    ap.add_argument("--mean-codec", default="",
                    help="mean_codec.py to read the channel with")
    ap.add_argument("--data-root", default="/data/Major-TOM/Core-S2L1C")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--cond-dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--depths-per-batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pool", type=int, default=2048)
    ap.add_argument("--pool-workers", type=int, default=8)
    ap.add_argument("--refresh-per-s", type=float, default=16.0)
    ap.add_argument("--val-chips", type=int, default=64)
    ap.add_argument("--val-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--no-tf32", action="store_true",
                    help="decode the training inputs in full float32 "
                         "(TF32 halves the codec's share of a step)")
    args = ap.parse_args()

    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32

    model = from_checkpoint(args.ckpt, device=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mc = load_mean_codec(args.mean_codec)
    channel = mc.load_channel(args.channel)
    window = model.full_config.model.window
    depth = model.specs[-1].depth
    S = RECIPE["frame"] + 2 * RECIPE["halo"]
    assert S % window == 0 and RECIPE["halo"] % window == 0
    recon = Recon(model, TorchRVQPCA(channel, device)
                  if getattr(channel, "kind", "") == "rvq-pca" else channel,
                  window)
    print(f"codec {args.ckpt}: window {window}, depths {recon.depths}; "
          f"channel {args.channel} ({channel.cost_bits()} bits); crops {S} px",
          flush=True)

    net = S2BoundaryDeblocker(width=args.width, cond_dim=args.cond_dim,
                              depth=depth, window=window).to(device)
    print(f"deblocker: {sum(p.numel() for p in net.parameters())} "
          "parameters", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if args.resume:
        ck = load_checkpoint(args.resume, map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"])
        rng = np.random.default_rng(args.seed + step)
        torch.manual_seed(args.seed + step)
        print(f"resumed at step {step}", flush=True)

    store = LiveS2Store(args.data_root)
    train_rows, val_rows = store.split(0.02, seed=0)
    vt = fixed_val_tokens(store, val_rows, token=S, per_chip=4,
                          max_chips=args.val_chips, max_tokens=4 *
                          args.val_chips)
    val = {g: vt[g].to(torch.int32) for g in GROUPS}
    print(f"data: {len(train_rows)} train / {len(val_rows)} val scenes, "
          f"{val['10m'].shape[0]} val crops", flush=True)
    pool = ChipPool(store, train_rows, args.pool, device,
                    workers=args.pool_workers,
                    refresh_chips_per_s=args.refresh_per_s,
                    seed=args.seed + step, slot_rows=store.chip_px,
                    pool_device="cpu", cycle=True, period=6, token=S)

    def save_state():
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "step": step}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        save_release_file(os.path.join(args.out_dir, "deblocker.pt"), net,
                          args, step, model, channel, args.mean_codec)

    log = open(os.path.join(args.out_dir, "val.jsonl"), "a")
    net.train()
    t0, acc, nacc = time.time(), np.zeros(4), 0
    while step < args.steps:
        raw = pool.sample_batch(args.batch, token=S, align=True)
        k = min(args.depths_per_batch, depth + 1)
        choice = rng.choice(depth + 1, size=k, replace=False)
        fine = torch.from_numpy(choice[rng.integers(0, k, args.batch)]) \
            .to(device)
        rec, truth, valid = recon(raw, fine)
        Ls = torch.stack([torch.full_like(fine, recon.depths[0]), fine], 1)
        lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        loss, parts, _ = residual_loss(net, rec, truth, Ls, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        opt.step()
        step += 1
        acc += [loss.item()] + [parts[g].item() for g in GROUPS]
        nacc += 1
        if step % args.log_interval == 0:
            a = acc / nacc
            el = time.time() - t0
            print(f"step {step} loss {a[0]:.5f} (10m {a[1]:.5f} 20m "
                  f"{a[2]:.5f} 60m {a[3]:.5f}) lr {lr:.2e} "
                  f"{el / nacc:.3f} s/step"
                  + (f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f}"
                     " GiB" if device.type == "cuda" else ""), flush=True)
            t0, acc, nacc = time.time(), np.zeros(4), 0
        if step % args.val_interval == 0 or step == args.steps:
            tv = time.time()
            r = validate(net, recon, val, device, depth)
            log.write(json.dumps({"step": step, "val": r}) + "\n")
            log.flush()
            for L, v in r.items():
                print(f"  VAL {step} 8-{L}: seam {v['seam_raw']:.3f} -> "
                      f"{v['seam_deb']:.3f}, mse ratio "
                      + " ".join(f"{x:.4f}" for x in v["mse_ratio"]),
                      flush=True)
            print(f"  val took {time.time() - tv:.0f}s", flush=True)
            t0 = time.time()
        if step % args.save_interval == 0 or step == args.steps or _stop:
            save_state()
        if _stop:
            print(f"signal: saved at step {step}, exiting", flush=True)
            pool.shutdown()
            sys.exit(0)
    pool.shutdown()
    print("done", flush=True)


if __name__ == "__main__":
    main()
