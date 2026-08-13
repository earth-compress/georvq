"""Encoder drift probe: how fast does the latent space move under the
encoder's feet?

A fixed batch -- the SAME windows every time -- is pushed through the
encoder every `interval` steps and compared with the previous reading.
That gives the encoder's velocity through latent space, which is the
quantity every codebook-maintenance decision implicitly depends on and
which we have so far only guessed at.

Why it matters concretely. An EMA codebook trades bias against variance:
its samples are numerous but smeared over a moving encoder, so

    bias      ~ delta * W / 2        (drift accumulated over the window)
    variance  ~ sigma^2 * K / (W*B)  (finite samples per codeword)

with delta the per-step drift, sigma the cell radius, W the effective
window (1+d)/(1-d), K the codebook size and B vectors seen per step.
Minimising bias^2 + variance gives

    W* = ( 2 * sigma^2 * K / (delta^2 * B) )^(1/3)

so a measured delta sets the decay instead of it being a hyperparameter.
The same number says when a periodic refit is preferable: once
delta*W/2 exceeds roughly sigma/10, extra EMA samples are buying nothing
because they describe an encoder that no longer exists.

Cheap: one encoder-only forward per interval, no decode, no backward.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class DriftProbe:
    def __init__(self, x, interval=100, tier_codes=None, vecs_per_step=None,
                 n_lags=4):
        self.x = x
        self.interval = max(1, int(interval))
        self.codes = tier_codes
        self.vps = vecs_per_step
        # snapshots at several lags, so the SCALING EXPONENT of drift can
        # be fitted: d(T) ~ T^alpha with alpha ~ 1 ballistic (systematic
        # motion, bias accumulates as delta*W) and alpha ~ 0.5 diffusive
        # (random walk, bias accumulates as delta*sqrt(W)). The two give
        # different optimal windows, so guessing which one applies is
        # exactly the thing this is meant to stop.
        self.hist = []                      # [(step, zs)], newest last
        self.n_lags = n_lags

    @torch.no_grad()
    def __call__(self, model, step):
        if step % self.interval:
            return None
        was_training = model.training
        model.eval()
        zs = [z.float() for z in model.encode_levels(self.x)]
        # cell radius: RMS distance to the nearest codeword at level 0 of
        # each tier -- the scale against which drift has to be judged
        radii = []
        for i, z in enumerate(zs):
            r = None
            if getattr(model, "rvqs", None) is not None:
                C = model.rvqs[i].levels[0].codebook.weight.data.float()
                flat = z.reshape(-1, z.shape[-1])
                d = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ C.t()
                     + C.pow(2).sum(1)[None])
                r = float((flat - C[d.argmin(1)]).pow(2).sum(1).mean().sqrt())
            radii.append(r)
        model.train(was_training)

        out = None
        if self.hist:
            # multi-lag: drift against every retained snapshot
            lags, dmean = [], []
            for (s0, z0) in self.hist:
                lag = step - s0
                if lag <= 0:
                    continue
                lags.append(lag)
                dmean.append(float(sum(
                    float((a - b).norm(dim=-1).mean())
                    for a, b in zip(zs, z0)) / len(zs)))
            if len(lags) >= 2:
                import math
                lx = [math.log(l) for l in lags]
                ly = [math.log(max(d, 1e-12)) for d in dmean]
                mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
                den = sum((v - mx) ** 2 for v in lx)
                alpha = (sum((a_ - mx) * (b_ - my)
                             for a_, b_ in zip(lx, ly)) / den) if den else 0.0
            else:
                alpha = float("nan")

            s_prev, z_prev = self.hist[-1]
            dt = step - s_prev
            out = {"drift/alpha": alpha}
            for k, (lg, dm) in enumerate(zip(lags, dmean)):
                out[f"drift/lag{lg}_l2"] = dm
            for i, (a, b) in enumerate(zip(zs, z_prev)):
                d = (a - b).norm(dim=-1)
                per_step = float(d.mean()) / dt
                out[f"drift/pyr{i}_l2_per_step"] = per_step
                out[f"drift/pyr{i}_rel_per_step"] = per_step / max(
                    float(b.norm(dim=-1).mean()), 1e-9)
                out[f"drift/pyr{i}_cos"] = float(
                    F.cosine_similarity(a, b, dim=-1).mean())
                sig = radii[i]
                if sig and per_step > 0:
                    # drift measured in cell radii -- the scale-free number
                    out[f"drift/pyr{i}_per_step_over_sigma"] = per_step / sig
                    if self.codes and self.vps:
                        # ballistic optimum; if alpha comes out
                        # near 0.5 the diffusive form applies instead
                        W = (2 * sig ** 2 * self.codes[i]
                             / (per_step ** 2 * self.vps[i])) ** (1 / 3)
                        W_diff = (sig * (self.codes[i]
                                         / self.vps[i]) ** 0.5) / per_step
                        out[f"drift/pyr{i}_optimal_window_diffusive"] = W_diff
                        # W < 1 means the encoder outruns any averaging:
                        # the honest reading is 'an EMA cannot help here',
                        # so clamp rather than report a negative decay
                        out[f"drift/pyr{i}_optimal_ema_window"] = W
                        out[f"drift/pyr{i}_optimal_ema_decay"] = min(
                            0.9999, max(0.0, (W - 1) / (W + 1)))
        self.hist.append((step, zs))
        if len(self.hist) > self.n_lags:
            self.hist.pop(0)
        return out
