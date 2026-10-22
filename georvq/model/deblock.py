"""A seam-removal post-decoder for window-based codecs.

A pyramid codec decodes each window independently, which leaves seams
at the window boundaries. The deblocker is a small FiLM-conditioned
single-resolution UNet that runs on the decoded mosaic, conditioned on
the decode depth tuple, and predicts a correction. It is decoder-side
and adds nothing to the bitstream: packets do not change and the
`bitstream_id` is untouched, while the post-processor's weights and
tiling recipe enter the `decoder_id` (georvq.inference.release.decoder_id).

The network was developed on AlphaEarth, where it removed most of the
excess seam error (lab finding `seam-deblocker`); it is kept here as an
exploratory tool for other sources. Any band count and window size
work. `unit_norm=True` renormalises each output pixel, which is right
for unit-norm embedding fields and wrong for physical fields.

Tiling: GroupNorm couples the output to the frame extents, so the frame
and halo sizes are part of the decoder's definition, not a tunable.
`deblock_mosaic` partitions the mosaic into frames anchored at the
window-grid origin, runs each frame with a halo clipped at the mosaic
edge, and keeps the frame core. A core pixel then depends only on the
windows within `halo` px of its frame, so any scheduler that follows
the recipe gives the same output. `RECIPE` holds the values measured on
AlphaEarth (512 px frames, 64 px halo); another source should measure
its own.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"frame": 512, "halo": 64, "blend": "crop", "version": 1}


def thermometer(L, depth, device):
    """L: (B,) ints in [0, depth] -> (B, depth) cumulative encoding."""
    return (torch.arange(depth, device=device)[None] < L[:, None]).float()


def boundary_mask(h, w, win, device, dtype=torch.float32):
    """(1,1,h,w) marking window-seam positions: 1 where the row or col
    index is a positive multiple of `win`. Frames are window-aligned, so
    a seam at absolute position k*win lands at relative position k*win
    in every clipped-halo frame."""
    on = lambda n: ((torch.arange(n, device=device) % win == 0)
                    & (torch.arange(n, device=device) > 0)).to(dtype)
    return (on(h)[:, None] + on(w)[None, :]).clamp(max=1.0)[None, None]


class FiLMResBlock(nn.Module):
    """Pre-act 3x3 residual block, FiLM-modulated between the two convs.
    FiLM is zero-init so conditioning is inert at step 0."""

    def __init__(self, width: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, width)
        self.c1 = nn.Conv2d(width, width, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, width)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        g, b = self.film(cond).chunk(2, dim=-1)
        h = self.c1(F.gelu(self.n1(x)))
        h = h * (1.0 + g[..., None, None]) + b[..., None, None]
        h = self.c2(F.gelu(self.n2(h)))
        return x + h


class Deblocker(nn.Module):
    """Single-res UNet-lite (down x2 -> bottleneck -> up x2 with skips).

    forward(x, Ls): x is the blocky (B, bands, H, W) decoded mosaic
    (H, W multiples of 4; frames may be rectangular), Ls is (B, n_tiers)
    the depth tuple. Returns the deblocked mosaic, renormalised per
    pixel when `unit_norm`. Zero-init head and FiLM make it the identity
    at step 0. State-dict names match the AlphaEarth research
    checkpoints, so those weights load unmapped.
    """

    def __init__(self, bands: int = 64, width: int = 48, cond_dim: int = 64,
                 depths=(8, 8, 8), win: int = 64, groups: int = 8,
                 blocks: int = 1, unit_norm: bool = True):
        super().__init__()
        self.bands = bands
        self.unit_norm = unit_norm
        self.win = win
        self.depths = list(depths)
        self.cond_mlp = nn.Sequential(
            nn.Linear(sum(self.depths), cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim))

        stack = lambda: nn.ModuleList(
            [FiLMResBlock(width, cond_dim, groups) for _ in range(blocks)])
        self.stem = nn.Conv2d(bands + 1, width, 3, padding=1)  # +1 seam mask
        self.down1 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.rb1 = stack()
        self.down2 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.rb2 = stack()                                     # bottleneck
        self.up1 = nn.Conv2d(width, width, 3, padding=1)
        self.fuse1 = nn.Conv2d(2 * width, width, 1)
        self.rb3 = stack()
        self.up2 = nn.Conv2d(width, width, 3, padding=1)
        self.fuse2 = nn.Conv2d(2 * width, width, 1)
        self.rb4 = stack()
        self.head = nn.Conv2d(width, bands, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def cond(self, Ls):
        th = torch.cat([thermometer(Ls[:, i], d, Ls.device)
                        for i, d in enumerate(self.depths)], dim=-1)
        return self.cond_mlp(th)

    @staticmethod
    def _stack(blocks, x, c):
        for b in blocks:
            x = b(x, c)
        return x

    def forward(self, x, Ls, mask=None):
        """mask: optional precomputed (B,1,H,W) seam mask — the ONNX
        export passes it as a graph input (clients build it in numpy)
        so the graph carries no shape-dependent arange."""
        H, W = x.shape[-2:]
        c = self.cond(Ls)
        if mask is None:
            mask = boundary_mask(H, W, self.win, x.device, x.dtype) \
                .expand(x.shape[0], -1, -1, -1)
        s0 = self.stem(torch.cat([x, mask], dim=1))            # (B,w,H,W)
        d1 = self._stack(self.rb1, F.gelu(self.down1(s0)), c)  # /2
        d2 = self._stack(self.rb2, F.gelu(self.down2(d1)), c)  # /4
        u1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        u1 = self._stack(self.rb3, self.fuse1(
            torch.cat([F.gelu(self.up1(u1)), d1], 1)), c)
        u2 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u2 = self._stack(self.rb4, self.fuse2(
            torch.cat([F.gelu(self.up2(u2)), s0], 1)), c)
        y = x + self.head(u2)
        return F.normalize(y, dim=1, eps=1e-6) if self.unit_norm else y


def _hw(x):
    if isinstance(x, dict):
        return next(iter(x.values())).shape[-2:]
    return x.shape[-2:]


def _cut(x, scales, r0, r1, c0, c1):
    """x[..., r0:r1, c0:c1] in units of the finest grid; a dict input is
    cut on each entry's own grid, `scales[name]` finest pixels per pixel."""
    if not isinstance(x, dict):
        return x[:, :, r0:r1, c0:c1]
    return {k: v[:, :, r0 // scales[k]:r1 // scales[k],
                 c0 // scales[k]:c1 // scales[k]] for k, v in x.items()}


@torch.no_grad()
def deblock_mosaic(net, x, Ls, frame=None, halo=None, scales=None,
                   batch=1):
    """The tiled application: x (1, bands, H, W) blocky mosaic ->
    deblocked mosaic, identical for any scheduler. Frames anchored at
    (0, 0), each run on its core plus a `halo` ring clipped at the
    mosaic edge and cropped back to the core. `frame` and `halo` default
    to RECIPE, and must match what the network was trained with.

    A source with several native grids passes x as a dict of
    (1, bands, H / s, W / s) planes and `scales` {name: s}; H, W, frame
    and halo are then on the finest grid and must divide by every s, and
    `net` maps such a dict to one of the same shapes. `batch` > 1 runs
    up to that many frames of equal input shape per call; batched
    convolutions need not round exactly as single ones do."""
    frame = frame or RECIPE["frame"]
    halo = halo or RECIPE["halo"]
    H, W = _hw(x)
    if isinstance(x, dict):
        for s in scales.values():
            assert H % s == 0 and W % s == 0 and frame % s == 0 \
                and halo % s == 0, (H, W, frame, halo, s)
    if frame >= H and frame >= W:
        return net(x, Ls)
    if isinstance(x, dict):
        out = {k: torch.empty_like(v) for k, v in x.items()}
    else:
        out = torch.empty_like(x)
    jobs = {}
    for r0 in range(0, H, frame):
        for c0 in range(0, W, frame):
            r1, c1 = min(r0 + frame, H), min(c0 + frame, W)
            ri0, ci0 = max(0, r0 - halo), max(0, c0 - halo)
            ri1, ci1 = min(H, r1 + halo), min(W, c1 + halo)
            jobs.setdefault((ri1 - ri0, ci1 - ci0), []).append(
                (r0, r1, c0, c1, ri0, ri1, ci0, ci1))
    for js in jobs.values():
        for a in range(0, len(js), batch):
            part = js[a:a + batch]
            xs = [_cut(x, scales, j[4], j[5], j[6], j[7]) for j in part]
            if isinstance(x, dict):
                xb = {k: torch.cat([t[k] for t in xs]) for k in x}
            else:
                xb = torch.cat(xs)
            y = net(xb, Ls.expand(len(part), -1))
            for i, (r0, r1, c0, c1, ri0, ri1, ci0, ci1) in enumerate(part):
                core = _cut(y, scales, r0 - ri0, r1 - ri0, c0 - ci0,
                            c1 - ci0)
                if isinstance(x, dict):
                    for k in out:
                        s = scales[k]
                        out[k][:, :, r0 // s:r1 // s, c0 // s:c1 // s] = \
                            core[k][i:i + 1]
                else:
                    out[:, :, r0:r1, c0:c1] = core[i:i + 1]
    return out


def frame_support(gh, gw, window, frame=None, halo=None):
    """Yield (rows, cols, support) per canonical frame of a (gh, gw)
    window grid: `rows`/`cols` are the core's window-index ranges and
    `support` the clipped ranges including the halo, which is the set
    of windows the frame's output depends on."""
    frame = frame or RECIPE["frame"]
    halo = halo or RECIPE["halo"]
    fw, hw = frame // window, halo // window
    for r0 in range(0, gh, fw):
        for c0 in range(0, gw, fw):
            r1, c1 = min(r0 + fw, gh), min(c0 + fw, gw)
            yield ((r0, r1), (c0, c1),
                   ((max(0, r0 - hw), min(gh, r1 + hw)),
                    (max(0, c0 - hw), min(gw, c1 + hw))))


def load_deblocker(path, device="cpu", cls=Deblocker):
    """Saved deblocker -> (eval-mode net, entry dict). The file holds
    `arch` (kwargs of `cls`), `model` (state dict), `recipe`,
    `trained_against` {bitstream_id, decoder_id} and `provenance`, and
    optionally `applies_to`, a list of {bitstream_id, decoder_id} of
    codecs it was validated on besides the one it was trained against.
    A deblocker is trained on one base decoder's outputs, so callers
    should refuse a codec that is in neither."""
    from georvq.repro import load_checkpoint
    ck = load_checkpoint(path)
    net = cls(**ck["arch"]).to(device).eval()
    net.load_state_dict(ck["model"])
    for p in net.parameters():
        p.requires_grad_(False)
    entry = {k: ck[k] for k in ("recipe", "trained_against", "provenance")}
    if "applies_to" in ck:
        entry["applies_to"] = list(ck["applies_to"])
    return net, entry
