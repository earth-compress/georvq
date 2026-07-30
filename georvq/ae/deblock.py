"""The AE deblock decoder variant: seam removal as release policy.

The base codec decodes 64 px windows independently, leaving seams at
every window boundary (~20x excess adjacency error at full rate, ~200x
at low rate). This module is the shipped fix: a small FiLM-conditioned
single-resolution UNet (ported weight-compatibly from the research
repo's `ring_space/ae_deblock.py`) that runs on the decoded mosaic and
removes the seams. It is decoder-side and bitstream-free — packets and
archives never change, and `bitstream_id` is untouched.

Canonical recipe (measured in the 2026-07-30 recipe sweep, 32 val
chips): the tile is partitioned into 512 px frames anchored at the
window-grid origin; each frame is deblocked from its frame mosaic plus
a one-window (64 px) halo clipped at the mosaic edge, and the frame
core is kept (hard crop — exact partition, one network evaluation per
output pixel). This is NOT a tunable: the network's GroupNorm couples
its output to the frame extents, so the tiling IS part of the decoder
definition. The recipe constants are stamped into the packaged decoder
file and folded into its `decoder_id`; quality layers minted
server-side use the identical routine. Measured properties: deviation
from whole-mosaic inference ~1e-5 dcos (three orders below
reconstruction error); residual frame-stitch adjacency within 2.3% of
the ordinary post-deblock window-seam level; 1.25 GB peak CUDA memory
per 640 px input frame in fp32.

Locality contract: a deblocked output pixel depends on the code indices
within 64 px of its frame, and nothing else. Any scheduler that honors
the recipe (per-frame, whole-tile-with-shared-window-decodes,
ranged-read viewport) produces the same output; window decodes may be
shared between frames as a pure optimization. At mosaic (tile) edges
the halo is clipped: the network sees no cross-tile context and leaves
the tile-boundary seam essentially untouched rather than hallucinating
(measured +1.3% dcos in the 128 px border band).
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


class AEDeblocker(nn.Module):
    """Single-res UNet-lite (down x2 -> bottleneck -> up x2 with skips).

    forward(x, Ls): x is the blocky (B, bands, H, W) unit-norm mosaic
    (H, W multiples of 4; the research original required square inputs —
    this port generalizes the seam mask so clipped-halo edge frames
    work), Ls is (B, n_tiers) the matryoshka tuple. Returns the
    deblocked mosaic, unit-norm. Zero-init head + zero-init FiLM ->
    exact identity at step 0. State-dict names match the research
    checkpoint so trained weights load unmapped.
    """

    def __init__(self, bands: int = 64, width: int = 48, cond_dim: int = 64,
                 depths=(8, 8, 8), win: int = 64, groups: int = 8,
                 blocks: int = 1):
        super().__init__()
        self.bands = bands
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
        delta = self.head(u2)
        return F.normalize(x + delta, dim=1, eps=1e-6)


@torch.no_grad()
def deblock_mosaic(net, x, Ls, frame=None, halo=None):
    """The canonical tiled application: x (1, bands, H, W) unit-norm
    blocky mosaic -> deblocked mosaic, identical for any scheduler.

    Frames of RECIPE size anchored at (0, 0), each run on its core plus
    a `halo` ring clipped at the mosaic edge, hard-cropped back to the
    core. frame/halo default to the RECIPE; overriding them exists for
    tests only — shipped decode paths must not pass them."""
    frame = frame or RECIPE["frame"]
    halo = halo or RECIPE["halo"]
    H, W = x.shape[-2:]
    if frame >= H and frame >= W:
        return net(x, Ls)
    out = torch.empty_like(x)
    for r0 in range(0, H, frame):
        for c0 in range(0, W, frame):
            r1, c1 = min(r0 + frame, H), min(c0 + frame, W)
            ri0, ci0 = max(0, r0 - halo), max(0, c0 - halo)
            ri1, ci1 = min(H, r1 + halo), min(W, c1 + halo)
            y = net(x[:, :, ri0:ri1, ci0:ci1], Ls)
            out[:, :, r0:r1, c0:c1] = \
                y[:, :, r0 - ri0:r1 - ri0, c0 - ci0:c1 - ci0]
    return out


def frame_support(gh, gw, window, frame=None, halo=None):
    """Yield (rows, cols, support) per canonical frame of a (gh, gw)
    window grid: `rows`/`cols` are the core's window-index ranges,
    `support` the clipped halo'd ranges — the exact window set a frame's
    output depends on (the locality contract, in window units)."""
    frame = frame or RECIPE["frame"]
    halo = halo or RECIPE["halo"]
    fw, hw = frame // window, halo // window
    for r0 in range(0, gh, fw):
        for c0 in range(0, gw, fw):
            r1, c1 = min(r0 + fw, gh), min(c0 + fw, gw)
            yield ((r0, r1), (c0, c1),
                   ((max(0, r0 - hw), min(gh, r1 + hw)),
                    (max(0, c0 - hw), min(gw, c1 + hw))))


def load_deblocker(path, device="cpu"):
    """Packaged decoder file -> (eval-mode net, entry dict). The entry
    carries `recipe`, `trained_against` {bitstream_id, decoder_id} and
    provenance; callers must refuse mismatched bundles (a deblocker is
    trained against one base decoder's outputs)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = AEDeblocker(**ck["arch"]).to(device).eval()
    net.load_state_dict(ck["model"])
    for p in net.parameters():
        p.requires_grad_(False)
    entry = {k: ck[k] for k in ("recipe", "trained_against", "provenance")}
    return net, entry
