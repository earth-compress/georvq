"""Pyramid attention decoder: full self-attention over all tier tokens with
AdaLN-Zero conditioning on the matryoshka depth vector (or a continuous
sigma embedding in stage-1 mode), followed by a per-source pixel head.

The pixel head is pluggable per data source (`DenseHead` for single-group
dense fields like AlphaEarth; multi-resolution heads for sources like
Sentinel-2). The bitstream is decoder-independent: this module can be
upgraded or replaced without re-encoding (see docs/BITSTREAM.md).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.model.layers import AdaLNBlock, sigma_embed, thermometer


class DenseHead(nn.Module):
    """Fine tokens -> a single dense pixel plane.

    Per-fine-token linear patch projection then a small conv stack that
    smooths patch seams; optional per-pixel L2 normalization for unit-norm
    embedding fields (AlphaEarth)."""

    def __init__(self, in_dim, grid, p, bands, head_feat=64, normalize=True,
                 dw=False):
        super().__init__()
        self.grid, self.p, self.normalize = grid, p, normalize
        self.to_patch = nn.Linear(in_dim, head_feat * p * p)

        def block(cin, cout):
            if not dw:
                return [nn.Conv2d(cin, cout, 3, 1, 1)]
            # depthwise-separable: 3x3 per-channel then 1x1 mix (~8x fewer
            # MACs at these widths)
            return [nn.Conv2d(cin, cin, 3, 1, 1, groups=cin),
                    nn.Conv2d(cin, cout, 1)]
        self.conv = nn.Sequential(
            *block(head_feat, head_feat), nn.GELU(),
            *block(head_feat, head_feat), nn.GELU(),
            *block(head_feat, bands))

    def forward(self, fine):                       # fine: (B, g*g, in_dim)
        B, g, p = fine.shape[0], self.grid, self.p
        f = self.to_patch(fine).view(B, g, g, -1, p, p)
        f = f.permute(0, 3, 1, 4, 2, 5).reshape(B, -1, g * p, g * p)
        out = self.conv(f)
        return F.normalize(out, dim=1, eps=1e-6) if self.normalize else out


class PyramidAttnDecoder(nn.Module):
    """Mixing decoder over [tier-0 .. tier-n cells] tokens with AdaLN-Zero
    conditioning on the depth vector, then a per-source pixel head.

    sigma_cond: condition on a per-tier continuous noise level (B, n_tiers)
    via a Fourier embedding instead of the per-tier depth thermometer
    (stage-1: a noise-robust, codebook-agnostic decoder)."""

    def __init__(self, specs, head, dim=384, depth=8, heads=8,
                 cls_bridge=False, ff_ratio=4.0, sigma_cond=False,
                 sigma_freqs=8, late_join=0):
        super().__init__()
        assert 0 <= late_join < depth
        # With one tier the fine token IS the coarse token, so bridging
        # would concatenate it with itself and double the head's input
        # width against what `head_in_dim` reports. Coerce here so the
        # two agree for every spec list, including a bank member that
        # decodes the coarse tier alone.
        cls_bridge = cls_bridge and len(specs) > 1
        self.cls_bridge = cls_bridge
        self.sigma_cond = sigma_cond
        self.sigma_freqs = sigma_freqs
        self.late_join = late_join
        self.grad_ckpt = False        # set by the factory from train cfg
        self.specs = specs
        self.fine = specs[-1]
        n_tokens = sum(sp.cells for sp in specs)
        self.projs = nn.ModuleList([
            nn.Linear(sp.dim, dim)
            for sp in specs])
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        cond_in = (len(specs) * 2 * sigma_freqs if sigma_cond
                   else sum(sp.depth for sp in specs))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_in, dim), nn.SiLU(),
                                      nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([
            AdaLNBlock(dim, heads, cond_dim=dim, ff_ratio=ff_ratio)
            for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)
        if cls_bridge and len(specs) > 2:
            # fine cell (r,c) -> ancestor cell index per intermediate tier
            g_f = specs[-1].grid
            maps = []
            for sp in specs[1:-1]:
                f = g_f // sp.grid
                idx = [(r // f) * sp.grid + (c // f)
                       for r in range(g_f) for c in range(g_f)]
                maps.append(torch.tensor(idx))
            self.register_buffer("anc_maps", torch.stack(maps))
        self.head = head

    @staticmethod
    def head_in_dim(specs, dim, cls_bridge):
        """Input width of the pixel head: fine token, plus (with cls_bridge)
        every ancestor token concatenated per fine cell."""
        cls_bridge = cls_bridge and len(specs) > 1
        return dim + (dim * (len(specs) - 1) if cls_bridge else 0)

    def forward(self, zqs, cond_arg, stats=None):
        """zqs: list of (B, cells_i, dim_i). cond_arg: (B, n_tiers) integer
        depths (depth mode), or (B, n_tiers) per-tier sigma (sigma_cond)."""
        toks = torch.cat([proj(z) for proj, z in zip(self.projs, zqs)], 1)
        x = toks + self.pos
        c = self._cond(cond_arg, stats)

        def run(h, blocks):
            m = getattr(self, "attn_mask", None)
            if m is not None:
                m = m[:h.shape[1], :h.shape[1]]    # late-join prefix slice
            if self.grad_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint
                for blk in blocks:
                    h = checkpoint(blk, h, c, m, use_reentrant=False)
            else:
                for blk in blocks:
                    h = blk(h, c, m)
            return h

        if self.late_join:
            nb = x.shape[1] - self.fine.cells
            base = run(x[:, :nb], self.blocks[:len(self.blocks) - self.late_join])
            x = run(torch.cat([base, x[:, nb:]], 1),
                    self.blocks[len(self.blocks) - self.late_join:])
        else:
            x = run(x, self.blocks)
        xo = self.norm_out(x)
        fine = xo[:, -self.fine.cells:]                    # (B, cells, dim)
        if self.cls_bridge:    # lineage: concat all ancestors per fine cell
            parts = [fine, xo[:, :1].expand(-1, self.fine.cells, -1)]
            off = 1
            for li, sp in enumerate(self.specs[1:-1]):
                toks = xo[:, off:off + sp.cells]
                parts.append(toks[:, self.anc_maps[li]])
                off += sp.cells
            fine = torch.cat(parts, -1)
        return self.head(fine)

    def _cond(self, cond_arg, stats=None):
        """AdaLN conditioning. `stats` (B, dim) is the patch-mean embedding,
        added to the depth/sigma conditioning rather than concatenated, so
        cond_in is unchanged and a checkpoint without stats conditioning
        keeps exactly its current geometry."""
        if self.sigma_cond:
            c = self.cond_mlp(sigma_embed(cond_arg, self.sigma_freqs))
        else:
            Ls = cond_arg
            c = self.cond_mlp(torch.cat(
                [thermometer(Ls[:, i], sp.depth, Ls.device)
                 for i, sp in enumerate(self.specs)], -1))
        return c if stats is None else c + stats



def active_tiers(depths) -> int:
    """How many tiers a depth tuple actually carries codes for.

    Tuples are saturating: a tier only opens once the one below it is
    full, so the active set is a prefix and its length is enough to pick
    a decoder. A tuple that breaks that (a zero below a nonzero) has no
    prefix decoder that could serve it, so it routes to the full decoder
    rather than being silently mis-routed.
    """
    d = list(depths)
    last = 0
    for i, v in enumerate(d):
        if v > 0:
            last = i + 1
    if any(v == 0 for v in d[:last]):
        return len(d)
    return max(1, last)


class DecoderBank(nn.Module):
    """One decoder per tier prefix, each emitting at its own resolution.

    Decoding a rung that carries only the coarse tier costs the same as
    the full pyramid today: 273 tokens through the trunk and a full-size
    head. A prefix decoder sees only the tokens that exist at that rung
    and emits a smaller image, which the caller resamples back to the
    window. Members are ordinary `PyramidAttnDecoder`s over `specs[:k]`,
    so the block, conditioning and head machinery is shared rather than
    duplicated.
    """

    def __init__(self, members, window):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.window = window
        self.specs = members[-1].specs

    @property
    def grad_ckpt(self):
        return self.members[-1].grad_ckpt

    @grad_ckpt.setter
    def grad_ckpt(self, v):
        for m in self.members:
            m.grad_ckpt = v

    def pick(self, cond_arg) -> int:
        """Index of the member serving a whole batch at one depth tuple."""
        d = cond_arg[0].tolist() if hasattr(cond_arg, "tolist") else cond_arg
        return active_tiers([int(v) for v in d]) - 1

    def route(self, cond_arg):
        """(B,) member index per row."""
        import torch
        return torch.tensor(
            [active_tiers([int(v) for v in row]) - 1
             for row in cond_arg.tolist()],
            device=cond_arg.device, dtype=torch.long)

    def _one(self, k, zqs, cond_arg, stats, native):
        dec = self.members[k]
        n = len(dec.specs)
        out = dec(list(zqs[:n]), cond_arg[:, :n], stats=stats)
        return out if native else _to_window(out, self.window)

    def forward(self, zqs, cond_arg, stats=None, member=None,
                native=False):
        """Decode, routing each row to the member its depth tuple needs.

        Training draws a depth tuple per row, so a batch is generally
        mixed and the rows that only carry the coarse tier must not pay
        for the full decoder -- that saving is most of the point. Rows
        are grouped, each group runs through one member, and the results
        are scattered back into batch order. A batch that happens to be
        uniform takes the single-call path.

        `member` forces one member for the whole batch; `native` returns
        each member's own resolution instead of resampling to the window,
        which only makes sense when the batch is uniform.
        """
        if member is not None:
            return self._one(member, zqs, cond_arg, stats, native)
        idx = self.route(cond_arg)
        uniq = idx.unique().tolist()
        self.last_counts = {k: int((idx == k).sum()) for k in uniq}
        if len(uniq) == 1:
            return self._one(uniq[0], zqs, cond_arg, stats, native)
        if native:
            raise ValueError(
                "native=True needs a batch that routes to one member; this "
                f"batch spans {uniq}. Pass member= or decode uniform rows.")
        import torch
        parts = {}
        for k in uniq:
            sel = (idx == k).nonzero(as_tuple=True)[0]
            sub_stats = None if stats is None else stats[sel]
            parts[k] = (sel, self._one(k, [z[sel] for z in zqs],
                                       cond_arg[sel], sub_stats, False))
        any_out = parts[uniq[0]][1]
        B = cond_arg.shape[0]
        # Members can disagree on dtype under autocast: a member that is
        # resampled to the window goes through F.interpolate, which runs
        # in fp32, while the full-resolution member returns bf16
        # untouched. Promote to whichever is widest rather than letting
        # the scatter reject the mix.
        def _dtypes(o):
            return list(o.values()) if isinstance(o, dict) else [o]

        dt = _dtypes(any_out)[0].dtype
        for _, o in parts.values():
            for t in _dtypes(o):
                dt = torch.promote_types(dt, t.dtype)

        def blank(t):
            return torch.zeros((B,) + tuple(t.shape[1:]), dtype=dt,
                               device=t.device)

        if isinstance(any_out, dict):
            full = type(any_out)({g: blank(t) for g, t in any_out.items()})
            for sel, o in parts.values():
                for g, t in o.items():
                    full[g][sel] = t.to(dt)
            return full
        full = blank(any_out)
        for sel, o in parts.values():
            full[sel] = o.to(dt)
        return full


def _to_window(out, window):
    """Resample a member's output up to the model's window.

    Bilinear, and applied to whatever the head emitted -- a dict of band
    groups for a multi-resolution source, a tensor otherwise. Each group
    keeps its own ratio to the window, so a group already at its full
    size is left alone.
    """
    import torch.nn.functional as F

    def up(t, edge):
        return (t if t.shape[-1] == edge else
                F.interpolate(t, size=(edge, edge), mode="bilinear",
                              align_corners=False))

    if isinstance(out, dict):
        ratios = {k: v.shape[-1] for k, v in out.items()}
        fine = max(ratios.values())
        return type(out)({k: up(v, max(1, window * v.shape[-1] // fine))
                          for k, v in out.items()})
    return up(out, window)
