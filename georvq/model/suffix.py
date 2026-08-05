"""Narrow blocked-halo cross-attention suffix for the finest pyramid tier.

The finest tier carries most trunk tokens but only its (small) latent
dim of content; these layers process it at its own width. Attention is
blocked-halo (HaloNet/Swin family): the cell grid is partitioned into
b x b blocks and each block's queries attend to
    [all base-tier tokens | the block's (b+2r) x (b+2r) halo]
K/V are gathered once per BLOCK (duplication ~(b+2r)^2/b^2, ~4x at
b=2r), and the attention itself is one batched
F.scaled_dot_product_attention call over (B*blocks, H, b^2, Nb+halo^2)
— dense flash kernels on tensor-core-friendly tiles. A per-query gather
formulation was ~150x off arithmetic peak (permute/bandwidth-bound);
this form is the practical kernel. The neighborhood is the block's halo
(a superset of the centered (2r+1)^2 set for interior queries).
Base tokens are read, never written (Perceiver-style; the AE attention
probe showed fine tokens receive below-uniform attention as keys).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.model.layers import modulate


class NeighborhoodCrossAttention(nn.Module):
    """Multi-head attention of fine queries over [base | block halo],
    computed as batched dense attention per block."""

    def __init__(self, dim, base_dim, grid, radius, heads=8, block=None):
        super().__init__()
        assert dim % heads == 0
        b = block or max(2 * radius, 2)
        while grid % b:
            b -= 1                                     # largest divisor <= 2r
        self.g, self.b, self.r = grid, b, radius
        self.nb = grid // b
        self.halo = b + 2 * radius
        self.h, self.dh = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv_f = nn.Linear(dim, 2 * dim)
        self.kv_b = nn.Linear(base_dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        # out-of-bounds halo positions per block (nb*nb, halo*halo)
        rr = torch.arange(self.halo) - radius
        oob = torch.zeros(self.nb, self.nb, self.halo, self.halo,
                          dtype=torch.bool)
        for bi in range(self.nb):
            for bj in range(self.nb):
                r_abs = bi * b + rr[:, None]
                c_abs = bj * b + rr[None, :]
                oob[bi, bj] = ((r_abs < 0) | (r_abs >= grid)
                               | (c_abs < 0) | (c_abs >= grid))
        self.register_buffer(
            "oob", oob.reshape(self.nb * self.nb, self.halo * self.halo),
            persistent=False)

    def _blockify(self, t):
        """(B, g*g, D) -> halo blocks (B, nb*nb, halo*halo, D)."""
        B, _, D = t.shape
        x = t.view(B, self.g, self.g, D).permute(0, 3, 1, 2)
        x = F.pad(x, (self.r,) * 4)                    # zero halo padding
        x = x.unfold(2, self.halo, self.b).unfold(3, self.halo, self.b)
        # (B, D, nb, nb, halo, halo) -> (B, nb*nb, halo*halo, D)
        return x.permute(0, 2, 3, 4, 5, 1).reshape(
            B, self.nb * self.nb, self.halo * self.halo, D)

    def forward(self, fine, base):
        B, N, D = fine.shape
        H, dh = self.h, self.dh
        nb2, hh = self.nb * self.nb, self.halo * self.halo
        Nb = base.shape[1]
        q = self.q(fine)
        kf, vf = self.kv_f(fine).chunk(2, -1)
        kb, vb = self.kv_b(base).chunk(2, -1)          # (B, Nb, D)
        # queries per block: (B, nb2, b*b, D)
        bb = self.b * self.b
        qx = q.view(B, self.nb, self.b, self.nb, self.b, D) \
              .permute(0, 1, 3, 2, 4, 5).reshape(B, nb2, bb, D)
        # keys/values: [base | halo]
        khl = self._blockify(kf)
        vhl = self._blockify(vf)
        k = torch.cat([kb.unsqueeze(1).expand(-1, nb2, -1, -1), khl], 2)
        v = torch.cat([vb.unsqueeze(1).expand(-1, nb2, -1, -1), vhl], 2)
        # heads: (B*nb2, H, L, dh)
        def heads_of(t, L):
            return t.reshape(B * nb2, L, H, dh).transpose(1, 2)
        qh = heads_of(qx, bb)
        kh = heads_of(k, Nb + hh)
        vh = heads_of(v, Nb + hh)
        mask = torch.zeros(nb2, 1, 1, Nb + hh, device=fine.device,
                           dtype=qh.dtype)
        mask[..., Nb:] = self.oob.view(nb2, 1, 1, hh) * torch.finfo(
            qh.dtype).min
        mask = mask.repeat(B, 1, 1, 1)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        o = o.transpose(1, 2).reshape(B, nb2, bb, D)
        o = o.view(B, self.nb, self.nb, self.b, self.b, D) \
             .permute(0, 1, 3, 2, 4, 5).reshape(B, N, D)
        return self.out(o)


class CrossSuffixLayer(nn.Module):
    """Pre-norm blocked-halo cross-attention block (tokenizer side)."""

    def __init__(self, dim, base_dim, grid, radius, heads=8, ff_ratio=2.0):
        super().__init__()
        self.norm_b = nn.LayerNorm(base_dim, eps=1e-6)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = NeighborhoodCrossAttention(dim, base_dim, grid, radius,
                                               heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        ff = int(dim * ff_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, ff), nn.GELU(),
                                 nn.Linear(ff, dim))

    def forward(self, fine, base):
        fine = fine + self.attn(self.norm1(fine), self.norm_b(base))
        fine = fine + self.mlp(self.norm2(fine))
        return fine


class CrossAdaLNBlock(nn.Module):
    """Blocked-halo cross-attention block with AdaLN-Zero conditioning
    (decoder side)."""

    def __init__(self, dim, base_dim, grid, radius, heads=8, cond_dim=384,
                 ff_ratio=2.0):
        super().__init__()
        self.norm_b = nn.LayerNorm(base_dim, eps=1e-6)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = NeighborhoodCrossAttention(dim, base_dim, grid, radius,
                                               heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        ff = int(dim * ff_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, ff), nn.GELU(),
                                 nn.Linear(ff, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)   # identity block at init

    def forward(self, fine, base, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        q = modulate(self.norm1(fine), s1, sc1)
        fine = fine + g1.unsqueeze(1) * self.attn(q, self.norm_b(base))
        h = modulate(self.norm2(fine), s2, sc2)
        fine = fine + g2.unsqueeze(1) * self.mlp(h)
        return fine
