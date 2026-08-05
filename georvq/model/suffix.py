"""Narrow neighborhood-cross-attention suffix for the finest pyramid tier.

The finest tier carries most trunk tokens but only its (small) latent
dim of content; these layers process it at its own width, and attention
is computed ONLY on the allowed pairs — each fine query attends to
    [all base-tier tokens | its (2r+1)x(2r+1) spatial neighborhood]
via an explicit gather (unfold-style neighborhood attention), not a
masked dense matrix: the QK^T matmul is (Nf, k) with k = n_base+(2r+1)^2
instead of (Nf, N). Exact, mask-free, and scales linearly in tokens.
Base tokens are read, never written (Perceiver-style; the AE attention
probe showed fine tokens receive below-uniform attention as keys).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.model.layers import modulate


def neighbor_index(grid, radius):
    """(grid*grid, (2r+1)^2) neighbor flat-indices, -1 = out of bounds."""
    rr = torch.arange(grid)
    qr, qc = torch.meshgrid(rr, rr, indexing="ij")
    off = torch.arange(-radius, radius + 1)
    orr, occ = torch.meshgrid(off, off, indexing="ij")
    nr = qr.reshape(-1, 1) + orr.reshape(1, -1)       # (N, k)
    nc = qc.reshape(-1, 1) + occ.reshape(1, -1)
    idx = nr * grid + nc
    idx[(nr < 0) | (nr >= grid) | (nc < 0) | (nc >= grid)] = -1
    return idx


class NeighborhoodCrossAttention(nn.Module):
    """Multi-head attention of fine queries over [base | local fine] keys,
    computed on the gathered pair set only."""

    def __init__(self, dim, base_dim, grid, radius, heads=8):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv_f = nn.Linear(dim, 2 * dim)
        self.kv_b = nn.Linear(base_dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        idx = neighbor_index(grid, radius)             # (N, k_local)
        self.register_buffer("nidx", idx, persistent=False)
        self.register_buffer("noob", idx < 0, persistent=False)

    def forward(self, fine, base):
        B, N, D = fine.shape
        H, dh = self.h, self.dh
        k_l = self.nidx.shape[1]
        q = self.q(fine).view(B, N, H, dh)
        kf, vf = self.kv_f(fine).chunk(2, -1)          # (B, N, D)
        kb, vb = self.kv_b(base).chunk(2, -1)          # (B, Nb, D)
        # gather local keys/values: (B, N, k_l, D)
        gi = self.nidx.clamp(min=0).reshape(-1)
        kfl = kf[:, gi].view(B, N, k_l, D)
        vfl = vf[:, gi].view(B, N, k_l, D)
        # logits: local (B,H,N,k_l) + base (B,H,N,Nb)
        ql = q.permute(0, 2, 1, 3)                     # (B,H,N,dh)
        ll = torch.einsum("bhnd,bnkhd->bhnk", ql,
                          kfl.view(B, N, k_l, H, dh))
        lb = torch.einsum("bhnd,bmhd->bhnm", ql,
                          kb.view(B, -1, H, dh))
        scale = dh ** -0.5
        logits = torch.cat([lb, ll], -1) * scale       # (B,H,N,Nb+k_l)
        oob = F.pad(self.noob, (kb.shape[1], 0))       # base keys never OOB
        logits = logits.masked_fill(oob.view(1, 1, N, -1), float("-inf"))
        a = logits.softmax(-1)
        ab, al = a[..., :kb.shape[1]], a[..., kb.shape[1]:]
        o = torch.einsum("bhnm,bmhd->bhnd", ab, vb.view(B, -1, H, dh)) \
            + torch.einsum("bhnk,bnkhd->bhnd", al,
                           vfl.view(B, N, k_l, H, dh))
        return self.out(o.permute(0, 2, 1, 3).reshape(B, N, D))


class CrossSuffixLayer(nn.Module):
    """Pre-norm neighborhood cross-attention block (tokenizer side)."""

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
    """Neighborhood cross-attention block with AdaLN-Zero conditioning
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
