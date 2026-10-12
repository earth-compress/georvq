"""Quantize-aware codebooks: RVQ whose codewords live on a b-bit grid.

Ship-ready codebooks (int8 by default: per-codeword symmetric scale +
b-bit integers) trained with projected-EMA QAT. The fp32 "master" state
is the EMA buffers — `VectorQuantizer` re-derives the weight from
ema_sum/ema_count every update — so projecting `codebook.weight` onto
the quantization grid after each training forward implements
quantize-aware fitting without touching the parent's numerics:

- assignment always searches the QUANTIZED book (encoder/decoder
  symmetry: the search sees exactly what ships);
- z_q is gathered before the EMA update, i.e. from the quantized book;
- residual levels quantize the residual the *quantized* previous level
  left behind, so early levels' grid coarseness is self-corrected and
  only the last level's grid error reaches the reconstruction;
- outside a training forward (eval, checkpoints, direct
  `codebook.weight` reads) the state is always on-grid, so a saved
  model's books re-encode to integers + scales losslessly.

`georvq/model/vq.py` is untouched: existing checkpoints decode
bit-faithfully through the default classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from georvq.model.vq import ResidualVQ, VectorQuantizer


class QuantizedBookVQ(VectorQuantizer):
    def __init__(self, *args, book_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        assert 2 <= book_bits <= 8
        self.book_bits = book_bits

    @torch.no_grad()
    def project_book(self):
        """Snap every codeword onto its symmetric b-bit grid (per-codeword
        max-abs scale). Idempotent."""
        W = self.codebook.weight.data
        qmax = float(2 ** (self.book_bits - 1) - 1)
        s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
        W.copy_((W / s).round().clamp(-qmax, qmax) * s)

    def forward(self, z):
        """Parent semantics with two changes: the book is projected onto
        the b-bit grid around the update, and the EMA update uses
        bincount/index_add instead of a dense one-hot — the (N, K) one-hot
        is ~16 GiB at K=16384 with 512x1024-cell batches."""
        import torch.nn.functional as F
        if self.training:
            self.project_book()
        flat_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.dim)
        if self.training and not self.initialized.item():
            self._data_init(z_flat.detach())
        if self.bf16_scores:
            E = self.codebook.weight
            s = z_flat.bfloat16() @ E.t().bfloat16()
            indices = (s - (E.pow(2).sum(1) * 0.5).bfloat16()[None]) \
                .argmax(dim=-1)
        else:
            d = (z_flat.pow(2).sum(1, keepdim=True)
                 - 2 * z_flat @ self.codebook.weight.t()
                 + self.codebook.weight.pow(2).sum(1, keepdim=True).t())
            indices = d.argmin(dim=-1)
        z_q = self.codebook(indices)
        if self.training and not self.grad_codebook:
            with torch.no_grad():
                count = torch.bincount(
                    indices, minlength=self.num_codes).to(z_flat.dtype)
                vec_sum = torch.zeros_like(self.ema_sum, dtype=z_flat.dtype) \
                    .index_add_(0, indices, z_flat)
                self.ema_count.mul_(self.decay).add_(count,
                                                     alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(vec_sum,
                                                   alpha=1 - self.decay)
                n = self.ema_count.sum()
                cs = (self.ema_count + self.eps) \
                    / (n + self.num_codes * self.eps) * n
                self.codebook.weight.data.copy_(self.ema_sum / cs.unsqueeze(1))
                if self.reset_mode == "split":
                    # this forward duplicates the parent's EMA block, so
                    # the distortion tracker has to be maintained here too
                    # -- otherwise reset_mode silently degrades to random
                    # on the QAT path while still reporting "split"
                    se = (z_flat - z_q).pow(2).sum(1).float()
                    self.ema_sse.mul_(self.decay).add_(
                        torch.zeros_like(self.ema_sse).index_add_(
                            0, indices, se), alpha=1 - self.decay)
                self.step += 1
                if self.reset_pending:
                    self.reset_pending = False
                    self._reset_dead(z_flat.detach(), indices,
                                     z_q.detach())
        commit_loss = F.mse_loss(z_flat, z_q.detach())
        if self.grad_codebook:
            commit_loss = commit_loss + F.mse_loss(z_q, z_flat.detach())
        z_q_st = z_flat + (z_q - z_flat).detach()
        if self.training:
            self.project_book()
        return (z_q_st.reshape(*flat_shape, self.dim),
                indices.reshape(flat_shape), commit_loss)


class QuantizedResidualVQ(ResidualVQ):
    """ResidualVQ with QuantizedBookVQ levels. Same state-dict layout."""

    def __init__(self, num_levels=8, num_codes=256, dim=128, decay=0.99,
                 kmeans_init=True, reset_interval=2500, dead_threshold=0.01,
                 grad_codebook=False, bf16_scores=False, book_bits=8,
                 reset_mode="random"):
        nn.Module.__init__(self)
        self.num_levels = num_levels
        self.levels = nn.ModuleList([
            QuantizedBookVQ(num_codes, dim, decay,
                            kmeans_init=kmeans_init,
                            reset_interval=reset_interval,
                            dead_threshold=dead_threshold,
                            grad_codebook=grad_codebook,
                            bf16_scores=bf16_scores,
                            book_bits=book_bits,
                            reset_mode=reset_mode)
            for _ in range(num_levels)])

    @torch.no_grad()
    def project_books(self):
        """Project every level (call after external writes, e.g. the
        residual k-means init)."""
        for lev in self.levels:
            lev.project_book()
