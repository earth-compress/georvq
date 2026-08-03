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
        # pre: k-means init / dead resets may have written off-grid vectors
        # post: the EMA update re-derived an fp32 weight from the buffers
        if self.training:
            self.project_book()
        out = super().forward(z)
        if self.training:
            self.project_book()
        return out


class QuantizedResidualVQ(ResidualVQ):
    """ResidualVQ with QuantizedBookVQ levels. Same state-dict layout."""

    def __init__(self, num_levels=8, num_codes=256, dim=128, decay=0.99,
                 kmeans_init=True, reset_interval=2500, dead_threshold=0.01,
                 grad_codebook=False, bf16_scores=False, book_bits=8):
        nn.Module.__init__(self)
        self.num_levels = num_levels
        self.levels = nn.ModuleList([
            QuantizedBookVQ(num_codes, dim, decay,
                            kmeans_init=kmeans_init,
                            reset_interval=reset_interval,
                            dead_threshold=dead_threshold,
                            grad_codebook=grad_codebook,
                            bf16_scores=bf16_scores,
                            book_bits=book_bits)
            for _ in range(num_levels)])

    @torch.no_grad()
    def project_books(self):
        """Project every level (call after external writes, e.g. the
        residual k-means init)."""
        for lev in self.levels:
            lev.project_book()
