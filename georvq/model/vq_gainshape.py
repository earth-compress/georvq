"""Gain-shape vector quantization (CELT/CELP formulation) for the RVQ stack.

Motivation (measured on tessera_uf_vq8_s2): the selected codeword's norm
tracks ||r||*cos(q,r) almost exactly, i.e. EMA k-means already places
codewords at the projection-optimal magnitude for their direction. The
codebook is therefore spending entries on magnitude *diversity* —
several codewords pointing the same way at different lengths — which
buys nothing in reconstruction and costs direction coverage. The fine
tier showed the consequence: perplexity 636 of K=16384 (3.9%).

Gain-shape splits the two:
  1. shape: pick the unit codeword maximizing COSINE (== inner product,
     since the book is on the sphere), so every entry buys a direction;
  2. gain: code the OPTIMAL projection gain g* = r . u_hat with a small
     scalar quantizer in the log domain (a learned 1-D book, EMA-fit).

Reconstruction is g_hat * u_hat, so the residual handed to the next RVQ
level is near-orthogonal to the chosen shape. Forcing g_hat = ||r||
instead of the projection gain would be WORSE than plain VQ by
2/(1+cos) in MSE — the projection gain is load-bearing, not cosmetic.

Spherical fitting also removes the collapse mechanism: in unconstrained
k-means an under-used codeword drifts toward the origin, where its short
norm makes it lose every distance comparison, so it stops receiving
assignments and dies. On the sphere there is no such attractor.

`georvq/model/vq.py` is untouched: existing checkpoints decode
bit-faithfully through the default classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.model.vq import ResidualVQ, VectorQuantizer

EPS = 1e-6


class GainShapeVQ(VectorQuantizer):
    """VQ with a unit-sphere shape codebook plus a scalar gain codebook.

    State-dict adds `gain_book` (2**gain_bits, log-domain gain levels) and
    `gain_count` (EMA occupancy); everything else matches the parent.
    """

    def __init__(self, *args, gain_bits=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain_bits = gain_bits
        n_g = 2 ** gain_bits
        # log-domain gain levels, initialized on first data sight
        self.register_buffer("gain_book", torch.zeros(n_g))
        self.register_buffer("gain_count", torch.zeros(n_g))
        self.register_buffer("gain_init", torch.tensor(False,
                                                       dtype=torch.bool))

    # ---- book maintenance ----
    @torch.no_grad()
    def normalize_book(self):
        """Project every codeword onto the unit sphere. Idempotent."""
        w = self.codebook.weight.data
        w.div_(w.norm(dim=1, keepdim=True).clamp_min(EPS))

    @torch.no_grad()
    def _data_init(self, z_flat):
        super()._data_init(z_flat)
        self.normalize_book()

    @torch.no_grad()
    def _reset_dead(self, z_flat):
        n = super()._reset_dead(z_flat)
        if n:
            self.normalize_book()
        return n

    @torch.no_grad()
    def init_gain_book(self, g):
        """Log-domain quantiles of the observed gains (Lloyd-Max seed)."""
        lg = g.clamp_min(EPS).log()
        qs = torch.linspace(0.5 / self.gain_book.numel(),
                            1 - 0.5 / self.gain_book.numel(),
                            self.gain_book.numel(), device=g.device)
        self.gain_book.copy_(torch.quantile(lg.float(), qs))
        self.gain_count.fill_(1.0)
        self.gain_init.fill_(True)

    def quantize_gain(self, g):
        """Nearest log-domain level. Returns (g_hat, index)."""
        lg = g.clamp_min(EPS).log()
        idx = (lg[:, None] - self.gain_book[None, :]).abs().argmin(-1)
        return self.gain_book[idx].exp(), idx

    # ---- forward ----
    def forward(self, z):
        """z: (..., D) -> (z_q straight-through, shape indices, commit).

        Gain indices are recoverable via `last_gain_idx` (the bitstream
        carries both; rate = log2(K) + gain_bits per cell per level)."""
        flat_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.dim)

        if self.training and not self.initialized.item():
            self._data_init(z_flat.detach())
        if self.training:
            self.normalize_book()

        E = self.codebook.weight
        # unit-norm book => argmax cosine == argmax inner product
        if self.bf16_scores:
            indices = (z_flat.bfloat16() @ E.t().bfloat16()).argmax(dim=-1)
        else:
            indices = (z_flat @ E.t()).argmax(dim=-1)
        u = E[indices]                                   # (N, D) unit shapes
        g_star = (z_flat * u).sum(-1)                    # projection gain

        if self.training and not self.gain_init.item():
            self.init_gain_book(g_star.detach())
        g_hat, g_idx = self.quantize_gain(g_star.detach())
        self.last_gain_idx = g_idx.reshape(flat_shape)
        z_q = g_hat.unsqueeze(-1) * u

        if self.training and not self.grad_codebook:
            with torch.no_grad():
                # spherical k-means EMA: mean of assigned vectors, then
                # re-normalized by normalize_book() on the next forward
                count = torch.bincount(indices,
                                       minlength=self.num_codes).to(z_flat.dtype)
                vec_sum = torch.zeros_like(self.ema_sum, dtype=z_flat.dtype) \
                    .index_add_(0, indices, z_flat)
                self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(vec_sum, alpha=1 - self.decay)
                n = self.ema_count.sum()
                cs = (self.ema_count + self.eps) \
                    / (n + self.num_codes * self.eps) * n
                self.codebook.weight.data.copy_(self.ema_sum / cs.unsqueeze(1))
                self.normalize_book()
                # 1-D Lloyd step on the gain levels (log domain, EMA)
                gc = torch.bincount(g_idx, minlength=self.gain_book.numel()) \
                    .to(g_star.dtype)
                gsum = torch.zeros_like(self.gain_book, dtype=g_star.dtype) \
                    .index_add_(0, g_idx, g_star.clamp_min(EPS).log())
                self.gain_count.mul_(self.decay).add_(gc, alpha=1 - self.decay)
                live = gc > 0
                new = gsum[live] / gc[live]
                self.gain_book[live] = (self.decay * self.gain_book[live]
                                        + (1 - self.decay) * new)
                self.step += 1
                if self.reset_interval > 0 and \
                        (self.step.item() % self.reset_interval == 0):
                    self._reset_dead(z_flat.detach())

        commit_loss = F.mse_loss(z_flat, z_q.detach())
        if self.grad_codebook:
            commit_loss = commit_loss + F.mse_loss(z_q, z_flat.detach())
        z_q_st = z_flat + (z_q - z_flat).detach()
        return (z_q_st.reshape(*flat_shape, self.dim),
                indices.reshape(flat_shape), commit_loss)

    def bits_per_code(self):
        """Index cost of one coded cell at this level."""
        import math
        return math.log2(self.num_codes) + self.gain_bits


class GainShapeResidualVQ(ResidualVQ):
    """ResidualVQ with GainShapeVQ levels. Same state-dict layout plus the
    per-level gain books."""

    def __init__(self, num_levels=8, num_codes=256, dim=128, decay=0.99,
                 kmeans_init=True, reset_interval=2500, dead_threshold=0.01,
                 grad_codebook=False, bf16_scores=False, gain_bits=5,
                 reset_mode="random"):
        nn.Module.__init__(self)
        self.num_levels = num_levels
        self.levels = nn.ModuleList([
            GainShapeVQ(num_codes, dim, decay,
                        kmeans_init=kmeans_init,
                        reset_interval=reset_interval,
                        dead_threshold=dead_threshold,
                        grad_codebook=grad_codebook,
                        bf16_scores=bf16_scores,
                        gain_bits=gain_bits,
                        reset_mode=reset_mode)
            for _ in range(num_levels)])

    @torch.no_grad()
    def spherize_books(self, z=None, iters=6, chunk=16384):
        """Post-k-means hook: run proper SPHERICAL k-means per level.

        Projecting an unconstrained k-means book onto the sphere is not a
        fair init — those centroids were fitted under a different
        objective, and the residual cascade behind them was produced by a
        different reconstruction rule. This re-fits each level with
        max-cosine assignment and normalized-mean updates, propagating
        residuals with the gain-shape reconstruction, then seeds the gain
        book and the EMA buffers consistently."""
        if z is None:
            for lev in self.levels:
                lev.normalize_book()
            return
        resid = z.reshape(-1, z.shape[-1]).float()
        if resid.shape[0] > 200_000:
            sel = torch.randperm(resid.shape[0], device=resid.device)[:200_000]
            resid = resid[sel]
        N = resid.shape[0]
        ones = torch.ones(N, device=resid.device)

        def assign(C):
            out = []
            for r in resid.split(chunk):
                out.append((r @ C.t()).argmax(-1))
            return torch.cat(out)

        for lev in self.levels:
            K = lev.num_codes
            C = resid[torch.randperm(N, device=resid.device)[:K]].clone()
            C = C / C.norm(dim=1, keepdim=True).clamp_min(EPS)
            for _ in range(iters):
                a = assign(C)
                sums = torch.zeros_like(C).index_add_(0, a, resid)
                cnt = torch.zeros(K, device=resid.device).index_add_(0, a, ones)
                live = cnt > 0
                C[live] = sums[live] / sums[live].norm(
                    dim=1, keepdim=True).clamp_min(EPS)
                n_dead = int((~live).sum())
                if n_dead:
                    rs = resid[torch.randint(0, N, (n_dead,),
                                             device=resid.device)]
                    C[~live] = rs / rs.norm(dim=1, keepdim=True).clamp_min(EPS)
            a = assign(C)
            cnt = torch.zeros(K, device=resid.device).index_add_(0, a, ones)
            lev.codebook.weight.data.copy_(C)
            lev.ema_count.copy_((cnt / N * 4096).clamp_min(
                2 * lev.dead_threshold))
            lev.ema_sum.copy_(C * lev.ema_count[:, None])
            lev.initialized.fill_(True)
            u = C[a]
            g = (resid * u).sum(-1)
            lev.init_gain_book(g)
            resid = resid - lev.quantize_gain(g)[0].unsqueeze(-1) * u
