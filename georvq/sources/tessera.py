"""Tessera embedding-field source: 128-band vectors stored as per-pixel
max-abs-quantized int8 chips (nodata = all-bands-zero).

Tessera tiles ship as int8 quantized embeddings plus a per-pixel float32
scale (dequant: x = q * scale). Max-abs quantization pins the largest
band of every valid pixel at exactly +-127, so the int8 vector is a
*reparametrization* of its unit direction: q = 127 * u / max_b|u_b|
(exact round trip, verified on the store). The unit sphere is therefore
the minimal coordinate system for q — `prep` normalizes and loses
nothing. Full original reconstruction is x_hat = s * 127 * u_hat /
max_b|u_hat_b|, where the scale plane s rides along as side information
(float16 sidecars in the store; part of the payload rate accounting, not
of the model). `vec_rel_l2` in recon_stats measures exactly this
end-to-end error (s cancels in the relative norm).

Stem is the identity and the head is a dense plane head with per-pixel L2
normalization, exactly as for AlphaEarth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from georvq.metrics import masked_recon_stats
from georvq.model.decoder import DenseHead
from georvq.sources import Source


class TesseraSource(Source):
    name = "tessera"
    # 10 m native output (no upsampled lattice): trivial period-1 grid
    grid_period = 1

    def prep(self, raw, device):
        """int8 (B, 128, W, W) -> (x, valid). Keeps the first `bands`
        channels, re-normalized to unit vectors; nodata (all-zero vector)
        pixels stay zero and are masked out."""
        s = raw.to(device, non_blocking=True)
        valid = s.ne(0).any(dim=1)
        x = F.normalize(s[:, :self.cfg.bands].float(), dim=1, eps=1e-6)
        return x, valid

    def build_stem(self, model_cfg) -> nn.Module:
        return nn.Identity()

    def build_head(self, model_cfg, in_dim) -> nn.Module:
        fine = model_cfg.tiers[-1]
        return DenseHead(in_dim, grid=fine.grid,
                         p=model_cfg.window // fine.grid,
                         bands=model_cfg.bands,
                         head_feat=model_cfg.decoder.head_feat,
                         normalize=True)

    def recon_stats(self, out, x, valid):
        mse, cos = masked_recon_stats(out, x, valid)
        with torch.no_grad():
            # End-to-end fidelity of the dequantized original x = q*s: map
            # both unit fields back to int8 space via q = 127*u/max|u_b|
            # (exact for the true field; see module docstring) and take the
            # relative L2 error — the per-pixel scale s cancels.
            q_t = x / x.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
            q_o = out / out.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
            num = (q_o - q_t).pow(2).sum(1).sqrt()
            den = q_t.pow(2).sum(1).sqrt().clamp_min(1e-6)
            rel = (num / den)[valid].mean()
        return mse, {"mse": mse, "cos": cos, "vec_rel_l2": rel}
