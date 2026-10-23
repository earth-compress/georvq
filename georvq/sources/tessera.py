"""Tessera embedding-field source: 128-band vectors stored as per-pixel
max-abs-quantized int8 chips (nodata = all-bands-zero) plus float16
per-pixel dequant scales in scales_XXXX.npy sidecars.

The model works in ORIGINAL dequantized units: the store surfaces each
chip's scale plane as two trailing byte-bands (ChipStore.aux_bytes) and
`prep` reassembles x = q * scale / XNORM on the GPU. XNORM is a single
global constant (store median ||x|| is ~31, p1-p99 spread only 1.5x) that
puts the loss at O(1); it rescales units uniformly and never reweights
pixels. The training loss is masked MSE on these vectors; cosine is
computed on the fly (normalized copies) as a metric only, alongside
`vec_rel_l2`, the per-pixel relative L2 error of the reconstruction.

Stem is the identity; the head is a dense plane head WITHOUT the unit
normalization used for AlphaEarth — outputs are raw vector fields.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from georvq.model.decoder import DenseHead
from georvq.sources import Source


class TesseraSource(Source):
    name = "tessera"
    # 10 m native output (no upsampled lattice): trivial period-1 grid
    grid_period = 1
    # global loss-conditioning constant (~median ||x|| of the store);
    # part of the model-space definition, so changing it invalidates
    # trained checkpoints
    XNORM = 32.0
    # scales above this are corrupt source pixels, masked as nodata: real
    # chips top out ~0.15 (median 0.065) while the anomalies — a few per
    # billion pixels, incl. fp16 inf/nan — start ~2.9. One inf reaching
    # the raw-space loss NaNs the whole run.
    SCALE_MAX = 1.0

    def prep(self, raw, device):
        """int8 (B, bands+2, W, W) -> (x, valid). The last two bands are
        the float16 scale plane's bytes; x = q * scale / XNORM in original
        (rescaled) units. nodata (all-zero vector / zero scale) pixels
        stay zero and are masked out."""
        assert raw.shape[1] == 130, \
            "tessera raw must be 128 data bands + 2 scale bytes — is the " \
            "store missing its scales_XXXX.npy sidecars?"
        s8 = raw.to(device, non_blocking=True)
        q = s8[:, :self.cfg.bands].float()
        scale = s8[:, -2:].permute(0, 2, 3, 1).contiguous() \
                          .view(torch.float16).squeeze(-1).float()
        valid = (s8[:, :-2].ne(0).any(dim=1) & (scale > 0)
                 & (scale < self.SCALE_MAX))
        # torch.where (not multiply): inf/nan scales must not touch x
        x = q * torch.where(valid, scale / self.XNORM,
                            torch.zeros((), device=scale.device)).unsqueeze(1)
        return x, valid

    def build_stem(self, model_cfg) -> nn.Module:
        return nn.Identity()

    def build_head(self, model_cfg, in_dim, n_tiers=None,
                   out_px=None) -> nn.Module:
        """Dense pixel head WITHOUT unit normalization: Tessera vectors carry
        real magnitude (p1-p99 spread 1.57x), so normalizing the output
        would discard a component of the signal. Compare
        `AESource.build_head`, whose field is unit-norm by construction.
        """
        grid, p = self.head_geom(model_cfg, n_tiers, out_px)
        return DenseHead(in_dim, grid=grid, p=p,
                         bands=model_cfg.bands,
                         head_feat=model_cfg.decoder.head_feat,
                         normalize=False)

    def recon_stats(self, out, x, valid):
        """Loss: masked per-pixel MSE in original (rescaled) units.
        Metric: relative L2 of the full vectors."""
        se = (out - x).pow(2).sum(1)
        mse = se[valid].mean()
        with torch.no_grad():
            rel = (se.sqrt()
                   / x.pow(2).sum(1).sqrt().clamp_min(1e-6))[valid].mean()
        return mse, {"mse": mse, "vec_rel_l2": rel}

    def report(self, model_cfg):
        """Relative L2 of the pixel vectors, and nothing based on cosine.
        Tessera vectors carry real magnitude (spread 1.57x), so cosine
        ignores part of the error, and the shared `cos` metric is a dot
        product that only equals cosine on unit-norm fields. The zero-rate
        floor is the window mean the stats channel sends."""
        from dataclasses import replace
        from georvq.sources.report import (WINDOW_MEAN, Metric,
                                           embedding_field_spec)
        rel = Metric("vec_rel_l2", "relative L2", _vec_rel_l2,
                     higher_better=False, bounded=False, min_bands=2)
        return replace(embedding_field_spec(), quality=rel, extra=(),
                       baselines=(WINDOW_MEAN,))


def _vec_rel_l2(out, x, valid):
    se = (out - x).pow(2).sum(1)
    return (se.sqrt() / x.pow(2).sum(1).sqrt().clamp_min(1e-6))[valid].mean()
