"""Validity-masked reconstruction metrics for dense embedding fields."""
from __future__ import annotations


def masked_recon_stats(out, x, valid):
    """out, x: (B,C,t,t) unit-norm fields; valid: (B,t,t) bool.
    Returns (mse_per_unit_pixel, cosine) over valid pixels."""
    mse = ((out - x) ** 2).sum(1)[valid].mean()
    cos = (out * x).sum(1)[valid].mean()
    return mse, cos


def centered_cos(out, x, valid, eps=1e-8):
    """Spatial-structure probe: per-pixel cosine after subtracting each
    token's own valid-pixel mean from both sides. A constant (block-mean)
    reconstruction scores ~0 here regardless of its plain cosine."""
    vm = valid.unsqueeze(1).float()
    n = vm.sum((2, 3)).clamp_min(1)
    xc = x - ((x * vm).sum((2, 3)) / n)[:, :, None, None]
    oc = out - ((out * vm).sum((2, 3)) / n)[:, :, None, None]
    num = (xc * oc).sum(1)
    den = xc.norm(dim=1) * oc.norm(dim=1) + eps
    return (num / den)[valid].mean()


def masked_r2(out, x, valid, eps=1e-8):
    """Variance explained on valid pixels, for single-plane scalar fields
    (elevation, backscatter). Cosine is degenerate on one band -- it
    collapses to sign agreement -- so a scalar source's quality metric has
    to come from the residual against the field's own variance instead.

    Every band is pooled around one grand mean. On a multi-band field
    whose band means differ, the between-band spread counts as explained
    variance, so the score reads higher than a per-band R^2 would."""
    o = out[valid.unsqueeze(1).expand_as(out)]
    t = x[valid.unsqueeze(1).expand_as(x)]
    ss_res = (t - o).pow(2).sum()
    ss_tot = (t - t.mean()).pow(2).sum()
    return 1.0 - ss_res / ss_tot.clamp_min(eps)


def masked_rmse(out, x, valid):
    """Root mean squared error over valid pixels, in model-space units."""
    m = valid.unsqueeze(1).expand_as(out)
    return (out[m] - x[m]).pow(2).mean().sqrt()


def spectral_angle(out, x, valid, eps=1e-8):
    """Mean spectral angle (radians) between reconstructed and true pixel
    spectra. The standard multispectral fidelity measure: it ignores a
    per-pixel gain, so it reads the band RATIOS that spectral indices and
    classifiers actually use, which a per-band MSE averages away."""
    num = (out * x).sum(1)
    den = out.norm(dim=1) * x.norm(dim=1) + eps
    return (num / den).clamp(-1.0, 1.0).arccos()[valid].mean()
