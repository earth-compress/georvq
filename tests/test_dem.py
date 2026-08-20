import numpy as np
import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.sources import get_source
from georvq.sources.dem import NODATA, STD_FLOOR, DEMNormalizer


@pytest.fixture
def dem_cfg(tiny_raw_cfg):
    raw = dict(tiny_raw_cfg)
    raw["data"] = {"source": "dem", "store_dir": "/tmp/georvq-nonexistent"}
    raw["model"] = dict(raw["model"], bands=1)
    return Config.model_validate(raw)


def test_prep_masked_standardization(dem_cfg):
    src = get_source("dem", dem_cfg.model)
    raw = torch.rand(4, 1, 16, 16) * 500 + 1200      # metres
    raw[0, 0, :4, :4] = NODATA
    raw[3] = 137.0                                    # flat window
    x, valid = src.prep(raw.clone(), "cpu")

    assert valid.dtype == torch.bool and valid.shape == (4, 16, 16)
    assert not valid[0, :4, :4].any() and valid[1].all()
    # nodata pixels are zeroed, never leak into the plane
    assert (x[0, 0, :4, :4] == 0).all()
    # valid pixels are standardized on the masked stats
    v = x[0, 0][valid[0]]
    assert abs(v.mean()) < 1e-4 and abs(v.std(correction=0) - 1) < 1e-3
    # flat window: std floored, residual exactly zero
    mu, sd = src.last_stats()
    assert sd[3, 0] == STD_FLOOR and (x[3] == 0).all()
    assert mu[3, 0] == pytest.approx(137.0)


def test_quality_penalizes_overshoot(dem_cfg):
    """R^2, unlike the raw channel dot product, must not reward amplitude
    overshoot (the unit-norm cosine reads > 1 there)."""
    src = get_source("dem", dem_cfg.model)
    q = src.report(dem_cfg.model).quality
    x = torch.randn(4, 1, 16, 16)
    valid = torch.ones(4, 16, 16, dtype=torch.bool)
    exact = q(x, x, valid)
    over = q(1.5 * x, x, valid)
    assert exact == pytest.approx(1.0)
    assert over < exact and over <= 1.0


def test_dem_reports_r2_not_cosine(dem_cfg):
    """The bug this branch carried: DEM published R^2 under the key "cos"
    so the shared logging path would accept it. It now says r2, and the
    band-count gate would refuse a cosine here anyway."""
    from georvq.sources.report import resolve_report

    src = get_source("dem", dem_cfg.model)
    spec = resolve_report(src, dem_cfg)
    assert spec.quality.key == "r2"
    assert [m.key for m in spec.extra] == ["rmse"]
    assert spec.baselines == ()          # the mean is a transmitted channel
    assert spec.preview.key == "gray"    # PCA-RGB cannot draw one band
    assert spec.has_figure("error_hist") and spec.units == "metres"
    _, m = src.recon_stats(torch.zeros(2, 1, 16, 16),
                           torch.zeros(2, 1, 16, 16),
                           torch.ones(2, 16, 16, dtype=torch.bool))
    assert "cos" not in m and "r2" in m


def test_normalizer_roundtrip(dem_cfg):
    norm = DEMNormalizer()
    x = torch.rand(2, 1, 16, 16) * 3000 - 200
    valid = torch.ones(2, 16, 16, dtype=torch.bool)
    xn, stats = norm.normalize(x, valid)
    assert torch.allclose(norm.denormalize(xn, stats), x, atol=1e-3)
    assert norm.stats_vector(stats).shape == (2, 2)


def test_dem_model_roundtrip(dem_cfg):
    torch.manual_seed(0)
    model = build_model(dem_cfg, torch.device("cpu"))
    src = model.source
    raw = torch.rand(2, 1, 16, 16) * 500
    x, valid = src.prep(raw, "cpu")
    zs = model.encode_levels(x)
    Ls = torch.tensor([[sp.depth for sp in model.specs]] * 2)
    zqs, commit, _ = model.quantize(zs, Ls)
    out = model.decode(zqs, Ls).float()
    assert out.shape == x.shape
    loss, m = src.recon_stats(out, x, valid)
    assert torch.isfinite(loss) and {"mse", "r2"} <= set(m)


def test_dem_figures_draw_under_the_declared_surface(dem_cfg):
    """Both source-specific figures must actually render for one band.

    The recon panel is the sharper case: PCA-RGB takes three components of
    a 1x1 SVD and hands imshow an (H, W, 1) array, so before the source
    declared a grayscale preview this raised rather than plotting.
    """
    import matplotlib.pyplot as plt

    from georvq.sources.report import resolve_report
    from georvq.train.figures import error_hist_figure, recon_figure

    torch.manual_seed(0)
    model = build_model(dem_cfg, torch.device("cpu"))
    spec = resolve_report(model.source, dem_cfg)
    val_s = torch.rand(8, 1, 16, 16) * 500 + 1000

    f = recon_figure(model, model.source, val_s, step=1, n=2,
                     preview=spec.preview)
    plt.close(f)

    # the second panel is the point of this figure for DEM: signed error
    # in metres, via the normalizer
    f2 = error_hist_figure(model, model.source, val_s, step=1,
                           max_tokens=8, chunk=4, unit=spec.units)
    titles = [ax.get_title() for ax in f2.axes]
    assert titles == ["model space (per-tile normalized)", "metres"]
    plt.close(f2)
