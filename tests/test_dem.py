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
    x = torch.randn(4, 1, 16, 16)
    valid = torch.ones(4, 16, 16, dtype=torch.bool)
    exact = src.quality(x, x, valid)
    over = src.quality(1.5 * x, x, valid)
    assert exact == pytest.approx(1.0)
    assert over < exact and over <= 1.0


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
    assert torch.isfinite(loss) and {"mse", "cos"} <= set(m)
