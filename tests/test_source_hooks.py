"""Source hooks that every adapter must honour the same way."""
import torch

from georvq.config import load_config
from georvq.sources import get_source


def test_dem_charges_its_mean_and_std():
    """DEM removes a per-window mean and std before encoding and needs
    both to restore metres, so they are part of every rate figure."""
    cfg = load_config("configs/smoke.yaml", ["model.bands=1"])
    src = get_source("dem", cfg.model)
    assert src.normalizer(cfg.model) is not None
    assert src.side_bits(cfg.model) == 64.0


def test_every_source_head_accepts_a_tier_prefix():
    """A decoder-bank member builds its head over a prefix of the tiers at
    a smaller output size; an adapter with the old two-argument
    build_head would fail as soon as a bank is configured."""
    cfg = load_config("configs/smoke.yaml", ["model.bands=1"])
    m = cfg.model
    n = 1
    grid = m.tiers[n - 1].grid
    out_px = m.window // 2
    for name in ("ae", "tessera", "dem", "era5:t"):
        bands = {"ae": 64, "era5:t": 37}.get(name, 1)
        mc = m.model_copy(update={"bands": bands})
        src = get_source(name, mc)
        head = src.build_head(mc, 32, n_tiers=n, out_px=out_px)
        y = head(torch.zeros(2, grid * grid, 32))
        y = y[0] if isinstance(y, (tuple, list)) else y
        assert y.shape[-1] == out_px, (name, tuple(y.shape))
