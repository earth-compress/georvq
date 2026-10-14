"""Golden tests for the sampler and the Prep seam: the defaults must
reproduce the legacy pool.sample_batch / Source.prep path exactly."""
import pytest
import torch

from georvq.config import load_config
from georvq.data.build import build_synthetic_store
from georvq.data.pool import ChipPool
from georvq.data.sampler import WindowSampler
from georvq.factory import build_model
from georvq.sources.prep import SourcePrep, build_prep


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    d = tmp_path_factory.mktemp("store")
    build_synthetic_store(d / "s", n_chips=8, bands=8, chip_px=64,
                          shard_size=4)
    cfg = load_config("configs/smoke.yaml",
                      [f"data.store_dir={d / 's'}"])
    model = build_model(cfg, "cpu")
    source = model.source
    from georvq.data.store import open_store
    store = open_store(str(d / "s"))
    rows, _ = store.split(val_frac=0.25)
    pool = ChipPool(store, rows, pool_size=8, device=torch.device("cpu"),
                    workers=0, seed=0)
    yield cfg, source, pool
    pool.shutdown()


def test_window_sampler_matches_legacy_sample_batch(rig):
    cfg, source, pool = rig
    s = WindowSampler(pool, cfg.model)
    torch.manual_seed(7)
    a, meta = s.draw(4)
    torch.manual_seed(7)
    b = pool.sample_batch(4, cfg.model.window)
    assert torch.equal(a, b) and meta == {}
    torch.manual_seed(9)
    a, meta = s.draw(4, return_rows=True)
    torch.manual_seed(9)
    b, rows = pool.sample_batch(4, cfg.model.window, return_rows=True)
    assert torch.equal(a, b) and torch.equal(meta["row"], rows)


def test_source_prep_matches_legacy_prep(rig):
    cfg, source, pool = rig
    p = build_prep(None, source, cfg.model)
    assert isinstance(p, SourcePrep)
    raw = pool.sample_batch(4, cfg.model.window)
    x0, v0 = source.prep(raw.clone(), "cpu")
    x1, v1, meta = p(raw.clone(), {"k": 1}, "cpu")
    assert torch.equal(x0, x1) and torch.equal(v0, v1)
    assert meta == {"k": 1}          # passed through untouched


def test_unknown_prep_names_fail_loud(rig):
    cfg, source, pool = rig
    with pytest.raises(KeyError):
        build_prep("no_such", source, cfg.model)


def test_config_accepts_a_prep_block(tmp_path):
    cfg = load_config("configs/smoke.yaml", [
        "data.store_dir=/tmp/x",
        "data.prep={name: null}"])
    assert cfg.data.prep.name is None


def test_era5_boxcox_named_prep_matches_legacy_grammar():
    import torch
    from georvq.config import load_config
    from georvq.sources import get_source
    cfg = load_config("configs/era5/era5_q64bc_ladder.yaml")
    assert cfg.data.prep.name == "boxcox"
    src = get_source(cfg.data.source, cfg.model)
    p = build_prep("boxcox", src, cfg.model)
    legacy = get_source("era5:q+boxcox", cfg.model)
    raw = torch.rand(2, 37, 8, 8) * 2e-2 + 1e-7
    x_new, v, meta = p(raw.clone(), {}, "cpu")
    x_old, _ = legacy.prep(raw.clone(), "cpu")
    assert torch.equal(x_new, x_old) and meta == {}
    back = p.inverse(x_new, {})
    assert float((back - raw.clamp_min(0)).abs().max()) < 1e-6
    st = p.state()["boxcox"]
    assert len(st["lam"]) == 37 and st["eps"] > 0


def test_physical_metric_routes_through_prep_inverse():
    import torch
    from georvq.sources.report import Metric
    from georvq.train.validate import validate
    # a run's rmse_phys must equal masked_rmse on prep-inverted pairs;
    # checked at the unit level against the era5 SourcePrep
    from georvq.config import load_config
    from georvq.sources import get_source
    from georvq.metrics import masked_rmse
    cfg = load_config("configs/era5/era5_q64_ladder.yaml")
    src = get_source(cfg.data.source, cfg.model)
    p = build_prep(None, src, cfg.model)
    raw = torch.rand(2, 37, 8, 8) * 2e-2 + 1e-7
    x, v, _ = p(raw, {}, "cpu")
    o = x + 0.1 * torch.randn_like(x)
    spec = src.report(cfg.model)
    phys = [m for m in spec.extra if m.key == "rmse_phys"][0]
    assert phys.space == "physical"
    want = masked_rmse(p.inverse(o, {}), p.inverse(x, {}), v)
    assert torch.allclose(phys(p.inverse(o, {}), p.inverse(x, {}), v), want)


def test_params_reach_the_constructor(rig):
    """data.prep.params is passed on, so a prep that takes none rejects
    it instead of ignoring it."""
    cfg, source, pool = rig
    with pytest.raises(ValueError, match="data.prep.params"):
        build_prep(None, source, cfg.model, params="fitted_v1")

    class _Prep(SourcePrep):
        def __init__(self, source, model_cfg, params=None):
            super().__init__(source, model_cfg)
            self.params = params

    source.named_preps = lambda: {"fitted": _Prep}
    try:
        assert build_prep("fitted", source, cfg.model,
                          params="v1").params == "v1"
    finally:
        del source.named_preps
