"""Tessera source adapter + store-builder units (no network)."""
import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "build_store_tessera",
    os.path.join(os.path.dirname(__file__), "..", "scripts",
                 "build_store_tessera.py"))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_source_registered():
    from georvq.sources import get_source
    src = get_source("tessera", SimpleNamespace(bands=128))
    assert src.name == "tessera" and src.grid_period == 1


def test_prep_normalizes_and_masks():
    from georvq.sources.tessera import TesseraSource
    src = TesseraSource(SimpleNamespace(bands=16))
    rng = np.random.default_rng(0)
    raw = torch.from_numpy(
        rng.integers(-127, 128, size=(2, 128, 8, 8), dtype=np.int8))
    raw[0, :, :3, :3] = 0                     # nodata: all-bands-zero
    x, valid = src.prep(raw, "cpu")
    assert x.shape == (2, 16, 8, 8) and valid.shape == (2, 8, 8)
    assert not valid[0, :3, :3].any() and valid[1].all()
    norms = x.norm(dim=1)
    assert torch.allclose(norms[valid], torch.ones(1), atol=1e-4)
    assert (norms[~valid] == 0).all()


def test_prep_scale_invariance():
    """Unit direction from raw int8 must match dequantized-then-normalized."""
    from georvq.sources.tessera import TesseraSource
    src = TesseraSource(SimpleNamespace(bands=128))
    rng = np.random.default_rng(1)
    raw = torch.from_numpy(
        rng.integers(-127, 128, size=(1, 128, 4, 4), dtype=np.int8))
    scale = torch.rand(1, 1, 4, 4) * 0.1 + 0.01
    x, _ = src.prep(raw, "cpu")
    ref = torch.nn.functional.normalize(raw.float() * scale, dim=1)
    assert torch.allclose(x, ref, atol=1e-5)


def test_direction_reparametrizes_int8_exactly():
    """Max-abs quantization pins max|q_b| at 127, so q = 127*u/max|u_b| is
    an exact round trip — unit-norming discards nothing of the original."""
    from georvq.sources.tessera import TesseraSource
    src = TesseraSource(SimpleNamespace(bands=128))
    rng = np.random.default_rng(2)
    q = rng.integers(-126, 127, size=(1, 128, 6, 6)).astype(np.float64)
    q[0, rng.integers(0, 128, size=(6, 6)), np.arange(6)[:, None],
      np.arange(6)[None, :]] = 127          # pin one band per pixel
    x, _ = src.prep(torch.from_numpy(q.astype(np.int8)), "cpu")
    q_rec = 127 * x.double() / x.double().abs().amax(dim=1, keepdim=True)
    assert torch.allclose(q_rec, torch.from_numpy(q), atol=1e-3)


def test_recon_stats_vec_rel_l2():
    """Perfect recon -> 0; sign-flipped max band -> order-1 error."""
    from georvq.sources.tessera import TesseraSource
    src = TesseraSource(SimpleNamespace(bands=8))
    x = torch.nn.functional.normalize(torch.randn(1, 8, 4, 4), dim=1)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    _, stats = src.recon_stats(x, x, valid)
    assert stats["vec_rel_l2"] < 1e-6
    out = x.clone()
    out[0, 0] *= -1
    _, stats = src.recon_stats(out, x, valid)
    assert stats["vec_rel_l2"] > 0.1


def test_tile_key_snapping():
    # geotessera docstring example: (-120, 36) snaps to (-119.95, 36.05)
    assert builder.tile_key(-120.0, 36.0) == (-11995, 3605)
    assert builder.tile_key(0.11, 52.09) == (15, 5205)
    assert builder.tile_key(-0.01, -0.01) == (-5, -5)


class _FixedPointRng:
    """Stub rng whose uniform draws always map to one fixed lon/lat."""

    def __init__(self, lon, lat):
        self.lon, self.sinlat = lon, np.sin(np.radians(lat))

    def uniform(self, lo, hi):
        return self.lon if lo == -180 else self.sinlat

    def choice(self, seq):
        return seq[0]


def test_sampler_membership_and_dedup():
    available = [(2024, 0.15, 52.05), (2023, 0.15, 52.05)]
    s = builder.Sampler(available, seed=0)
    s.rng = _FixedPointRng(0.11, 52.09)       # inside tile (0.15, 52.05)
    t = s.draw()
    assert t == (2024, 0.15, 52.05)
    assert s.draw() is None                   # dedup: same (year, tile)
    s.rng = _FixedPointRng(10.0, 10.0)        # no coverage there
    assert s.draw() is None


def test_utm_epsg():
    assert builder.utm_epsg(0.15, 52.05) == "EPSG:32631"
    assert builder.utm_epsg(-119.95, 36.05) == "EPSG:32611"
    assert builder.utm_epsg(147.05, -42.15) == "EPSG:32755"


def test_scale_sidecar_layout(tmp_path):
    sc = builder.ScaleSidecar(str(tmp_path), shard_size=4, px=8)
    planes = [np.full((8, 8), i / 10, np.float16) for i in range(6)]
    for i, p in enumerate(planes):
        sc.write(i, p)
    s0 = np.load(tmp_path / "scales_0000.npy")
    s1 = np.load(tmp_path / "scales_0001.npy")
    assert s0.shape == (4, 8, 8) and s0.dtype == np.float16
    assert (s0[3] == planes[3]).all() and (s1[1] == planes[5]).all()


def test_limit_chips_nested_subsample(tmp_path):
    """data.limit_chips trains on a deterministic nested subset; val full."""
    from georvq.config import load_config
    from georvq.data.build import build_synthetic_store
    from georvq.sources import get_source
    from georvq.train.loop import setup_data
    build_synthetic_store(tmp_path / "store", n_chips=12, bands=8,
                          chip_px=64, shard_size=4)
    base = [f"data.store_dir={tmp_path / 'store'}"]

    def train_ids(limit):
        cfg = load_config("configs/smoke.yaml",
                          base + [f"data.limit_chips={limit}"])
        src = get_source(cfg.data.source, cfg.model)
        _, train_rows, val_rows, _, pool = setup_data(cfg, src, "cpu")
        pool.shutdown()
        return set(train_rows.chip_id), len(val_rows)

    ids3, nval3 = train_ids(3)
    ids6, nval6 = train_ids(6)
    assert len(ids3) == 3 and len(ids6) == 6
    assert ids3 < ids6                        # nested for the same split_seed
    assert nval3 == nval6                     # val split untouched


@pytest.mark.slow
def test_tessera_training_smoke(tmp_path):
    """End-to-end pyramid smoke on a synthetic 128-band tessera store."""
    from georvq.config import load_config
    from georvq.data.build import build_synthetic_store
    from georvq.train.loop import train
    build_synthetic_store(tmp_path / "store", n_chips=8, bands=128,
                          chip_px=64, shard_size=4, nodata=0, grid_period=1)
    cfg = load_config("configs/smoke.yaml",
                      [f"data.store_dir={tmp_path / 'store'}",
                       f"log.out_dir={tmp_path / 'ckpt'}",
                       "data.source=tessera", "model.bands=128",
                       "train.steps=10"])
    run_dir = train(cfg, gpu=-1)
    assert os.path.exists(os.path.join(run_dir, "model.pt"))
