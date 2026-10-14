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


def _pack_raw(q, scale):
    """(B,128,H,W) int8 + (B,H,W) float16 -> (B,130,H,W) int8 raw, the
    scale plane's bytes as two trailing pseudo-bands (the pool layout)."""
    B, _, H, W = q.shape
    sb = scale.astype(np.float16).view(np.int8).reshape(B, H, W, 2) \
              .transpose(0, 3, 1, 2)
    return torch.from_numpy(np.concatenate([q, sb], axis=1))


def test_prep_dequantizes_and_masks():
    from georvq.sources.tessera import TesseraSource
    src = TesseraSource(SimpleNamespace(bands=128))
    rng = np.random.default_rng(0)
    q = rng.integers(-127, 128, size=(2, 128, 8, 8), dtype=np.int8)
    scale = rng.uniform(0.03, 0.11, size=(2, 8, 8)).astype(np.float16)
    q[0, :, :3, :3] = 0                       # nodata: all-bands-zero
    scale[1, 0, 0] = 0.0                      # nodata: zero scale
    x, valid = src.prep(_pack_raw(q, scale), "cpu")
    assert x.shape == (2, 128, 8, 8) and valid.shape == (2, 8, 8)
    assert not valid[0, :3, :3].any() and not valid[1, 0, 0]
    ref = torch.from_numpy(
        q.astype(np.float32) * scale.astype(np.float32)[:, None]
        / TesseraSource.XNORM)
    assert torch.allclose(x[valid.unsqueeze(1).expand_as(x)],
                          ref[valid.unsqueeze(1).expand_as(ref)], atol=1e-5)
    assert (x[0, :, :3, :3] == 0).all() and (x[1, :, 0, 0] == 0).all()


def test_prep_masks_corrupt_scales():
    """inf / nan / absurd fp16 scales (rare corrupt source pixels) must be
    masked as nodata and never reach x — one inf NaNs a raw-space run."""
    from georvq.sources.tessera import TesseraSource
    src = TesseraSource(SimpleNamespace(bands=128))
    rng = np.random.default_rng(3)
    q = rng.integers(1, 128, size=(1, 128, 4, 4), dtype=np.int8)
    scale = np.full((1, 4, 4), 0.06, np.float16)
    scale[0, 0, 0] = np.inf
    scale[0, 0, 1] = np.nan
    scale[0, 0, 2] = 3000.0        # finite but far beyond SCALE_MAX
    x, valid = src.prep(_pack_raw(q, scale), "cpu")
    assert torch.isfinite(x).all()
    assert not valid[0, 0, 0] and not valid[0, 0, 1] and not valid[0, 0, 2]
    assert (x[0, :, 0, :3] == 0).all() and valid[0, 1:].all()


def test_prep_rejects_missing_scales():
    from georvq.sources.tessera import TesseraSource
    src = TesseraSource(SimpleNamespace(bands=128))
    raw = torch.zeros(1, 128, 4, 4, dtype=torch.int8)
    with pytest.raises(AssertionError, match="scale"):
        src.prep(raw, "cpu")


def test_recon_stats_raw_space():
    """Loss is raw-space MSE; cos is on-the-fly (scale-blind); rel_l2 is
    not — a uniformly rescaled recon keeps cos=1 but pays in mse/rel."""
    from georvq.sources.tessera import TesseraSource
    src = TesseraSource(SimpleNamespace(bands=8))
    x = torch.randn(1, 8, 4, 4)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    mse, stats = src.recon_stats(x, x, valid)
    assert mse < 1e-10 and stats["vec_rel_l2"] < 1e-5
    assert abs(stats["cos"] - 1.0) < 1e-5
    mse, stats = src.recon_stats(2 * x, x, valid)
    assert abs(stats["cos"] - 1.0) < 1e-5     # direction untouched
    assert abs(stats["vec_rel_l2"] - 1.0) < 1e-5   # magnitude off by 100%
    assert mse > 0.1


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


def _write_scale_sidecars(store_dir, n_chips, shard_size, chip_px, seed=7):
    """Random positive fp16 scale planes matching the store's shard/slot
    layout, as the tessera builder would have written them."""
    rng = np.random.default_rng(seed)
    n_shards = (n_chips + shard_size - 1) // shard_size
    for sh in range(n_shards):
        s = rng.uniform(0.03, 0.11,
                        size=(shard_size, chip_px, chip_px)).astype(np.float16)
        np.save(os.path.join(store_dir, f"scales_{sh:04d}.npy"), s)


def test_store_aux_planes_reach_prep(tmp_path):
    """Scale sidecars ride through ChipStore -> pool / val tokens as two
    trailing byte-bands and dequantize correctly in prep."""
    from georvq.data.build import build_synthetic_store
    from georvq.data.pool import ChipPool, fixed_val_tokens
    from georvq.data.store import ChipStore
    from georvq.sources.tessera import TesseraSource

    d = str(tmp_path / "store")
    build_synthetic_store(d, n_chips=6, bands=128, chip_px=32,
                          shard_size=4, nodata=0, grid_period=1)
    _write_scale_sidecars(d, n_chips=6, shard_size=4, chip_px=32)
    store = ChipStore(d)
    assert store.aux_bands == 2

    # pool path
    pool = ChipPool(store, store.index, pool_size=6,
                    device=torch.device("cpu"), workers=0)
    batch = pool.sample_batch(4, token=16)
    pool.shutdown()
    assert batch.shape == (4, 130, 16, 16)

    # val-token path + end-to-end dequant check on a full chip
    toks = fixed_val_tokens(store, store.index, token=16, per_chip=4)
    assert toks.shape[1] == 130
    src = TesseraSource(SimpleNamespace(bands=128))
    x, valid = src.prep(toks[:2], "cpu")
    q = toks[:2, :128].float()
    s = np.load(os.path.join(d, "scales_0000.npy"))
    # crop grid starts at (0,0): token 0 of chip 0 is its top-left corner
    ref = q[0] * torch.from_numpy(
        s[0, :16, :16].astype(np.float32)) / TesseraSource.XNORM
    assert torch.allclose(x[0][:, valid[0]], ref[:, valid[0]], atol=1e-5)


def test_tessera_training_smoke(tmp_path):
    """End-to-end pyramid smoke on a synthetic 128-band tessera store."""
    from georvq.config import load_config
    from georvq.data.build import build_synthetic_store
    from georvq.train.loop import train
    build_synthetic_store(tmp_path / "store", n_chips=8, bands=128,
                          chip_px=64, shard_size=4, nodata=0, grid_period=1)
    _write_scale_sidecars(str(tmp_path / "store"), n_chips=8, shard_size=4,
                          chip_px=64)
    cfg = load_config("configs/smoke.yaml",
                      [f"data.store_dir={tmp_path / 'store'}",
                       f"log.out_dir={tmp_path / 'ckpt'}",
                       "data.source=tessera", "model.bands=128",
                       "train.steps=10"])
    run_dir = train(cfg, gpu=-1)
    assert os.path.exists(os.path.join(run_dir, "model.pt"))
