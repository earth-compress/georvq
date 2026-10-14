import numpy as np
import pandas as pd
import pytest
import torch

from georvq.data.build import ShardWriter, build_synthetic_store
from georvq.data.pool import ChipPool, fixed_val_tokens
from georvq.data.store import ChipStore, geographic_split


@pytest.fixture
def store(tmp_path):
    build_synthetic_store(tmp_path / "store", n_chips=10, bands=8,
                          chip_px=64, shard_size=4)
    return ChipStore(tmp_path / "store")


def test_store_reads_manifest(store):
    assert store.bands == 8 and store.chip_px == 64
    assert store.shard_size == 4 and store.dtype == np.int8
    assert store.nodata == -128 and store.grid_period == 2
    assert len(store.index) == 10
    assert store.open_shard(0).shape == (4, 8, 64, 64)


def test_store_infers_without_manifest(store, tmp_path):
    (tmp_path / "store" / "manifest.json").unlink()
    s2 = ChipStore(tmp_path / "store")
    assert (s2.bands, s2.chip_px, s2.shard_size) == (8, 64, 4)
    assert s2.nodata is None and s2.grid_period is None


def test_store_filters_missing_shards(store, tmp_path):
    (tmp_path / "store" / "shard_0001.npy").unlink()
    s2 = ChipStore(tmp_path / "store")
    assert len(s2.index) == 6            # 10 - shard of 4
    assert not (s2.index.shard == 1).any()


def test_geographic_split_deterministic(store):
    tr1, va1 = geographic_split(store.index, val_frac=0.3, seed=0)
    tr2, va2 = geographic_split(store.index, val_frac=0.3, seed=0)
    assert tr1.chip_id.tolist() == tr2.chip_id.tolist()
    assert len(tr1) + len(va1) == len(store.index)
    # no shared lat/lon cells across the split
    def cells(df, deg=10.0):
        return set(zip(np.floor(df.lat / deg), np.floor(df.lon / deg)))
    assert not (cells(tr1) & cells(va1))


def _mk_pool(store, **kw):
    return ChipPool(store, store.index, pool_size=kw.pop("pool_size", 4),
                    device=torch.device("cpu"), workers=1,
                    refresh_chips_per_s=1000.0, seed=0,
                    pool_device="cpu", **kw)


def test_pool_coverage_sampling(store):
    pool = ChipPool(store, store.index, pool_size=len(store.index),
                    device=torch.device("cpu"), workers=1, seed=0,
                    pool_device="cpu")
    assert pool.coverage
    s = pool.sample_batch(16, token=16)
    assert s.shape == (16, 8, 16, 16) and s.dtype == torch.int8
    s, rows = pool.sample_batch(4, token=16, return_rows=True)
    assert rows.shape == (4,) and (rows >= 0).all()
    pool.shutdown()


def test_pool_cycle_mode(store):
    pool = _mk_pool(store, pool_size=3, slot_rows=32, cycle=True)
    assert not pool.coverage
    assert pool._n_tiles == 10 * 2       # ceil(64/32) bands per chip
    s = pool.sample_batch(8, token=16)
    assert s.shape == (8, 8, 16, 16)
    assert np.isfinite(pool.reuse_estimate) or pool.tiles_streamed <= 3
    pool.shutdown()


def test_pool_aligned_crops(store):
    pool = _mk_pool(store, pool_size=4, slot_rows=32, period=2)
    torch.manual_seed(0)
    pdev = pool.pool_device
    # reproduce the offset draws sample_batch makes with align=True
    torch.manual_seed(3)
    _ = torch.randint(0, 4, (64,), device=pdev)      # ci draw
    r = torch.randint(0, (32 - 16) // 2 + 1, (64,), device=pdev) * 2
    c = torch.randint(0, (64 - 16) // 2 + 1, (64,), device=pdev) * 2
    assert (r % 2 == 0).all() and (c % 2 == 0).all()
    torch.manual_seed(3)
    s = pool.sample_batch(64, token=16, align=True)
    assert s.shape == (64, 8, 16, 16)
    pool.shutdown()


def test_fixed_val_tokens(store):
    toks = fixed_val_tokens(store, store.index, token=16, per_chip=4,
                            max_chips=4)
    assert toks.dtype == torch.int8
    assert toks.shape[1:] == (8, 16, 16)
    assert 1 <= toks.shape[0] <= 16
    sat = torch.arange(4)[:, None].float().expand(-1, 3)
    toks2, sats = fixed_val_tokens(store, store.index, token=16, per_chip=4,
                                   max_chips=4, sat_table=sat)
    assert len(toks2) == len(sats)


def test_shard_writer_append_dedupe(tmp_path):
    d = tmp_path / "s"
    build_synthetic_store(d, n_chips=6, bands=4, chip_px=32, shard_size=4)
    w = ShardWriter(d, 4, bands=4, chip_px=32, append=True)
    # append starts at a fresh shard above all existing
    assert w.n_done == 2 * 4
    assert len(w.prior_index) == 6
    a = np.ones((4, 32, 32), dtype=np.int8)
    w.add(a, {"lat": 0.0, "lon": 0.0, "crs": "x", "year": 2024}, log_every=0)
    m = w.finalize("ae", nodata=-128, grid_period=2)
    assert m["n_chips"] == 7
    s = ChipStore(d)
    assert len(s.index) == 7
    assert s.index.chip_id.is_unique


def _coded_pool(store, **kw):
    """Coverage pool whose band 0 holds each pixel's column and band 1 its
    row, so a crop's first pixel gives its offset."""
    pool = ChipPool(store, store.index, pool_size=len(store.index),
                    device=torch.device("cpu"), workers=0, seed=0,
                    pool_device="cpu", **kw)          # no refresh threads
    g = pool.gpools[None]
    g[:, 0] = torch.arange(g.shape[-1], dtype=g.dtype)[None, None, :]
    g[:, 1] = torch.arange(g.shape[-2], dtype=g.dtype)[None, :, None]
    return pool


@pytest.mark.parametrize("align", [False, True])
def test_pool_crop_offsets_reach_the_last_position(store, align):
    pool = _coded_pool(store, slot_rows=32)
    torch.manual_seed(0)
    s = pool.sample_batch(4000, token=16, align=align)
    c, r = s[:, 0, 0, 0].long(), s[:, 1, 0, 0].long()
    step = pool.period if align else 1
    assert set(c.tolist()) == set(range(0, 64 - 16 + 1, step))
    assert set(r.tolist()) == set(range(0, 32 - 16 + 1, step))
    pool.shutdown()


def test_pool_crop_equal_to_slot(store):
    pool = _coded_pool(store, slot_rows=16, token=16)
    for align in (False, True):
        s = pool.sample_batch(8, token=16, align=align)
        assert (s[:, 1, 0, 0] == 0).all()
    pool.shutdown()


def test_pool_rejects_crop_larger_than_slot(store):
    with pytest.raises(ValueError, match="slot_rows >= 32"):
        _mk_pool(store, slot_rows=16, token=32)
    pool = _mk_pool(store, slot_rows=16)
    with pytest.raises(ValueError, match="does not fit"):
        pool.sample_batch(2, token=32)
    pool.shutdown()
