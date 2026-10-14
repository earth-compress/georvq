"""Pool refresh threads survive read errors; the training thread raises
once the failures say the store is broken; the host-RAM warning."""
import time

import pytest
import torch

from georvq.data import pool as P
from georvq.data.build import build_synthetic_store
from georvq.data.pool import ChipPool
from georvq.data.store import ChipStore


@pytest.fixture
def store(tmp_path):
    build_synthetic_store(tmp_path / "store", n_chips=10, bands=8,
                          chip_px=64, shard_size=4)
    return ChipStore(tmp_path / "store")


def _wait(cond, timeout=10.0):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        time.sleep(0.01)
    return cond()


@pytest.mark.parametrize("cycle", [False, True])
def test_refresh_survives_and_reports_read_errors(store, monkeypatch, cycle):
    pool = ChipPool(store, store.index, 4, torch.device("cpu"), workers=2,
                    refresh_chips_per_s=2000.0, seed=0,
                    slot_rows=32 if cycle else None, cycle=cycle)
    try:
        real = store.open_shard

        def broken(*a, **kw):
            raise OSError("injected read failure")
        monkeypatch.setattr(store, "open_shard", broken)
        assert _wait(lambda: pool.read_failures >= 4)
        assert all(w.is_alive() for w in pool.workers)
        pool.sample_batch(2, token=16)       # below the threshold: no raise
        pool.MAX_CONSECUTIVE_FAILURES = 4
        with pytest.raises(RuntimeError, match="injected read failure"):
            pool.sample_batch(2, token=16)

        # the store comes back: reads resume on the same threads
        monkeypatch.setattr(store, "open_shard", real)
        n = pool.chips_loaded
        assert _wait(lambda: pool.chips_loaded > n + 2, timeout=30.0)
        pool.sample_batch(2, token=16)
    finally:
        pool.shutdown()


def test_failure_fraction_threshold(store):
    pool = ChipPool(store, store.index, 4, torch.device("cpu"), workers=1,
                    refresh_chips_per_s=1e-3, seed=0)
    try:
        pool.read_failures = pool.MIN_FAILURES
        pool._last_failure = "chip x: OSError()"
        pool.chips_loaded = 100 * pool.MIN_FAILURES      # 1% failing
        pool.check_health()
        pool.chips_loaded = pool.MIN_FAILURES            # 50% failing
        with pytest.raises(RuntimeError, match="pool refresh failing"):
            pool.check_health()
    finally:
        pool.shutdown()


def test_slurm_memory_warning(monkeypatch, capsys):
    for k in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_GPU",
              "SLURM_GPUS_ON_NODE", "SLURM_GPUS"):
        monkeypatch.delenv(k, raising=False)
    P._warn_slurm_mem(200 * 2 ** 30)
    assert "WARNING" not in capsys.readouterr().out      # not in a job
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(160 * 1024))
    P._warn_slurm_mem(100 * 2 ** 30)
    assert "WARNING" not in capsys.readouterr().out
    P._warn_slurm_mem(200 * 2 ** 30)
    out = capsys.readouterr().out
    assert "WARNING" in out and "200.0 GiB" in out and "160.0 GiB" in out
    monkeypatch.delenv("SLURM_MEM_PER_NODE")
    monkeypatch.setenv("SLURM_MEM_PER_GPU", str(100 * 1024))
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
    P._warn_slurm_mem(150 * 2 ** 30)
    assert "WARNING" not in capsys.readouterr().out
    P._warn_slurm_mem(250 * 2 ** 30)
    assert "WARNING" in capsys.readouterr().out
