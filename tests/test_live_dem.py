"""Live COPDEM store: column-map invertibility and (data-local) reads."""
import os

import numpy as np
import pytest

from georvq.data.live_dem import (CHIP_PX, N_SRC, col_map, first_occurrence,
                                  invert_cols)

LIVE_DIR = "/data/georvq-stores/copdem_live"
needs_data = pytest.mark.skipif(not os.path.isdir(LIVE_DIR),
                                reason="live COPDEM store not on this host")


@pytest.mark.parametrize("n_src", sorted(set(N_SRC.values())))
def test_col_map_invertible(n_src):
    """Every source column appears >= once; first_occurrence inverts the
    replication exactly for any chip content."""
    m = col_map(n_src)
    assert m.shape == (CHIP_PX,)
    assert m[0] == 0 and m[-1] == n_src - 1
    assert set(np.unique(m)) == set(range(n_src))     # nothing skipped
    assert (np.diff(m) >= 0).all()                    # monotone
    chip = np.random.default_rng(0).normal(size=(1, 4, n_src))
    rep = chip[..., m]
    assert rep.shape[-1] == CHIP_PX
    assert (invert_cols(rep, n_src) == chip).all()
    assert len(first_occurrence(n_src)) == n_src


@needs_data
def test_round_trip_bit_exact():
    """Chip -> invert_cols == the raw native window from the band VRT."""
    from rasterio.windows import Window

    from georvq.data.store import open_store
    store = open_store(LIVE_DIR)
    rng = np.random.default_rng(1)
    for i in rng.integers(0, len(store.index), 4):
        r = store.index.iloc[int(i)]
        chip = store.read_chip(int(r.chip_id))
        assert chip.shape == (1, CHIP_PX, CHIP_PX)
        assert chip.dtype == np.float32
        n_src = N_SRC[int(r.band)]
        ds = store._handle(int(r.band))
        raw = ds.read(1, window=Window(int(r.vcol0), int(r.vrow0),
                                       n_src, CHIP_PX),
                      boundless=True, fill_value=store.nodata)
        assert (invert_cols(chip[0], n_src) == raw).all()


@needs_data
def test_shard_slot_contract():
    """open_shard(sh)[slot] must address the same chip as chip_id."""
    from georvq.data.store import open_store
    store = open_store(LIVE_DIR)
    r = store.index.iloc[123456 % len(store.index)]
    via_shard = store.open_shard(int(r["shard"]))[int(r["slot"])]
    direct = store.read_chip(int(r.chip_id))
    assert (via_shard == direct).all()


def test_index_must_be_in_chip_id_order(tmp_path):
    import json

    import pandas as pd

    from georvq.data.live_dem import LiveDEMStore
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"kind": "live_dem", "chip_px": CHIP_PX, "bands": 1,
         "dtype": "float32", "nodata": -32767.0}))
    idx = pd.DataFrame({"chip_id": [0, 2, 1], "band": [0, 0, 0],
                        "vrow0": [0, 0, 0], "vcol0": [0, 0, 0]})
    idx.to_parquet(tmp_path / "index.parquet")
    with pytest.raises(ValueError, match="chip_id"):
        LiveDEMStore(tmp_path)
    idx.sort_values("chip_id").to_parquet(tmp_path / "index.parquet")
    LiveDEMStore(tmp_path)
