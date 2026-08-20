"""Live Sentinel-2 reads: Major-TOM GeoTIFFs straight into the pool.

No chip store is built. The corpus-backed tests self-skip where the data
is absent, so the suite still runs on a machine without it.
"""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from georvq.data.live_s2 import (GROUP_BANDS, GROUP_PX, LiveS2Store,
                                 cell_lat_lon)
from georvq.data.pool import ChipPool

ROOT = "/data/Major-TOM/Core-S2L1C"
have_corpus = pytest.mark.skipif(
    not os.path.isdir(ROOT), reason="Major-TOM Core-S2L1C not present")


def test_grid_cell_position_is_monotone_and_signed():
    """The split only needs neighbours to bucket together, but the signs
    and the ordering have to be right or the geographic split is a
    random split wearing a hat."""
    n_lat, _ = cell_lat_lon("142U", "142U_1091R")
    s_lat, _ = cell_lat_lon("142D", "142D_1091R")
    assert n_lat > 0 > s_lat and n_lat == pytest.approx(-s_lat)
    lo, hi = cell_lat_lon("10U", "10U_1R")[0], cell_lat_lon("300U", "300U_1R")[0]
    assert hi > lo                                    # rows go north
    _, e = cell_lat_lon("142U", "142U_1091R")
    _, w = cell_lat_lon("142U", "142U_1091L")
    assert e > 0 > w                                  # R east, L west
    # columns are denser near the poles: same index is further east at
    # high latitude
    _, near_eq = cell_lat_lon("10U", "10U_100R")
    _, near_pole = cell_lat_lon("700U", "700U_100R")
    assert abs(near_pole) > abs(near_eq)
    # always a legal longitude
    for cell in ("1U", "400U", "800U"):
        assert -180.0 <= cell_lat_lon(cell, f"{cell}_5000R")[1] <= 180.0


def test_store_surface_matches_chipstore():
    """The pool talks to whatever `open_store` returns; a live store has
    to present the same attributes a ChipStore does."""
    store = LiveS2Store.__new__(LiveS2Store)      # no filesystem needed
    store.groups = [{"name": n, "bands": len(b), "px": GROUP_PX[n]}
                    for n, b in GROUP_BANDS.items()]
    store.chip_px = 1068
    assert {g["name"] for g in store.groups} == {"10m", "20m", "60m"}
    # the three native grids must register exactly on one crop origin
    for g in store.groups:
        assert store.chip_px % g["px"] == 0
    ratios = sorted(store.chip_px // g["px"] for g in store.groups)
    assert ratios == [1, 2, 6]
    assert sum(g["bands"] for g in store.groups) == 13


def _index(n):
    import glob
    hits = []
    for h in glob.iglob(os.path.join(ROOT, "*", "*", "*", "B02.tif")):
        hits.append(h)
        if len(hits) >= n:
            break
    rows = []
    for i, h in enumerate(sorted(hits)):
        d = os.path.dirname(h)
        p = d.split(os.sep)
        lat, lon = cell_lat_lon(p[-3], p[-2])
        rows.append({"chip_id": i, "shard": 0, "slot": i, "path": d,
                     "scene_id": p[-1], "cell": p[-3], "col_cell": p[-2],
                     "lat": lat, "lon": lon, "crs": "", "year": 0})
    return pd.DataFrame(rows)


@have_corpus
def test_reads_real_scene_at_native_resolutions():
    store = LiveS2Store(ROOT, index=_index(2))
    for g, px in GROUP_PX.items():
        a = store.open_shard(0, g)[0]
        assert a.shape == (len(GROUP_BANDS[g]), px, px)
        assert a.dtype == np.uint16
    assert store.nodata == 0 and store.grid_period == 6


@have_corpus
def test_pool_serves_registered_crops_from_live_tiffs():
    """The end the user cares about: batches out of the pool, off the
    original GeoTIFFs, with no store built."""
    store = LiveS2Store(ROOT, index=_index(4))
    pool = ChipPool(store, store.index, pool_size=4,
                    device=torch.device("cpu"), workers=1,
                    refresh_chips_per_s=2.0, seed=0, pool_device="cpu",
                    period=store.grid_period)
    try:
        b = pool.sample_batch(4, token=96)
        assert b["10m"].shape == (4, 4, 96, 96)
        assert b["20m"].shape == (4, 6, 48, 48)
        assert b["60m"].shape == (4, 3, 16, 16)
        assert b["10m"].dtype == torch.uint16 or b["10m"].dtype == torch.int16
    finally:
        pool.shutdown()


@have_corpus
def test_clouds_are_not_filtered():
    """An explicit statement of policy: the cloud mask exists next to the
    bands and the store never opens it."""
    store = LiveS2Store(ROOT, index=_index(1))
    d = store.index.path.iloc[0]
    assert os.path.exists(os.path.join(d, "cloud_mask.tif"))
    assert "cloud_mask" not in sum(GROUP_BANDS.values(), [])
    assert store.manifest["clouds"].startswith("kept")


@have_corpus
def test_every_figure_the_run_draws_renders_on_real_data():
    """A figure that raises kills the run.

    The trainer draws figures inside the validation branch with no
    try/except, so a preview that cannot render its source aborts training
    at the first fig_interval -- an hour or a day in, having produced
    nothing. Every plot the S2 preset will draw is exercised here on real
    scenes at the configured window.
    """
    import matplotlib.pyplot as plt

    from georvq.config import load_config
    from georvq.data.pool import fixed_val_tokens
    from georvq.factory import build_model
    from georvq.sources.report import resolve_report
    from georvq.train.figures import codebook_figure, rd_figure, recon_figure
    from georvq.train.validate import canonical_tuples, tname, validate

    store = LiveS2Store(ROOT, index=_index(4))
    cfg = load_config("configs/s2_pyr96.yaml")
    torch.manual_seed(0)
    model = build_model(cfg)
    model.eval()
    spec = resolve_report(model.source, cfg)

    val_s = fixed_val_tokens(store, store.index, token=cfg.model.window,
                             per_chip=4, max_chips=4, max_tokens=8,
                             nodata=store.nodata)
    assert val_s["10m"].shape[1:] == (4, cfg.model.window, cfg.model.window)

    canon = canonical_tuples(model.specs, cfg.model.min_depths)
    tuples = list(canon.values())
    vs, cbt = validate(model, model.source, val_s, tuples,
                       {tname(t) for t in tuples}, spec=spec)
    assert "r2_ceiling" in vs and "rmse_ceiling" in vs
    assert "sam_ceiling" in vs

    f = rd_figure(cbt, vs, model.specs, tuples, 2000, spec,
                  px=cfg.model.window ** 2, side_bits=model.side_bits())
    assert f.axes[0].get_ylabel().startswith("R\u00b2")
    plt.close(f)
    # R2 is bounded above by 1, so 1 - m on a log axis is meaningful
    assert spec.has_figure("rd_log")

    plt.close(codebook_figure(vs, model.specs, 2000))

    f3 = recon_figure(model, model.source, val_s, 2000, n=2,
                      mins=cfg.model.min_depths, preview=spec.preview)
    plt.close(f3)


@have_corpus
def test_true_colour_preview_uses_the_visible_bands():
    """R=B04, G=B03, B=B02 out of the 10m group. Getting the indices wrong
    still renders -- it just renders a false-colour image nobody notices,
    so pin the mapping rather than the fact that it drew something."""
    from georvq.sources.report import band_rgb_preview
    from georvq.sources.s2 import BANDS_10M

    assert BANDS_10M.index("B04") == 2
    assert BANDS_10M.index("B03") == 1
    assert BANDS_10M.index("B02") == 0

    store = LiveS2Store(ROOT, index=_index(1))
    plane = torch.from_numpy(store.open_shard(0, "10m")[0]).float()[None]
    x = {"10m": plane / 10000.0}
    valid = torch.ones(1, plane.shape[-2], plane.shape[-1], dtype=torch.bool)
    pv = band_rgb_preview(2, 1, 0, select=lambda t: t["10m"])
    rgb = pv.render(x, pv.basis(x, valid))
    assert rgb.shape == (1, plane.shape[-2], plane.shape[-1], 3)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    # a real scene is not a constant field
    assert rgb.std() > 0.01
