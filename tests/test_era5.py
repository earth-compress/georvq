import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.sources import get_source
from georvq.sources.era5 import _STATS_PATH

REPO = Path(__file__).parents[1]


@pytest.fixture
def era5_cfg(tiny_raw_cfg):
    raw = dict(tiny_raw_cfg)
    raw["data"] = {"source": "era5:t",
                   "store_dir": "/tmp/georvq-nonexistent"}
    raw["model"] = dict(raw["model"], bands=37)
    return Config.model_validate(raw)


def test_prep_standardizes_per_level(era5_cfg):
    src = get_source("era5:t", era5_cfg.model)
    with open(_STATS_PATH) as f:
        s = json.load(f)["t"]
    mu = torch.tensor(s["mean"]).view(1, -1, 1, 1)
    sd = torch.tensor(s["std"]).view(1, -1, 1, 1)
    raw = mu + sd * torch.randn(6, 37, 16, 16,
                                generator=torch.Generator().manual_seed(0))
    x, valid = src.prep(raw, "cpu")
    assert valid.all() and valid.shape == (6, 16, 16)
    # each level lands near N(0, 1) regardless of its physical scale
    m = x.mean(dim=(0, 2, 3))
    v = x.std(dim=(0, 2, 3))
    assert m.abs().max() < 0.1 and (v - 1).abs().max() < 0.1


def test_level_subset_selects_the_right_channels(era5_cfg):
    from georvq.sources.era5 import _LEVEL_SUBSETS
    idx = _LEVEL_SUBSETS["std13"]
    cfg13 = era5_cfg.model.model_copy(update={"bands": 13})
    src = get_source("era5:t@std13", cfg13)
    # encode each level's index into its value; prep must pick exactly
    # the subset levels (then standardize with the subset moments)
    raw = torch.arange(37).float().view(1, 37, 1, 1).expand(2, 37, 4, 4)
    x, valid = src.prep(raw, "cpu")
    assert x.shape == (2, 13, 4, 4)
    back = x * src._sd + src._mu
    assert torch.allclose(back[0, :, 0, 0], torch.tensor(idx).float(),
                          atol=1e-4)
    with pytest.raises(KeyError):
        get_source("era5:t@nosuchset", cfg13)


def test_unknown_variable_and_band_mismatch(era5_cfg):
    with pytest.raises(KeyError):
        get_source("era5:nosuchvar", era5_cfg.model)
    bad = era5_cfg.model.model_copy(update={"bands": 8})
    with pytest.raises(ValueError):
        get_source("era5:t", bad)


def test_model_builds_and_runs(era5_cfg):
    torch.manual_seed(0)
    model = build_model(era5_cfg, torch.device("cpu"))
    src = model.source
    raw = torch.randn(2, 37, 16, 16) * 5 + 250
    x, valid = src.prep(raw, "cpu")
    zs = model.encode_levels(x)
    Ls = torch.tensor([[sp.depth for sp in model.specs]] * 2)
    zqs, commit, _ = model.quantize(zs, Ls)
    out = model.decode(zqs, Ls).float()
    assert out.shape == x.shape
    loss, stats = src.recon_stats(out, x, valid)
    assert torch.isfinite(loss) and {"mse", "r2", "rmse_phys"} <= set(stats)


def test_rmse_phys_is_in_physical_units(era5_cfg):
    src = get_source("era5:t", era5_cfg.model)
    x = torch.zeros(2, 37, 8, 8)
    out = x + 1.0    # one sigma of error on every level
    valid = torch.ones(2, 8, 8, dtype=torch.bool)
    _, stats = src.recon_stats(out, x, valid)
    # one sigma pooled over levels = rms of the per-level sds, in K
    want = src._sd.pow(2).mean().sqrt()
    assert torch.allclose(stats["rmse_phys"], want, rtol=1e-5)
    spec = src.report(era5_cfg.model)
    assert "rmse_phys" in [m.key for m in spec.extra]
    assert spec.units == "K"


def _synth_store(tmp_path, n_cubes=2, levels=3, hours=8, px=8):
    zarr = pytest.importorskip("zarr")
    import pandas as pd
    root = zarr.open(str(tmp_path / "cubes.zarr"), mode="w")
    rng = np.random.default_rng(0)
    q = rng.integers(-32768, 32767,
                     (n_cubes, levels, hours, px, px)).astype(np.int16)
    root.create_array("t", shape=q.shape, dtype="int16")[:] = q
    sc = rng.uniform(1e-3, 2e-3, (n_cubes, levels))
    off = rng.uniform(200, 300, (n_cubes, levels))
    root.create_array("t_scale", shape=sc.shape, dtype="float64")[:] = sc
    root.create_array("t_offset", shape=off.shape, dtype="float64")[:] = off
    man = pd.DataFrame({"cube_id": range(n_cubes),
                        "window": [0, 1][:n_cubes],
                        "lat_center": [10.0] * n_cubes,
                        "lon_center": [20.0] * n_cubes})
    man.to_parquet(tmp_path / "manifest.parquet")
    return q, sc, off


def test_store_dequant_and_window_split(tmp_path):
    q, sc, off = _synth_store(tmp_path)
    out = tmp_path / "store_t"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "index_store_era5.py"),
         "--zarr-dir", str(tmp_path), "--var", "t",
         "--stride-h", "2", "--out", str(out)],
        check=True)
    from georvq.data.store import open_store
    s = open_store(out)
    assert s.bands == 3 and s.chip_px == 8 and s.dtype == np.float32
    # stride keeps hours 0,2,4,6 of each cube
    assert sorted(s.index.slot.unique()) == [0, 2, 4, 6]
    chip = s.open_shard(1)[4]
    want = ((q[1, :, 4].astype(np.int64) + 32768)
            * sc[1][:, None, None] + off[1][:, None, None])
    np.testing.assert_allclose(chip, want.astype(np.float32), rtol=0, atol=0)
    # the split severs whole draw windows, never rows within one
    tr, va = s.split(val_frac=0.5, seed=0)
    assert len(tr) and len(va)
    assert not set(tr.window) & set(va.window)


def test_boxcox_variant_roundtrips_and_departs_from_raw():
    import torch
    from georvq.config import load_config
    from georvq.sources import get_source
    cfg = load_config("configs/era5/era5_q64_ladder.yaml").model
    raw = get_source("era5:q", cfg)
    bc = get_source("era5:q+boxcox", cfg)
    x = torch.rand(2, 37, 8, 8) * 2e-2 + 1e-7
    x[0, 5, 0, 0] = -1e-6          # spectral-artifact negative
    yb, _ = bc.prep(x.clone(), "cpu")
    yr, _ = raw.prep(x.clone(), "cpu")
    assert not torch.allclose(yb, yr)
    assert torch.isfinite(yb).all()
    # physical roundtrip through the normalizer (clipped negatives come
    # back at the clip floor, so exclude the planted artifact pixel)
    back = bc.normalizer(cfg).denormalize(yb, None)
    err = (back - x.clamp_min(0)).abs().max()
    assert float(err) < 1e-6
    # rmse_phys of a perfect reconstruction is zero
    v = torch.ones(2, 8, 8, dtype=torch.bool)
    assert float(bc._rmse_phys(yb, yb, v)) < 1e-9


def test_nan_coded_variable_is_served_as_nan(tmp_path):
    """sst-style variables reserve -32768 as NaN and dequantize with
    +32767; variables not listed in nan_coded_vars keep +32768."""
    zarr = pytest.importorskip("zarr")
    q_t, sc_t, off_t = _synth_store(tmp_path)
    root = zarr.open(str(tmp_path / "cubes.zarr"), mode="a")
    rng = np.random.default_rng(1)
    q = rng.integers(-32767, 32767, (2, 8, 8, 8)).astype(np.int16)
    q[:, :, :3, :2] = -32768                      # "land"
    root.create_array("sst", shape=q.shape, dtype="int16")[:] = q
    sc = rng.uniform(1e-3, 2e-3, 2)
    off = rng.uniform(270, 280, 2)
    root.create_array("sst_scale", shape=sc.shape, dtype="float64")[:] = sc
    root.create_array("sst_offset", shape=off.shape, dtype="float64")[:] = off
    root.attrs.update({"nan_coded_vars": ["sst"]})
    from georvq.data.store import open_store
    stores = {}
    for var in ("sst", "t"):
        out = tmp_path / f"store_{var}"
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "index_store_era5.py"),
             "--zarr-dir", str(tmp_path), "--var", var,
             "--stride-h", "2", "--out", str(out)], check=True)
        stores[var] = open_store(out)
    chip = stores["sst"].open_shard(1)[4]
    assert chip.shape == (1, 8, 8)
    gap = q[1, 4] == -32768
    assert np.isnan(chip[0][gap]).all() and gap.sum() == 6
    want = (q[1, 4].astype(np.int64) + 32767) * sc[1] + off[1]
    np.testing.assert_allclose(chip[0][~gap], want[~gap].astype(np.float32),
                               rtol=0, atol=0)
    t = stores["t"].open_shard(1)[4]
    want_t = ((q_t[1, :, 4].astype(np.int64) + 32768)
              * sc_t[1][:, None, None] + off_t[1][:, None, None])
    np.testing.assert_allclose(t, want_t.astype(np.float32), rtol=0, atol=0)


def test_prep_marks_nan_pixels_invalid(era5_cfg):
    src = get_source("era5:t", era5_cfg.model)
    raw = 250 + 5 * torch.randn(3, 37, 8, 8,
                                generator=torch.Generator().manual_seed(0))
    x0, v0 = src.prep(raw, "cpu")
    assert v0.all()
    gappy = raw.clone()
    gappy[1, :, 2:4, 5] = float("nan")
    gappy[2, 7, 0, 0] = float("nan")              # one level missing
    x, v = src.prep(gappy, "cpu")
    assert torch.isfinite(x).all()
    assert int((~v).sum()) == 3 and not v[1, 2, 5] and not v[2, 0, 0]
    assert (x[1, :, 2:4, 5] == 0).all()
    # finite pixels are prepped exactly as in a gap-free batch
    assert torch.equal(x[v.unsqueeze(1).expand_as(x)],
                       x0[v.unsqueeze(1).expand_as(x0)])
