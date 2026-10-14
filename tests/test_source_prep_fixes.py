"""Source preps, artifact resolution, heads and eval guards: the DEM/S2
inverse, the ERA5 named prep under level subsets, the S2 stats-PCA rate
independent of the working directory, and the eval exclusion files."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from georvq.config import Config, load_config
from georvq.data.pool import GroupBatch
from georvq.factory import build_model, from_checkpoint
from georvq.repro import save_checkpoint
from georvq.sources import get_source
from georvq.sources.prep import SourcePrep, build_prep, prep_of

REPO = Path(__file__).resolve().parents[1]
PCA = "artifacts/s2_stats_pca.npz"


@pytest.fixture
def dem_cfg(tiny_raw_cfg):
    raw = dict(tiny_raw_cfg)
    raw["data"] = {"source": "dem", "store_dir": "/tmp/georvq-nonexistent"}
    raw["model"] = dict(raw["model"], bands=1)
    return Config.model_validate(raw)


def _s2_cfg(**model):
    return Config.model_validate({
        "data": {"source": "s2", "store_dir": "/tmp/unused"},
        "model": {
            "window": 24, "bands": 32,
            "tiers": ["1:16:32:3", "4:8:32:3"],
            "ff_ratio": 2.0,
            "tokenizer": {"dim": 32, "depth": 1},
            "decoder": {"dim": 32, "depth": 1, "heads": 4, "head_feat": 8},
            **model,
        },
        "log": {"run_tag": "s2test", "wandb": False},
    })


def _s2_raw(b=2, w=24):
    g = torch.Generator().manual_seed(0)
    return GroupBatch({
        "10m": torch.randint(1, 10000, (b, 4, w, w), generator=g),
        "20m": torch.randint(1, 10000, (b, 6, w // 2, w // 2), generator=g),
        "60m": torch.randint(1, 10000, (b, 3, w // 6, w // 6), generator=g)})


# ---- SourcePrep.inverse for per-sample normalizers ------------------------

def test_dem_prep_inverse_uses_meta_stats(dem_cfg):
    src = get_source("dem", dem_cfg.model)
    p = SourcePrep(src, dem_cfg.model)
    assert not p.invertible          # the harness inverts with meta={}
    raw = torch.rand(3, 1, 16, 16) * 500 + 100
    x, valid, meta = p(raw.clone(), {"k": 1}, "cpu")
    assert meta["k"] == 1 and "stats" in meta
    back = p.inverse(x, meta)
    assert torch.allclose(back, raw, atol=1e-3)
    with pytest.raises(ValueError, match="per-sample"):
        p.inverse(x, {})


def test_s2_prep_inverse_uses_meta_stats():
    cfg = _s2_cfg()
    src = get_source("s2", cfg.model)
    p = SourcePrep(src, cfg.model)
    assert not p.invertible
    raw = _s2_raw()
    x, _, meta = p(raw, {}, "cpu")
    back = p.inverse(x, meta)
    for g in raw:
        assert torch.allclose(back[g], raw[g].float() / 10000, atol=1e-5)
    with pytest.raises(ValueError):
        p.inverse(x, {})


def test_fixed_normalizer_stays_invertible():
    cfg = load_config(str(REPO / "configs/era5/era5_q64_ladder.yaml"))
    p = build_prep(None, get_source(cfg.data.source, cfg.model), cfg.model)
    assert p.invertible


# ---- ERA5 named prep under level subsets ----------------------------------

def test_era5_boxcox_prep_keeps_level_subset():
    cfg = load_config(str(REPO / "configs/era5/era5_q64bc_ladder.yaml"),
                      ["data.source=era5:q@std13", "model.bands=13"])
    src = get_source(cfg.data.source, cfg.model)
    p = build_prep("boxcox", src, cfg.model)
    legacy = get_source("era5:q@std13+boxcox", cfg.model)
    raw = torch.rand(2, 37, 8, 8) * 2e-2 + 1e-7
    x_new, _, _ = p(raw.clone(), {}, "cpu")
    x_old, _ = legacy.prep(raw.clone(), "cpu")
    assert x_new.shape[1] == 13 and torch.equal(x_new, x_old)
    assert len(p.state()["boxcox"]["lam"]) == 13
    back = p.inverse(x_new, {})
    sub = raw[:, legacy._levels]
    assert float((back - sub).abs().max()) < 1e-6


def test_era5_boxcox_with_single_raises():
    cfg = load_config(str(REPO / "configs/era5/era5_q64bc_ladder.yaml"),
                      ["data.source=era5:q@single", "model.bands=1"])
    src = get_source(cfg.data.source, cfg.model)
    with pytest.raises(ValueError, match="single"):
        build_prep("boxcox", src, cfg.model)
    with pytest.raises(ValueError, match="single"):
        get_source("era5:q@single+boxcox", cfg.model)


def test_era5_units_only_for_loadable_variables():
    import json
    from georvq.sources import era5
    with open(era5._STATS_PATH) as f:
        have = set(json.load(f))
    assert set(era5._UNITS) <= have


# ---- S2 stats-PCA rate: CWD-independent and embedded ----------------------

def test_s2_side_bits_independent_of_cwd(tmp_path, monkeypatch):
    from georvq.sources import s2
    cfg = _s2_cfg(stats_pca=PCA)
    s2._PACKED_BITS.clear()
    monkeypatch.chdir(REPO)
    at_root = get_source("s2", cfg.model)._stats_encodings(cfg.model)
    s2._PACKED_BITS.clear()
    monkeypatch.chdir(tmp_path)
    elsewhere = get_source("s2", cfg.model)._stats_encodings(cfg.model)
    assert at_root == elsewhere and at_root[0][0] == "pca"


def test_s2_checkpoint_embeds_stats_pca_bits(tmp_path, monkeypatch):
    from georvq.sources import s2
    art = tmp_path / "basis.npz"
    shutil.copy(REPO / PCA, art)
    cfg = _s2_cfg(stats_pca=str(art))
    m = build_model(cfg, "cpu")
    depths = (None, [1, 0], [3, 3])
    want = [m.side_bits(d) for d in depths]
    prep = build_prep(None, m.source, cfg.model)
    st = prep.state()
    assert st == {"source": {"stats_pca_bits": s2._packed_bits(str(art))}}
    ck = tmp_path / "m.pt"
    save_checkpoint(str(ck), m, cfg, 0, prep_state=st)
    art.unlink()
    s2._PACKED_BITS.clear()
    m2 = from_checkpoint(str(ck), "cpu")
    assert [m2.side_bits(d) for d in depths] == want


def test_s2_missing_artifact_names_where_it_looked(tmp_path):
    from georvq.sources.s2 import resolve_artifact
    with pytest.raises(FileNotFoundError, match="looked in"):
        resolve_artifact("artifacts/no_such_basis.npz")


# ---- from_checkpoint attaches the run's prep ------------------------------

def test_from_checkpoint_attaches_prep(dem_cfg, tmp_path):
    m = build_model(dem_cfg, "cpu")
    ck = tmp_path / "m.pt"
    save_checkpoint(str(ck), m, dem_cfg, 0)
    m2 = from_checkpoint(str(ck), "cpu")
    assert isinstance(m2.prep, SourcePrep) and prep_of(m2) is m2.prep
    del m2.prep
    assert isinstance(prep_of(m2), SourcePrep)


def test_prep_of_rebuilds_named_prep():
    from georvq.sources.era5 import Era5BoxcoxPrep
    cfg = load_config(str(REPO / "configs/era5/era5_q64bc_ladder.yaml"))

    class _M:                       # attributes prep_of reads
        full_config = cfg
        source = get_source(cfg.data.source, cfg.model)
    _M.cfg = cfg.model
    assert isinstance(prep_of(_M()), Era5BoxcoxPrep)


# ---- head layout -----------------------------------------------------------

def test_dem_head_state_dict_layout(dem_cfg):
    head = get_source("dem", dem_cfg.model).build_head(dem_cfg.model, 32)
    assert sorted(head.state_dict()) == sorted(
        f"{m}.{p}" for m in ("to_patch", "conv.0", "conv.2", "conv.4")
        for p in ("weight", "bias"))


# ---- eval exclusion guards and import side effects ------------------------

def test_ae_exclusion_files_required(tmp_path):
    from georvq.eval.extract import training_tile_paths
    with pytest.raises(FileNotFoundError):
        training_tile_paths(tmp_path / "a.parquet", tmp_path / "b.csv")


def test_tessera_exclusion_index_required(tmp_path):
    from georvq.eval.tessera_extract import training_tile_keys
    with pytest.raises(FileNotFoundError):
        training_tile_keys(tmp_path / "index.parquet")


def test_importing_eval_leaves_environment_alone():
    from georvq.eval.extract import _S3_ENV
    env = {k: v for k, v in os.environ.items() if k not in _S3_ENV}
    env["PYTHONPATH"] = str(REPO)
    code = ("import os, georvq.eval; "
            f"print(any(k in os.environ for k in {sorted(_S3_ENV)!r}))")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
