import copy
import glob

import pytest
from pydantic import ValidationError

from georvq.config import (Config, apply_overrides, config_diff, config_hash,
                           load_config)


def test_presets_validate():
    presets = glob.glob("configs/**/*.yaml", recursive=True)
    assert presets, "no config presets found"
    for p in presets:
        cfg = load_config(p)
        assert isinstance(cfg, Config)


def test_tiny_cfg_roundtrip(tiny_raw_cfg):
    cfg = Config.model_validate(tiny_raw_cfg)
    dumped = cfg.model_dump(mode="json")
    cfg2 = Config.model_validate(dumped)
    assert cfg == cfg2
    assert config_hash(cfg) == config_hash(cfg2)


def test_unknown_key_rejected(tiny_raw_cfg):
    bad = copy.deepcopy(tiny_raw_cfg)
    bad["model"]["typo_field"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_tier_shorthand(tiny_raw_cfg):
    cfg = Config.model_validate(tiny_raw_cfg)
    t = cfg.model.tiers[1]
    assert (t.grid, t.dim, t.codes, t.depth) == (4, 8, 32, 3)
    assert t.cells == 16


def test_min_depths_default_and_check(tiny_raw_cfg):
    raw = copy.deepcopy(tiny_raw_cfg)
    raw["model"].pop("min_depths")
    cfg = Config.model_validate(raw)
    assert cfg.model.min_depths == [1, 0]
    raw["model"]["min_depths"] = [1, 0, 0]      # wrong length
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_vq_sigma_exclusive(tiny_raw_cfg):
    raw = copy.deepcopy(tiny_raw_cfg)
    raw["model"]["sigma"] = {"max": 2.0}        # vq defaults on + sigma -> error
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
    raw["model"]["vq"] = None                   # stage-1: sigma only -> fine
    cfg = Config.model_validate(raw)
    assert cfg.model.sigma.max == 2.0


def test_grid_validators(tiny_raw_cfg):
    raw = copy.deepcopy(tiny_raw_cfg)
    raw["model"]["tiers"] = ["4:8:32:3", "1:16:32:3"]   # not ascending
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
    raw["model"]["tiers"] = ["1:16:32:3", "5:8:32:3"]   # 16 % 5 != 0
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_overrides(tiny_raw_cfg):
    raw = copy.deepcopy(tiny_raw_cfg)
    raw = apply_overrides(raw, ["train.steps=99",
                                "model.tokenizer.dim=64",
                                "model.tiers.0=1:16:64:4",
                                "log.wandb=false"])
    cfg = Config.model_validate(raw)
    assert cfg.train.steps == 99
    assert cfg.model.tokenizer.dim == 64
    assert cfg.model.tiers[0].codes == 64
    assert cfg.log.wandb is False


def test_config_diff(tiny_raw_cfg):
    a = Config.model_validate(tiny_raw_cfg)
    raw = apply_overrides(copy.deepcopy(tiny_raw_cfg), ["train.steps=99"])
    b = Config.model_validate(raw)
    d = config_diff(a, b)
    assert d == [("train.steps", 10, 99)]
