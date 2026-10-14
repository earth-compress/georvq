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


def _model_raw(tiny_raw_cfg, **model):
    raw = copy.deepcopy(tiny_raw_cfg)
    raw["model"].update(model)
    return raw


def test_validation_raises_without_assert(tiny_raw_cfg):
    """Validators raise ValueError, so `python -O` keeps them."""
    import ast
    import inspect

    import georvq.config as C
    tree = ast.parse(inspect.getsource(C))
    assert not [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    bad = copy.deepcopy(tiny_raw_cfg)
    bad["data"]["val_frac"] = 5.0
    with pytest.raises(ValidationError, match="val_frac"):
        Config.model_validate(bad)
    with pytest.raises(ValidationError, match="ascending"):
        Config.model_validate(_model_raw(
            tiny_raw_cfg, tiers=["4:8:32:3", "1:16:32:3"]))


def test_path_sampling_needs_zero_finer_minimums(tiny_raw_cfg):
    Config.model_validate(_model_raw(
        tiny_raw_cfg, depth_sampling="path", min_depths=[2, 0]))
    with pytest.raises(ValidationError, match="min_depths"):
        Config.model_validate(_model_raw(
            tiny_raw_cfg, depth_sampling="path", min_depths=[2, 1]))
    # nested sampling keeps accepting a finer minimum
    Config.model_validate(_model_raw(tiny_raw_cfg, min_depths=[2, 1]))


def test_sigma_with_stats_cond_rejected(tiny_raw_cfg):
    with pytest.raises(ValidationError, match="stats_cond"):
        Config.model_validate(_model_raw(
            tiny_raw_cfg, vq=None, sigma={}, stats_cond={}))


def test_local_attn_with_decoder_bank_rejected(tiny_raw_cfg):
    bank = [{"tiers": 1, "out_px": 4, "dim": 32, "depth": 1, "heads": 4,
             "head_feat": 8},
            {"tiers": 2, "out_px": 16, "dim": 32, "depth": 1, "heads": 4,
             "head_feat": 8}]
    Config.model_validate(_model_raw(tiny_raw_cfg, decoder_bank=bank))
    with pytest.raises(ValidationError, match="local_attn"):
        Config.model_validate(_model_raw(
            tiny_raw_cfg, decoder_bank=bank, local_attn=[None, 1]))


def test_norm_latents_declared_once():
    import inspect

    import georvq.config as C
    src = inspect.getsource(C.ModelConfig)
    assert src.count("norm_latents: bool") == 1
