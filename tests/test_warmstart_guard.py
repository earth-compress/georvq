"""A warm start and a VQ-free warmup are mutually exclusive.

`vq.warmup_steps` trains the autoencoder unquantized and then k-means
initialises the codebooks at warmup_steps + 1. That is correct from
scratch and destructive on a warm start: the k-means discards the very
books the run was warm-started with, halfway in, with nothing in the
loss to show for it. Presets inherit, so this is easy to do by accident.
"""
import pytest

from georvq.config import Config


def _cfg(raw, **train):
    raw = dict(raw)
    raw["train"] = dict(raw.get("train", {}), **train)
    return Config.model_validate(raw)


def test_warm_start_with_vq_warmup_is_refused(tiny_raw_cfg):
    cfg = _cfg(tiny_raw_cfg, init_from="checkpoints/whatever.pt", steps=10)
    cfg.model.vq.warmup_steps = 20000
    with pytest.raises(ValueError, match="warmup_steps"):
        _check(cfg)


def test_warm_start_without_vq_warmup_is_fine(tiny_raw_cfg):
    cfg = _cfg(tiny_raw_cfg, init_from="checkpoints/whatever.pt", steps=10)
    cfg.model.vq.warmup_steps = 0
    _check(cfg)


def test_from_scratch_may_keep_its_warmup(tiny_raw_cfg):
    cfg = _cfg(tiny_raw_cfg, init_from=None, steps=10)
    cfg.model.vq.warmup_steps = 20000
    _check(cfg)


def _check(cfg):
    """The guard as loop.py applies it, isolated from the training loop.

    The loop needs a GPU, a pool and a dataset to reach this line; the
    condition it guards on does not.
    """
    v = cfg.model.vq
    if cfg.train.init_from and v.warmup_steps:
        raise ValueError(
            f"train.init_from is set and model.vq.warmup_steps is "
            f"{v.warmup_steps}.")


def test_guard_matches_the_source(tiny_raw_cfg):
    """Pin the guard's condition to loop.py, so _check cannot drift."""
    import inspect

    from georvq.train import loop

    src = inspect.getsource(loop)
    assert "if cfg.train.init_from and v.warmup_steps:" in src
    assert "warmup_steps: 0 for a warm start" in src
