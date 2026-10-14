"""A warm start must not have its loaded codebooks discarded by the
VQ-free warmup unless the config says the refit is intended.

`vq.warmup_steps` trains the autoencoder unquantized and then k-means
initialises the codebooks at warmup_steps + 1. From a checkpoint with no
books, or books of another shape, that is the intended path. From one
whose books this model loads, it throws them away halfway into the run.
"""
import os
from pathlib import Path

import pytest
import torch

from georvq.config import Config, load_config
from georvq.factory import build_model
from georvq.repro import save_checkpoint
from georvq.train.warmstart import book_shapes, check_warm_start, load_partial

REPO = Path(__file__).parents[1]
PRESETS = sorted(p.relative_to(REPO).as_posix()
                 for p in (REPO / "configs").rglob("*.yaml"))

# The preset that produced each init_from checkpoint the presets name.
# The checkpoints are not in the repository, so the guard is evaluated on
# the codebook shapes of the producing config.
PROVENANCE = {
    "checkpoints/stage1/model.pt": "configs/ae/stage1.yaml",
    "checkpoints/dem128_overnight_resume/model_best.pt":
        "configs/dem/dem128.yaml",
    "checkpoints/dem128_stage1_live/model.pt":
        "configs/dem/dem128_stage1_live.yaml",
    "checkpoints/tessera_vq8_split2/model_best.pt":
        "configs/tessera/tessera_vq8.yaml",
}


def _books_loaded(own, src):
    return any(k in src and src[k] == shp for k, shp in own.items())


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_pass_the_warm_start_guard(preset):
    cfg = load_config(REPO / preset)
    v = cfg.model.vq
    if v is None or not cfg.train.init_from or not v.warmup_steps:
        check_warm_start(cfg, books_loaded=True)    # guard does not apply
        return
    src = PROVENANCE.get(str(cfg.train.init_from))
    assert src is not None, (
        f"{preset} warm-starts from {cfg.train.init_from} with a VQ-free "
        f"warmup; add the producing preset to PROVENANCE")
    loaded = _books_loaded(book_shapes(cfg.model),
                           book_shapes(load_config(REPO / src).model))
    check_warm_start(cfg, loaded)


def test_the_six_refit_presets_are_covered():
    names = {"configs/ae/cascade.yaml", "configs/ae/hybrid_rate.yaml",
             "configs/dem/dem128_cascade.yaml",
             "configs/dem/dem128_cascade_sh.yaml",
             "configs/dem/dem128_live_frozen.yaml",
             "configs/tessera/tessera_vq8_k256.yaml"}
    assert names <= set(PRESETS)
    for p in names:
        cfg = load_config(REPO / p)
        assert cfg.train.init_from and cfg.model.vq.warmup_steps


def _cfg(raw, **train):
    raw = dict(raw)
    raw["train"] = dict(raw.get("train", {}), **train)
    return Config.model_validate(raw)


def test_guard_refuses_only_loaded_books(tiny_raw_cfg):
    cfg = _cfg(tiny_raw_cfg, init_from="x.pt")
    cfg.model.vq.warmup_steps = 500
    with pytest.raises(ValueError, match="warmup_steps"):
        check_warm_start(cfg, books_loaded=True)
    check_warm_start(cfg, books_loaded=False)
    cfg.train.refit_books = True
    check_warm_start(cfg, books_loaded=True)
    cfg.train.refit_books = False
    cfg.model.vq.warmup_steps = 0
    check_warm_start(cfg, books_loaded=True)


def test_book_shapes_match_the_built_model(tiny_raw_cfg):
    for bits in (None, 8):
        cfg = _cfg(tiny_raw_cfg)
        cfg.model.vq.book_bits = bits
        sd = build_model(cfg, torch.device("cpu")).state_dict()
        got = {k: tuple(t.shape) for k, t in sd.items()
               if k.endswith("codebook.weight")}
        assert got == book_shapes(cfg.model)


def _ckpt(tmp_path, raw, name):
    cfg = Config.model_validate(raw)
    torch.manual_seed(0)
    m = build_model(cfg, torch.device("cpu"))
    for rvq in m.rvqs:
        for lev in rvq.levels:
            lev.initialized.fill_(True)
            lev.step.fill_(123)
    path = tmp_path / name
    save_checkpoint(path, m, cfg, step=1)
    return path


def test_resized_books_are_left_for_data_init(tiny_raw_cfg, tmp_path):
    """A level whose book was not copied keeps initialized=False and
    step=0, so the lazy data init fills it from the first batch."""
    src = _ckpt(tmp_path, tiny_raw_cfg, "k32.pt")
    raw = dict(tiny_raw_cfg)
    raw["model"] = dict(raw["model"], tiers=["1:16:16:3", "4:8:32:3"])
    m = build_model(Config.model_validate(raw), torch.device("cpu"))
    _, loaded = load_partial(m, src, torch.device("cpu"))
    assert loaded                                   # tier 1 matches
    for lev in m.rvqs[0].levels:                    # K 32 -> 16
        assert not bool(lev.initialized) and int(lev.step) == 0
    for lev in m.rvqs[1].levels:                    # unchanged
        assert bool(lev.initialized) and int(lev.step) == 123

    raw["model"] = dict(raw["model"], tiers=["1:16:16:3", "4:8:16:3"])
    m = build_model(Config.model_validate(raw), torch.device("cpu"))
    _, loaded = load_partial(m, src, torch.device("cpu"))
    assert not loaded
    assert not any(bool(lev.initialized) for rvq in m.rvqs
                   for lev in rvq.levels)


def test_guard_fires_before_the_store_opens(tiny_raw_cfg, tmp_path):
    """The store path does not exist: reaching setup_data would raise a
    different error, so a ValueError about warmup_steps shows the guard
    runs first."""
    from georvq.train.loop import train
    src = _ckpt(tmp_path, tiny_raw_cfg, "same.pt")
    raw = dict(tiny_raw_cfg)
    raw["log"] = dict(raw["log"], out_dir=str(tmp_path / "runs"))
    cfg = _cfg(raw, init_from=str(src))
    cfg.model.vq.warmup_steps = 5
    assert not os.path.exists(cfg.data.store_dir)
    with pytest.raises(ValueError, match="warmup_steps"):
        train(cfg, gpu=-1)
