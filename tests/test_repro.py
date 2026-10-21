"""Checkpoint provenance and loading: checkpoints load with
`weights_only=True`, including those written before the torch version
was stored as a string."""
import os
import random

import numpy as np
import pytest
import torch
from torch.torch_version import TorchVersion

from georvq.config import Config
from georvq.factory import build_model, from_checkpoint
from georvq.repro import load_checkpoint, run_metadata, save_checkpoint, seed_all

RELEASED = [
    "/home/ali/projects/georvq/scratch/ae-v2/ownk_sc4_step80000.pt",
    "/data/georvq-releases/tessera-v2/tessera_v2_step88000.pt",
]


def test_torch_version_is_stored_as_str(tiny_raw_cfg):
    meta = run_metadata(Config.model_validate(tiny_raw_cfg))
    assert type(meta["torch"]) is str


def test_new_checkpoint_needs_no_allowlist(tiny_raw_cfg, tmp_path):
    cfg = Config.model_validate(tiny_raw_cfg)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, cfg, step=3, opt=opt)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["step"] == 3
    m2 = from_checkpoint(path)
    for (k, a), b in zip(model.state_dict().items(),
                         m2.state_dict().values()):
        assert torch.equal(a, b), k


def test_legacy_metadata_types_load(tiny_raw_cfg, tmp_path):
    """A checkpoint holding a TorchVersion and numpy float scalars (the
    lr in optimizer and scheduler state) loads through load_checkpoint
    and from_checkpoint, and the plain weights_only load refuses it."""
    cfg = Config.model_validate(tiny_raw_cfg)
    model = build_model(cfg)
    path = tmp_path / "old.pt"
    torch.save({"model": model.state_dict(),
                "config": cfg.model_dump(mode="json"), "step": 1,
                "metrics": {}, "torch": TorchVersion("2.10.0"),
                "opt": {"param_groups": [{"lr": np.float64(1e-4)}]},
                "sched": {"_last_lr": [np.float32(1e-4)]}}, path)
    with pytest.raises(Exception):
        torch.load(path, map_location="cpu", weights_only=True)
    ck = load_checkpoint(path)
    assert ck["opt"]["param_groups"][0]["lr"] == np.float64(1e-4)
    from_checkpoint(path)


def test_arbitrary_globals_are_still_refused(tmp_path):
    path = tmp_path / "evil.pt"
    torch.save({"x": os.path.join}, path)
    with pytest.raises(Exception):
        load_checkpoint(path)


@pytest.mark.slow
@pytest.mark.parametrize("path", RELEASED)
def test_released_checkpoints_load_weights_only(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    ck = load_checkpoint(path)
    assert "model" in ck and "config" in ck


def test_seed_all_seeds_numpy_and_random():
    seed_all(7)
    a = (torch.rand(1).item(), np.random.rand(), random.random())
    seed_all(7)
    b = (torch.rand(1).item(), np.random.rand(), random.random())
    assert a == b
