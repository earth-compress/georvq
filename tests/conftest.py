import os

import pytest

RINGSPACE_DIR = "/home/ali/projects/ring-space"
DIVFULL_CKPT = os.path.join(RINGSPACE_DIR, "checkpoints/ae_pyr64_divfull/model_best.pt")

requires_ringspace = pytest.mark.ringspace


def pytest_collection_modifyitems(config, items):
    if os.path.isdir(RINGSPACE_DIR):
        return
    skip = pytest.mark.skip(reason="original research repo not present on this machine")
    for item in items:
        if "ringspace" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tiny_raw_cfg():
    """Small CPU-friendly config dict for model tests."""
    return {
        "data": {"source": "ae", "store_dir": "/tmp/georvq-nonexistent-store"},
        "model": {
            "window": 16,
            "bands": 8,
            "tiers": ["1:16:32:3", "4:8:32:3"],
            "min_depths": [1, 0],
            "ff_ratio": 2.0,
            "tokenizer": {"dim": 32, "depth": 1},
            "decoder": {"dim": 32, "depth": 1, "heads": 4, "head_feat": 8},
        },
        "train": {"batch_windows": 4, "steps": 10, "amp": False, "tf32": False},
        "log": {"run_tag": "test", "wandb": False},
    }
