import os

import pytest

RELEASE_BUNDLE = os.environ.get(
    "GEORVQ_AE_BUNDLE", "/data/georvq-releases/ae-beta-v1/bundle")


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
