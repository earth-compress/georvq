"""Legacy-checkpoint gate: the flagship trained model must load strict and
reproduce its recorded reconstruction quality exactly."""
import os

import numpy as np
import pytest
import torch

from tests.conftest import DIVFULL_CKPT

pytestmark = pytest.mark.ringspace

GOLDEN = os.path.join(os.path.dirname(__file__), "golden/divfull_val_tokens.npz")


def test_divfull_loads_and_reproduces():
    if not os.path.exists(DIVFULL_CKPT):
        pytest.skip("flagship checkpoint not on this machine")
    from georvq.compat import from_legacy_checkpoint
    from georvq.metrics import masked_recon_stats
    from georvq.sources.ae import NODATA

    model, cfg = from_legacy_checkpoint(DIVFULL_CKPT, device="cpu")
    assert cfg.model.window == 64 and len(cfg.model.tiers) == 3
    # reference cosines were recorded with fp32 assignment
    for rvq in model.rvqs:
        for lev in rvq.levels:
            lev.bf16_scores = False

    g = np.load(GOLDEN)
    s = torch.from_numpy(g["tokens"])
    x, valid = model.source.prep(s, "cpu")
    with torch.no_grad():
        _, cums = model.level_cumsums(x)
        cos_888 = masked_recon_stats(model.eval_tuple(cums, [8, 8, 8], x),
                                     x, valid)[1].item()
        cos_810 = masked_recon_stats(model.eval_tuple(cums, [8, 1, 0], x),
                                     x, valid)[1].item()
    assert abs(cos_888 - float(g["cos_888"])) < 1e-5
    assert abs(cos_810 - float(g["cos_810"])) < 1e-5


def test_remap_rejects_unknown_keys():
    from georvq.compat import remap_state_dict
    with pytest.raises(KeyError):
        remap_state_dict({"mystery.weight": torch.zeros(1)})
