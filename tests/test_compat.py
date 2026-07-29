"""Golden-checkpoint gate: the released model must load strict and
reproduce its recorded reconstruction quality exactly (bit-faithful
decode is the repo's core promise)."""
import os

import numpy as np
import pytest
import torch

from tests.conftest import RELEASE_BUNDLE

GOLDEN = os.path.join(os.path.dirname(__file__), "golden/divfull_val_tokens.npz")


def test_release_loads_and_reproduces():
    ckpt = os.path.join(RELEASE_BUNDLE, "model.pt")
    if not os.path.exists(ckpt):
        pytest.skip("release weights not present (set GEORVQ_AE_BUNDLE)")
    from georvq.factory import from_checkpoint
    from georvq.metrics import masked_recon_stats

    model = from_checkpoint(ckpt, "cpu")
    cfg = model.full_config
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
