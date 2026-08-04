"""Nested tier-truncation (layer-wise matryoshka) sigma sampling."""
from types import SimpleNamespace

import torch

from georvq.train.stage1 import sample_sigmas


def _cfg(**kw):
    base = dict(max=4.0, kept_max=1.0, truncate=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_legacy_independent_uniform():
    torch.manual_seed(0)
    s = sample_sigmas(_cfg(), 4096, 3, "cpu")
    assert s.shape == (4096, 3)
    assert s.min() >= 0 and s.max() <= 4.0 and s.max() > 1.5


def test_nesting_invariant_and_levels():
    torch.manual_seed(0)
    cfg = _cfg(truncate=[0.4, 0.2, 0.2, 0.2])
    s = sample_sigmas(cfg, 8192, 4, "cpu")
    dropped = s == 4.0
    kept = ~dropped
    # kept tiers are gently noised, dropped pinned at max
    assert s[kept].max() <= 1.0
    # nesting: a dropped tier implies every finer tier is dropped
    assert (dropped[:, :-1] & ~dropped[:, 1:]).sum() == 0
    # coarsest tier is never dropped
    assert not dropped[:, 0].any()
    # mixture proportions roughly honored (n_drop = 0,1,2,3)
    n_drop = dropped.sum(1).float()
    for lvl, p in enumerate([0.4, 0.2, 0.2, 0.2]):
        frac = (n_drop == lvl).float().mean().item()
        assert abs(frac - p) < 0.03, (lvl, frac)


def test_truncate_probs_length_asserted():
    from georvq.config import load_config
    import pytest
    cfg = load_config("configs/tessera_stage1_4x.yaml",
                      ["model.sigma.truncate=[0.5,0.5]"])
    from georvq.train.stage1 import train
    with pytest.raises(AssertionError, match="one probability per tier"):
        train(cfg, gpu=-1)
