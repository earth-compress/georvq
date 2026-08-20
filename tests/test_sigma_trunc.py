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


def test_dropped_tiers_are_zeroed_in_forward():
    """With drop mask set, a dropped tier's latents must carry NO signal:
    output is invariant to input changes that only affect that tier's
    information when every tier is dropped."""
    from georvq.config import load_config
    from georvq.factory import build_model
    cfg = load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3']",
        "model.vq=null", "model.sigma={max: 4.0, freqs: 4}"])
    m = build_model(cfg, "cpu"); m.eval()
    B, W = 2, cfg.model.window
    sig = torch.full((B, 2), 4.0)
    drop = torch.ones(B, 2, dtype=torch.bool)
    torch.manual_seed(0)
    with torch.no_grad():
        r1 = m.forward_stage1(torch.randn(B, cfg.model.bands, W, W), sig, drop)
        r2 = m.forward_stage1(torch.randn(B, cfg.model.bands, W, W), sig, drop)
    assert torch.allclose(r1, r2, atol=1e-5)      # all info zeroed out


def test_truncate_probs_length_asserted():
    from georvq.config import load_config
    import pytest
    cfg = load_config("configs/stage1.yaml",
                      ["model.sigma.truncate=[0.5,0.5]"])
    from georvq.train.stage1 import train
    with pytest.raises(AssertionError, match="one probability per tier"):
        train(cfg, gpu=-1)
