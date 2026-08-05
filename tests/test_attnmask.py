"""Hierarchical local attention masks."""
import torch

from georvq.config import TierSpec, load_config, parse_tier
from georvq.factory import build_model
from georvq.model.attnmask import build_local_mask


def test_mask_geometry():
    specs = [TierSpec(**parse_tier(t)) for t in
             ["1:16:32:3", "4:8:32:3", "16:8:32:3", "32:8:32:3"]]
    m = build_local_mask(specs, window=64, radii=[None, None, 2, 4])
    n = 1 + 16 + 256 + 1024
    assert m.shape == (n, n) and m.dtype == torch.bool
    assert not m[0].any() and not m[:, 0].any()      # coarse fully global
    assert not m[1:17].any()                          # mid rows global
    f0 = 17                                           # fine block start
    # fine cell (0,0) -> fine cell (0,2): centers 10px, 2*4px=8 -> BLOCKED?
    # radius 2 fine cells = 8px; distance (0,0)->(0,2) = 8px -> allowed
    assert not m[f0, f0 + 2]
    assert m[f0, f0 + 4]                              # 16px away -> masked
    # symmetric between spatial tiers: max of the two radii applies
    u0 = 17 + 256
    assert (m[f0, u0:].sum() > 0) and (m[u0, f0:u0].sum() > 0)
    assert not m[u0, 0] and not m[u0, 1]              # ultra sees globals


def test_masked_model_runs_and_matches_unmasked_when_all_global():
    base = ["model.tiers=['1:16:32:3','4:8:32:3','16:8:32:3']",
            "model.vq=null", "model.sigma={max: 4.0, freqs: 4}"]
    cfg = load_config("configs/smoke.yaml",
                      base + ["model.local_attn=[null, null, 2]"])
    m = build_model(cfg, "cpu")
    assert m.tokenizer.attn_mask.shape[0] == 273
    m.train()
    W = cfg.model.window
    x = torch.randn(2, cfg.model.bands, W, W)
    rec = m.forward_stage1(x, torch.rand(2, 3) * 4)
    rec.pow(2).mean().backward()
    assert torch.isfinite(rec).all()
    # all-null radii == no masking at all
    cfg2 = load_config("configs/smoke.yaml",
                       base + ["model.local_attn=[null, null, null]"])
    m2 = build_model(cfg2, "cpu")
    assert not m2.tokenizer.attn_mask.any()


def test_mask_not_in_state_dict():
    cfg = load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3','16:8:32:3']",
        "model.vq=null", "model.sigma={max: 4.0, freqs: 4}",
        "model.local_attn=[null, null, 2]"])
    m = build_model(cfg, "cpu")
    assert not any("attn_mask" in k for k in m.state_dict())
