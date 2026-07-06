import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.model.rate import ContextRateModel, calib_tau, soft_rate


def test_context_rate_model_grads():
    torch.manual_seed(0)
    B, cells, dim, K, L = 2, 16, 8, 64, 3
    m = ContextRateModel(cells, dim, K, L, model_dim=32, n_blocks=2,
                         heads=4, coarse_dim=16)
    z_prev = torch.randn(B, cells, dim)
    coarse = torch.randn(B, 5, 16)
    level = torch.tensor([1, 2])
    logits = m(z_prev, level, coarse)
    assert logits.shape == (B, cells, K)
    d = torch.randn(B * cells, K).abs()
    r = soft_rate(d, logits, calib_tau(d))
    r.backward()
    gsum = sum(p.grad.abs().sum().item() for p in m.parameters()
               if p.grad is not None)
    assert torch.isfinite(r) and r > 0 and gsum > 0


def test_pyramid_rate(tiny_raw_cfg):
    raw = {**tiny_raw_cfg,
           "model": {**tiny_raw_cfg["model"],
                     "rate": {"model_dim": 32, "blocks": 1, "ctx_dim": 8}}}
    cfg = Config.model_validate(raw)
    torch.manual_seed(0)
    model = build_model(cfg)
    model.eval()
    x = torch.nn.functional.normalize(torch.randn(4, 8, 16, 16), dim=1)
    zs = [z.float() for z in model.encode_levels(x)]
    bpp = model.train_rate_bits(zs, tau_rel=0.5)
    assert bpp.ndim == 0 and torch.isfinite(bpp) and bpp > 0
    bpp.backward()   # gradient reaches the rate model
    gsum = sum(p.grad.abs().sum().item() for p in model.rate.parameters()
               if p.grad is not None)
    assert gsum > 0

    with torch.no_grad():
        bits = model.entropy_level_bits(x)
    assert len(bits) == 2
    for b, sp in zip(bits, model.specs):
        assert b.shape == (4, sp.depth)
        assert torch.isfinite(b).all() and (b >= 0).all()
