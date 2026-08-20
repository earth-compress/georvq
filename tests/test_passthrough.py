"""`vq.passthrough_frac`: keep the continuous path alive during stage-2."""
import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.model.codec import sample_monotone_depths


def _cfg(frac):
    return load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3']",
        f"model.vq={{warmup_steps: 0, passthrough_frac: {frac}}}"])


def test_default_off():
    assert load_config("configs/smoke.yaml", []).model.vq.passthrough_frac \
        == 0.0


def _blend(m, cfg, x, frac, seed):
    """Mirror of the loop's passthrough splice."""
    torch.manual_seed(seed)
    specs = cfg.model.tiers
    dev = x.device
    Ls_full = torch.tensor([sp.depth for sp in specs], device=dev)[None]
    Ls = sample_monotone_depths(x.shape[0], specs, dev,
                                mins=cfg.model.min_depths,
                                full_frac=cfg.model.full_frac)
    pt = torch.rand(x.shape[0], device=dev) < frac
    Ls = torch.where(pt[:, None], Ls_full.expand_as(Ls), Ls)
    zs = m.encode_levels(x)
    zqs, _, _ = m.quantize(zs, Ls)
    m3 = pt[:, None, None]
    zqs = [torch.where(m3, z.to(zq.dtype), zq) for z, zq in zip(zs, zqs)]
    return pt, Ls, zs, zqs


def test_passthrough_rows_are_exactly_the_raw_latents():
    cfg = _cfg(0.5)
    m = build_model(cfg, "cpu")
    m.train()
    x = torch.randn(16, cfg.model.bands, cfg.model.window, cfg.model.window)
    for _ in range(2):                       # let the books initialize
        m.forward_train(x, torch.tensor([[3, 3]] * 16))
    pt, Ls, zs, zqs = _blend(m, cfg, x, 0.5, seed=0)
    assert pt.any() and not pt.all(), "test needs a mixed batch"
    for z, zq in zip(zs, zqs):
        assert torch.allclose(zq[pt], z[pt].to(zq.dtype), atol=1e-6), \
            "passthrough rows were quantized"
        assert not torch.allclose(zq[~pt], z[~pt].to(zq.dtype), atol=1e-4), \
            "non-passthrough rows were NOT quantized"


def test_passthrough_rows_are_conditioned_at_full_depth():
    """If the depth conditioning still said 'tier absent', the decoder
    would compensate for an absence that isn't there."""
    cfg = _cfg(0.5)
    m = build_model(cfg, "cpu")
    m.train()
    x = torch.randn(16, cfg.model.bands, cfg.model.window, cfg.model.window)
    m.forward_train(x, torch.tensor([[3, 3]] * 16))
    pt, Ls, _, _ = _blend(m, cfg, x, 0.5, seed=0)
    full = torch.tensor([sp.depth for sp in cfg.model.tiers])
    assert torch.equal(Ls[pt], full[None].expand(int(pt.sum()), -1))


def test_gradient_reaches_the_encoder_through_passthrough():
    cfg = _cfg(1.0)                          # entire batch continuous
    m = build_model(cfg, "cpu")
    m.train()
    x = torch.randn(8, cfg.model.bands, cfg.model.window, cfg.model.window)
    m.forward_train(x, torch.tensor([[3, 3]] * 8))
    m.zero_grad(set_to_none=True)
    pt, Ls, zs, zqs = _blend(m, cfg, x, 1.0, seed=1)
    m.decode(zqs, Ls).pow(2).mean().backward()
    g = m.tokenizer.patches[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
