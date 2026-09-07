import torch

from georvq.config import load_config
from georvq.factory import build_model


def _model(*ov):
    cfg = load_config("configs/smoke.yaml", list(ov))
    return build_model(cfg, "cpu")


def test_embed_norm_default_off_and_flag_changes_output():
    torch.manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(2, 8, 16, 16), dim=1)
    m0 = _model()
    assert m0.tokenizer.embed_norm is False
    m1 = _model("model.tokenizer.embed_norm=true")
    m1.load_state_dict(m0.state_dict())     # same weights, new forward
    with torch.no_grad():
        z0 = m0.encode_levels(x, m0._stats_of(x, None), None)
        z1 = m1.encode_levels(x, m1._stats_of(x, None), None)
    assert not torch.allclose(z0[0], z1[0])
    assert all(torch.isfinite(z).all() for z in z1)


def test_embed_norm_bounds_stream_into_trunk():
    torch.manual_seed(0)
    m = _model("model.tokenizer.embed_norm=true")
    x = torch.nn.functional.normalize(torch.randn(2, 8, 16, 16), dim=1)
    seen = {}
    def pre(mod, args):
        seen["std"] = args[0].std(-1)

    h = m.tokenizer.vit.layers[0].register_forward_pre_hook(pre)
    with torch.no_grad():
        m.encode_levels(x, m._stats_of(x, None), None)
    h.remove()
    # per-token std of the trunk input is pinned near 1 by the norm
    assert float((seen["std"] - 1).abs().max()) < 0.1


def test_checkpoint_carries_and_restores_optimizer(tmp_path):
    from georvq.repro import save_checkpoint
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.nn.functional.normalize(torch.randn(2, 8, 16, 16), dim=1)
    m.encode_levels(x, m._stats_of(x, None), None)[0].pow(2).mean().backward()
    opt.step()
    p = tmp_path / "ck.pt"
    save_checkpoint(str(p), m, m.full_config, 7, opt=opt)
    ck = torch.load(str(p), weights_only=False)
    assert "opt" in ck and "sched" not in ck and ck["step"] == 7
    opt2 = torch.optim.AdamW(_model().parameters(), lr=1e-3)
    opt2.load_state_dict(ck["opt"])         # moments transfer to a twin
    st = list(opt2.state.values())[0]
    assert "exp_avg_sq" in st and st["exp_avg_sq"].abs().sum() > 0
