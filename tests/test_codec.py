import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.model.codec import (monotone_tuples, sample_monotone_depths,
                                tuple_bits)
from georvq.model.layers import thermometer


def _model(tiny_raw_cfg, **model_over):
    raw = {**tiny_raw_cfg, "model": {**tiny_raw_cfg["model"], **model_over}}
    cfg = Config.model_validate(raw)
    torch.manual_seed(0)
    return build_model(cfg), cfg


def _x(cfg, B=4):
    torch.manual_seed(1)
    x = torch.randn(B, cfg.model.bands, cfg.model.window, cfg.model.window)
    return torch.nn.functional.normalize(x, dim=1, eps=1e-6)


def test_forward_shapes(tiny_raw_cfg):
    model, cfg = _model(tiny_raw_cfg)
    model.eval()
    x = _x(cfg)
    zs = model.encode_levels(x)
    assert [tuple(z.shape) for z in zs] == [(4, 1, 16), (4, 16, 8)]
    Ls = torch.tensor([[3, 2]]).expand(4, -1)
    out, commit, select = model.forward_train(x, Ls)
    assert out.shape == x.shape
    assert torch.isfinite(commit) and float(select) == 0.0


def test_eval_tuple_matches_forward(tiny_raw_cfg):
    """Decoding a depth tuple from cached cumsums must equal the training
    path's thermometer-masked quantization at the same tuple."""
    model, cfg = _model(tiny_raw_cfg)
    model.eval()
    x = _x(cfg)
    Ls = torch.tensor([[3, 2]]).expand(4, -1)
    with torch.no_grad():
        out1, _, _ = model.forward_train(x, Ls)
        _, cums = model.level_cumsums(x)
        out2 = model.eval_tuple(cums, [3, 2], x)
    assert torch.equal(out1, out2)


def test_cumsum_telescoping(tiny_raw_cfg):
    model, cfg = _model(tiny_raw_cfg)
    model.eval()
    with torch.no_grad():
        _, cums = model.level_cumsums(_x(cfg))
    for c, sp in zip(cums, model.specs):
        assert c.shape[-2] == sp.depth + 1
        assert torch.equal(c[:, :, 0], torch.zeros_like(c[:, :, 0]))
        # depth L slice == sum of the first L per-level contributions
        per = c[:, :, 1:] - c[:, :, :-1]
        assert torch.allclose(per.cumsum(-2)[:, :, -1], c[:, :, -1], atol=1e-6)


def test_skip_masked_equivalence(tiny_raw_cfg):
    """skip_masked quantization must produce the same decoder inputs as the
    plain thermometer-masked path (it only skips masked-off work)."""
    m_plain, cfg = _model(tiny_raw_cfg)
    m_skip, _ = _model(tiny_raw_cfg, vq={"skip_masked": True})
    m_skip.load_state_dict(m_plain.state_dict())
    m_plain.eval(), m_skip.eval()
    x = _x(cfg)
    Ls = torch.stack([torch.tensor([3, 1]), torch.tensor([2, 2]),
                      torch.tensor([1, 0]), torch.tensor([3, 3])])
    zs = [z.detach() for z in m_plain.encode_levels(x)]
    q1, c1, _ = m_plain.quantize(zs, Ls)
    q2, c2, _ = m_skip.quantize(zs, Ls)
    for a, b in zip(q1, q2):
        assert torch.allclose(a, b, atol=1e-6)


def test_stage1(tiny_raw_cfg):
    model, cfg = _model(tiny_raw_cfg, vq=None, sigma={"max": 2.0, "freqs": 4})
    model.eval()
    x = _x(cfg)
    zs = model.encode_stage1(x)
    for z in zs:   # unit-norm * sqrt(dim) => per-cell norm == sqrt(dim)
        n = z.norm(dim=-1)
        assert torch.allclose(n, torch.full_like(n, z.shape[-1] ** 0.5), atol=1e-4)
    sig = torch.zeros(4, 2)
    out = model.forward_stage1(x, sig)
    assert out.shape == x.shape
    # sigma=0 => zero noise => deterministic
    assert torch.equal(out, model.forward_stage1(x, sig))


def test_matryoshka_utils(tiny_raw_cfg):
    model, cfg = _model(tiny_raw_cfg)
    specs = model.specs
    tuples = monotone_tuples(specs)
    assert len(tuples) == 9          # L0 in 1..3, L1 in 0..min(L0,3)
    for t in tuples:
        assert t[0] >= 1 and t[1] <= t[0]
    assert tuple_bits(specs, [1, 0]) == specs[0].bits(1)
    torch.manual_seed(0)
    Ls = sample_monotone_depths(512, specs, "cpu", mins=[2, 1], full_frac=0.1)
    assert (Ls[:, 0] >= 2).all() and (Ls[:, 1] >= 1).all()
    assert (Ls[:, 1] <= Ls[:, 0]).all()
    assert (Ls[:, 0] <= 3).all() and (Ls[:, 1] <= 3).all()


def test_thermometer():
    L = torch.tensor([0, 2, 3])
    t = thermometer(L, 3)
    assert t.tolist() == [[0, 0, 0], [1, 1, 0], [1, 1, 1]]
