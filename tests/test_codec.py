import pytest
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


def test_canonical_points_lie_on_the_trained_path():
    """Named operating points must be tuples the run actually trains.

    Under path sampling the truncation path fills each tier to its depth
    before the next begins, so a balanced tuple like 4-4-4 is never
    sampled -- and is strictly dominated in rate-distortion terms besides.
    Reporting it as `half` measures the model at a point it never saw.
    """
    from georvq.config import TierSpec, parse_tier
    from georvq.model.codec import path_tuples, tuple_bits
    from georvq.train.validate import canonical_tuples

    specs = [TierSpec(**parse_tier(t))
             for t in ("1:32:64:8", "4:64:512:8", "16:64:4096:8")]
    mins = [4, 0, 0]
    path = [list(t) for t in path_tuples(specs, mins)]

    canon = canonical_tuples(specs, mins, sampling="path")
    for name, t in canon.items():
        assert t in path, f"{name}={t} is not on the trained path"

    assert canon["min"] == path[0] == [4, 0, 0]
    assert canon["full"] == path[-1] == [8, 8, 8]
    # coarse_only is the coarse tier carrying it alone, at full depth
    assert canon["coarse_only"] == [8, 0, 0]
    # half is chosen by RATE, not index: the path is front-loaded with
    # cheap coarse steps, so the midpoint of the list is nowhere near the
    # midpoint of the R-D span
    hb = tuple_bits(specs, canon["half"])
    assert 0.4 < hb / tuple_bits(specs, canon["full"]) < 0.6
    assert canon["half"] != path[len(path) // 2]

    # the historical nested behaviour is unchanged
    nested = canonical_tuples(specs, mins, sampling="nested")
    assert nested["half"] == [4, 4, 4]
    assert nested["half"] not in path


def test_the_path_saturates_each_tier_before_the_next():
    """Every step adds exactly one codebook to one tier, and no finer tier
    opens while a coarser one is unfilled."""
    from georvq.config import TierSpec, parse_tier
    from georvq.model.codec import path_tuples

    specs = [TierSpec(**parse_tier(t))
             for t in ("1:32:64:8", "4:64:512:8", "16:64:4096:8")]
    path = [list(t) for t in path_tuples(specs, [4, 0, 0])]

    for prev, cur in zip(path, path[1:]):
        step = [c - p for c, p in zip(cur, prev)]
        assert sum(step) == 1 and min(step) >= 0, f"{prev} -> {cur}"
    for t in path:
        for i in range(1, len(t)):
            if t[i] > 0:                    # a tier only opens once every
                assert t[i - 1] == specs[i - 1].depth   # coarser one is full


def test_side_bits_follows_the_source_signature(tiny_raw_cfg):
    """A source's side_bits may or may not take depths; a TypeError raised
    inside one that does is not swallowed."""
    model = build_model(Config.model_validate(tiny_raw_cfg))

    class Old:
        def side_bits(self, cfg):
            return 5.0

    class New:
        def side_bits(self, cfg, depths=None):
            return 7.0 if depths is None else float(sum(depths))

    class Broken:
        def side_bits(self, cfg, depths=None):
            raise TypeError("bug inside the source")

    model.source = Old()
    assert model.side_bits((1, 2)) == 5.0
    model.source = New()
    assert model.side_bits() == 7.0 and model.side_bits((1, 2)) == 3.0
    model.source = Broken()
    with pytest.raises(TypeError, match="bug inside"):
        model.side_bits()
