"""Rate-sweep helpers: which tuples are swept and what each one costs."""
import torch

from georvq.config import Config, TierSpec
from georvq.eval.codec_ab import bits_per_px, select_front_tuples, \
    trained_ladder
from georvq.factory import build_model
from georvq.model.codec import path_tuples, tuple_bits


def _specs(depth=8):
    return [TierSpec(grid=g, dim=8, codes=256, depth=depth)
            for g in (1, 4, 16)]


def _saturating(t, specs):
    # a tier may only be partly filled if every coarser tier is full
    for i in range(1, len(t)):
        if t[i] > 0 and t[i - 1] < specs[i - 1].depth:
            return False
    return True


def test_path_ladder_is_the_trained_path():
    specs = _specs()
    assert trained_ladder(specs, [4, 0, 0]) \
        == [list(t) for t in path_tuples(specs, [4, 0, 0])]


def test_nested_minimums_keep_only_saturating_points():
    specs = _specs()
    ts = trained_ladder(specs, [6, 1, 0])
    assert ts[0] == [8, 1, 0] and ts[-1] == [8, 8, 8]
    assert all(_saturating(t, specs) for t in ts)
    front = select_front_tuples(specs, [6, 1, 0], 8)
    assert len(front) == 8 and all(t in ts for t in front)
    assert [6, 1, 0] not in front and [6, 4, 4] not in front


def test_bits_per_px_charges_the_transmitted_mean(tiny_raw_cfg):
    raw = {**tiny_raw_cfg,
           "model": {**tiny_raw_cfg["model"], "stats_cond": {}}}
    torch.manual_seed(0)
    m = build_model(Config.model_validate(raw))
    sb = m.side_bits()
    assert sb > 0
    ls = [3, 1]
    W = m.window
    assert bits_per_px(m, ls) == (tuple_bits(m.specs, ls) + sb) / W ** 2
    plain = build_model(Config.model_validate(tiny_raw_cfg))
    assert bits_per_px(plain, ls) == tuple_bits(plain.specs, ls) / W ** 2
