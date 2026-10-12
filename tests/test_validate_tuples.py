"""Named operating points under strict fill-order (path) training.

Path sampling trains only tuples where each tier is saturated before the
next opens. Validation must score those and nothing else: a half point
like 4-4-4 is never trained and would put a misleading point on the R-D
front.
"""
from georvq.config import TierSpec
from georvq.model.codec import path_tuples
from georvq.train.validate import canonical_tuples


def _specs():
    return [TierSpec(grid=1, dim=8, codes=16, depth=8),
            TierSpec(grid=4, dim=8, codes=16, depth=8),
            TierSpec(grid=16, dim=8, codes=16, depth=8)]


def _saturated(t, specs):
    """Every tier before the last non-zero one is at full depth."""
    last = max((i for i, L in enumerate(t) if L > 0), default=0)
    return all(t[i] == specs[i].depth for i in range(last))


def test_path_canonical_points_lie_on_the_path():
    specs, mins = _specs(), [2, 0, 0]
    path = {tuple(p) for p in path_tuples(specs, mins)}
    canon = canonical_tuples(specs, mins, path=True)
    assert canon["min"] == [2, 0, 0]
    assert canon["coarse_only"] == [8, 0, 0]
    assert canon["full"] == [8, 8, 8]
    for name, t in canon.items():
        assert tuple(t) in path, (name, t)
        assert _saturated(t, specs), (name, t)


def test_path_tuples_are_all_saturated():
    specs = _specs()
    for t in path_tuples(specs, [2, 0, 0]):
        assert _saturated(t, specs), t


def test_nested_half_point_is_off_path():
    """The non-path half point is the per-tier half cascade, which the
    path never visits; this is why `path=True` exists."""
    specs, mins = _specs(), [2, 0, 0]
    half = canonical_tuples(specs, mins)["half"]
    assert not _saturated(half, specs)
