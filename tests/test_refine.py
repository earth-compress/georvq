"""Additive residual refinements (georvq.inference.refine)."""
import numpy as np
import pytest

from georvq.inference import refine as R


def _res(seed=0, n=6, cells=8, dim=32, outlier=True):
    g = np.random.default_rng(seed)
    r = g.standard_normal((n, cells, dim)).astype(np.float32) * 0.3
    if outlier:
        r[0] *= 8.0          # exercise the per-window scale
    return r


def test_cost_counts_codes_and_the_f16_scale():
    """1 cell x 256 dims x 2 bits + f16 scale = 528 bits = 66 B."""
    spec = R.RefinementSpec(tier=0, bits=2)
    assert spec.cost_bits(cells=1, dim=256) == 66 * 8


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_uniform_round_trip_shrinks_the_residual(bits):
    res = _res()
    spec = R.RefinementSpec(tier=1, bits=bits)
    back = R.unpack(*R.pack(res, spec), spec, cells=8, dim=32)
    assert back.shape == res.shape
    rel = ((back - res) ** 2).sum() / (res ** 2).sum()
    assert rel < 0.35


def test_uniform_at_one_bit_is_worse_than_nothing_at_the_default_sigma():
    """Documents a real limit of the uniform family, not a bug.

    The convention puts the outermost reconstruction point at
    sigma*std, so at 1 bit the two points land at +-2 std where the
    optimum is +-0.8 std — the quantizer overshoots every sample and
    the 'refinement' adds error. One bit per dimension is the Lloyd
    family's regime (and the rate the Tessera release actually uses),
    which is why `sigma` must be retuned if anyone wants uniform there.
    """
    res = _res(outlier=False)
    spec = R.RefinementSpec(tier=1, bits=1)
    back = R.unpack(*R.pack(res, spec), spec, cells=8, dim=32)
    assert ((back - res) ** 2).sum() > (res ** 2).sum()

    tuned = R.RefinementSpec(tier=1, bits=1, sigma=0.8)
    back = R.unpack(*R.pack(res, tuned), tuned, cells=8, dim=32)
    assert ((back - res) ** 2).sum() < (res ** 2).sum()


def test_lloyd_one_bit_hits_the_gaussian_bound():
    """On rotated (isotropic) coordinates the Lloyd-Max quantizer should
    reach the classic Gaussian value 1 - 2/pi at one bit per dimension."""
    res = _res(n=64, cells=64, dim=16, outlier=False)
    spec = R.RefinementSpec(tier=2, bits=1, quantizer="lloyd", rotate_seed=0)
    table = R.fit_lloyd(res, spec)
    assert table.shape == (16, 2)
    back = R.unpack(*R.pack(res, spec, table), spec, cells=64, dim=16,
                    table=table)
    rel = float(((back - res) ** 2).sum() / (res ** 2).sum())
    assert rel == pytest.approx(1 - 2 / np.pi, abs=0.05)


def test_rotation_is_orthogonal_and_reproducible():
    a, b = R.rotation(24, 7), R.rotation(24, 7)
    assert np.array_equal(a, b)                       # seed determines it
    assert not np.array_equal(a, R.rotation(24, 8))
    assert np.allclose(a @ a.T, np.eye(24), atol=1e-5)
    assert a.dtype == np.float32


def test_lloyd_requires_a_rotation():
    with pytest.raises(ValueError, match="rotate_seed"):
        R.RefinementSpec(tier=0, bits=2, quantizer="lloyd")


def test_refinement_is_tier_generic():
    """The coder treats every tier alike; which one to refine is release
    policy."""
    res = _res()
    for tier in (0, 1, 2):
        spec = R.RefinementSpec(tier=tier, bits=2)
        back = R.unpack(*R.pack(res, spec), spec, cells=8, dim=32)
        assert ((back - res) ** 2).sum() < (res ** 2).sum()
