"""Rotated mean channel: no clipping inside hard bounds, error bounds,
decode by table, save/load and ids."""
import numpy as np
import pytest

from georvq.inference.mean_codec import (RotatedChannel, channel_kind,
                                         load_channel)


def _unit_means(rng, n, d=8, spread=0.3):
    """Means of unit vectors: inside the unit ball."""
    c = rng.normal(size=(n, 1, d)) + spread * rng.normal(size=(n, 16, d))
    v = c / np.linalg.norm(c, axis=2, keepdims=True)
    return v.mean(1)


def _rotation(rng, d):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def test_hard_bounds_never_clip_and_errors_are_bounded():
    rng = np.random.default_rng(0)
    d = 8
    x = _unit_means(rng, 5000, d)
    mu0 = x.mean(0)
    B = 1 + np.linalg.norm(mu0)
    s = x.std(0)
    ch = RotatedChannel(mu0, _rotation(rng, d), -np.full(d, B), np.full(d, B),
                        np.full(d, 10), scale=s)
    # any point of the unit ball, however far from the data
    far = rng.normal(size=(2000, d))
    far /= np.linalg.norm(far, axis=1, keepdims=True)
    far *= rng.uniform(0, 1, (2000, 1))
    for v in (x, far, -far):
        assert not ch.clipped(v).any()
        y, yh = ch.rotate(v), ch.rotate(ch.round_trip(v))
        # half a companded step, in the component's own units
        bound = 0.5 * ch.step * np.sqrt(y ** 2 + ch.scale ** 2) * 1.01 + 1e-6
        assert (np.abs(yh - y) <= bound).all()


def test_uniform_identity_is_a_per_value_quantizer():
    ch = RotatedChannel(np.zeros(3), np.eye(3), -np.ones(3), np.ones(3),
                        [4, 8, 12])
    assert ch.cost_bits() == 24
    assert ch.record_fields() == [("rot_0", 1, 4), ("rot_1", 1, 8),
                                  ("rot_2", 1, 12)]
    x = np.array([[-1.0, 0.3, 1.0], [0.999, -0.5, 0.1234]])
    y = ch.round_trip(x)
    step = 2.0 / ((1 << ch.bits) - 1)
    assert (np.abs(y - x) <= step / 2 + 1e-7).all()
    assert y[0, 0] == -1 and y[0, 2] == 1          # the ends decode exactly
    out = np.array([[2.0, -3.0, 0.0]])
    assert ch.clipped(out)[0, :2].all()
    assert (ch.encode(out)[0, :2] == [15, 0]).all()


def test_decode_by_table_save_load_ids(tmp_path):
    rng = np.random.default_rng(2)
    d = 6
    ch = RotatedChannel(rng.normal(size=d) * 0.1, _rotation(rng, d),
                        -np.full(d, 1.2), np.full(d, 1.2),
                        [9, 9, 8, 7, 6, 5], scale=np.full(d, 0.05))
    x = _unit_means(rng, 300, d)
    q = ch.encode(x)
    y = ch.decode(q)
    assert y.dtype == np.float32
    assert np.array_equal(ch.encode(y), q)         # codes are fixed points
    p = tmp_path / "r.npz"
    ch.save(p)
    assert channel_kind(p) == "rotated"
    c2 = load_channel(p, "rotated")
    assert np.array_equal(c2.decode(q), y)
    for a, b in zip(ch.id_state(), c2.id_state()):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    bad = dict(np.load(p))
    bad["levels"] = bad["levels"] * np.float32(1.01)
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError):
        load_channel(tmp_path / "bad.npz")
    with pytest.raises(ValueError):
        RotatedChannel(np.zeros(2), [[1, 1], [0, 1]], [-1, -1], [1, 1], [4, 4])
    with pytest.raises(ValueError):
        ch.decode(np.full((1, d), 1 << 9))



def test_fit_pca_allocates_the_budget_and_never_clips():
    rng = np.random.default_rng(4)
    d = 8
    x = _unit_means(rng, 6000, d)
    ch = RotatedChannel.fit_pca(x[:4000], x[4000:5000], total_bits=64)
    assert ch.cost_bits() == 64 and (ch.bits >= 1).all()
    assert ch.bits[0] >= ch.bits[-1]           # more bits on wider axes
    assert np.allclose(ch.hi, 1 + np.linalg.norm(x[:4000].mean(0)))
    far = rng.normal(size=(500, d))
    far /= np.linalg.norm(far, axis=1, keepdims=True)
    assert not ch.clipped(far).any() and not ch.clipped(x[5000:]).any()
    e = np.linalg.norm(ch.round_trip(x[5000:]) - x[5000:], axis=1)
    assert np.median(e) < 0.02
