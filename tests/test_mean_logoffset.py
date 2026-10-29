"""Log-offset mean channel: error bounds of the round trip, saturation only
outside the range, decode by table, and the saved tables."""
import numpy as np
import pytest

from georvq.inference.bitpack import pack, unpack
from georvq.inference.mean_codec import (LogOffsetChannel, channel_kind,
                                         load_channel)
from georvq.s2 import v1


def _s2_values(rng, n):
    """(n, 26) S2-like statistics in x = DN / 10000: means from 0 to the
    top of the range, deviations from the floor to the top."""
    mean = rng.uniform(0, v1.MEAN_MAX_DN, (n, 13))
    sd = np.exp(rng.uniform(np.log(v1.STD_MIN_DN), np.log(v1.STD_MAX_DN),
                            (n, 13)))
    return np.concatenate([mean, sd], 1) / 1e4


def test_s2_channel_cost_and_fields():
    ch = v1.stats_channel()
    assert ch.kind == "log-offset" and ch.bands == 26
    assert ch.cost_bits() == 13 * 13 + 13 * 10 == 299
    assert ch.record_fields() == [("log_0", 13, 13), ("log_13", 13, 10)]
    assert list(ch.field_widths()) == [13] * 13 + [10] * 13


def test_round_trip_error_bounds():
    ch = v1.stats_channel()
    rng = np.random.default_rng(0)
    x = _s2_values(rng, 20000)
    y = ch.round_trip(x).astype(np.float64)
    D, Dh = x[:, :13] * 1e4, y[:, :13] * 1e4
    # half a step of ln(D + 1000), plus float32 rounding of x
    half = 0.5 * ch.step[0] * (D + v1.MEAN_OFFSET_DN) * 1.001 + 0.01
    assert (np.abs(Dh - D) <= half).all()
    assert np.abs(Dh - D)[D < 100].max() < 0.21         # step 0.37 to 0.41 DN
    assert np.abs(Dh - D)[(D > 8900) & (D < 9100)].max() < 1.9
    r = y[:, 13:] / x[:, 13:]
    assert np.abs(np.log(r)).max() <= 0.5 * ch.step[13] * 1.001
    assert np.abs(r - 1).max() < 0.0046                 # 10 bits over 4 decades


def test_no_clipping_inside_the_range_and_saturation_outside():
    ch = v1.stats_channel()
    nb = 13
    x = np.zeros((4, 26))
    # bright cloud, a dark lake, the floor of the normaliser, the range ends
    x[0, :nb], x[0, nb:] = 9000 / 1e4, 3000 / 1e4
    x[1, :nb], x[1, nb:] = 1 / 1e4, np.float32(1e-4)
    x[2, :nb], x[2, nb:] = v1.MEAN_MAX_DN / 1e4, v1.STD_MAX_DN / 1e4
    x[3, :nb], x[3, nb:] = 0.0, v1.STD_MIN_DN / 1e4
    assert not ch.clipped(x).any()
    y = ch.round_trip(x)
    assert np.abs(y[0, :nb] * 1e4 - 9000).max() < 1.9
    assert np.abs(y[0, nb:] / x[0, nb:] - 1).max() < 0.0046
    # the ends decode exactly, the floor to the normaliser's own float32
    assert (y[3, :nb] == 0).all()
    assert (y[1, nb:] == np.float32(1e-4)).all()
    assert (y[2, :nb] == np.float32(v1.MEAN_MAX_DN / 1e4)).all()
    out = np.array(x[:1])
    out[0, 0], out[0, nb] = 30000 / 1e4, 20000 / 1e4
    assert ch.clipped(out)[0, [0, nb]].all()
    q = ch.encode(out)
    assert q[0, 0] == 8191 and q[0, nb] == 1023


def test_decode_is_the_table_and_codes_pack():
    ch = v1.stats_channel()
    rng = np.random.default_rng(1)
    q = ch.encode(_s2_values(rng, 500))
    assert q.dtype == np.int64 and q.min() >= 0
    assert (q < (1 << ch.bits)).all()
    y = ch.decode(q)
    assert y.dtype == np.float32
    assert np.array_equal(y, ch.levels[np.arange(26)[None], q])
    assert np.array_equal(ch.encode(y), q)              # codes are fixed points
    assert np.array_equal(unpack(pack(q, ch.field_widths()),
                                 ch.field_widths()), q)
    with pytest.raises(ValueError):
        ch.decode(np.full((1, 26), 1024))               # past the 10-bit values


def test_save_load_and_ids(tmp_path):
    ch = v1.stats_channel()
    p = tmp_path / "c.npz"
    ch.save(p)
    assert channel_kind(p) == "log-offset"
    c2 = load_channel(p, "log-offset")
    for a, b in zip(ch.id_state(), c2.id_state()):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    with pytest.raises(ValueError):
        load_channel(p, "rvq-pca")
    # a table that does not describe the parameters is refused
    bad = dict(np.load(p))
    bad["levels"] = bad["levels"] * np.float32(1.001)
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError):
        load_channel(tmp_path / "bad.npz")
    # other parameters give another table, and so other ids
    other = LogOffsetChannel(ch.scale, ch.offset * 2, ch.lo, ch.hi, ch.bits)
    assert not np.array_equal(other.levels, ch.levels)


def test_generic_parameters():
    ch = LogOffsetChannel(scale=[1.0, 100.0], offset=[0.0, 5.0],
                          lo=[0.5, 0.0], hi=[2.0, 50.0], bits=[3, 10])
    assert ch.cost_bits() == 13
    assert ch.record_fields() == [("log_0", 1, 3), ("log_1", 1, 10)]
    x = np.array([[0.5, 0.0], [2.0, 0.5], [1.0, 0.25]])
    y = ch.round_trip(x)
    assert y[0, 0] == np.float32(0.5) and y[1, 0] == np.float32(2.0)
    assert y[0, 1] == 0 and y[1, 1] == np.float32(0.5)
    assert abs(y[2, 0] - 1.0) <= np.exp(0.5 * ch.step[0]) - 1 + 1e-6
    for bad in (dict(lo=[0.0, 0.0]),                    # ln(0) with offset 0
                dict(hi=[0.5, 50.0]), dict(bits=[0, 10]),
                dict(bits=[17, 10])):
        kw = dict(scale=[1.0, 100.0], offset=[0.0, 5.0], lo=[0.5, 0.0],
                  hi=[2.0, 50.0], bits=[3, 10])
        kw.update(bad)
        with pytest.raises(ValueError):
            LogOffsetChannel(**kw)
