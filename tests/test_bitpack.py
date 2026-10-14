"""Fixed-width record packing: exact round trip, MSB-first order, and a
byte length that depends only on the layout."""
import numpy as np
import pytest

from georvq.inference.bitpack import pack, record_bytes, unpack


def test_round_trip_mixed_widths():
    rng = np.random.default_rng(0)
    widths = [12, 12, 7, 0, 5, 1, 16, 3]
    v = np.stack([rng.integers(0, 1 << w if w else 1, 500) for w in widths], 1)
    data = pack(v, widths)
    assert data.shape == (500, record_bytes(widths)) == (500, 7)
    assert (unpack(data, widths) == v).all()


def test_msb_first_and_zero_padding():
    # 0b101 then 0b11111 -> 101 11111 = 0xBF; 0b1 -> 1 then 7 pad zeros
    data = pack([[5, 31, 1]], [3, 5, 1])
    assert data.tolist() == [[0xBF, 0x80]]


def test_rejects_values_that_overflow_their_field():
    with pytest.raises(ValueError):
        pack([[8]], [3])


def _unpack_per_bit(data, widths):
    """The original unpack, one int64 per bit, as the reference."""
    d = np.asarray(data, dtype=np.uint8)
    bits = np.unpackbits(d, axis=1).astype(np.int64)
    out = np.zeros((d.shape[0], len(widths)), dtype=np.int64)
    pos = 0
    for f, wf in enumerate(widths):
        if wf == 0:
            continue
        out[:, f] = bits[:, pos:pos + wf] @ (
            1 << np.arange(wf - 1, -1, -1, dtype=np.int64))
        pos += wf
    return out


def test_unpack_matches_the_per_bit_reference():
    rng = np.random.default_rng(1)
    cases = [[32], [32, 32, 1], [1] * 17, [0, 0, 3], [14] * 256, [7, 32, 9]]
    cases += [rng.integers(0, 33, int(rng.integers(1, 40))).tolist()
              for _ in range(200)]
    for widths in cases:
        n = int(rng.integers(1, 64))
        # arbitrary bytes, padding bits included, not only packed values
        data = rng.integers(0, 256, (n, record_bytes(widths)), dtype=np.uint8)
        got = unpack(data, widths)
        assert got.dtype == np.int64
        assert np.array_equal(got, _unpack_per_bit(data, widths)), widths
        v = np.stack([rng.integers(0, 1 << w, n) if w else np.zeros(n, int)
                      for w in widths], 1)
        assert np.array_equal(unpack(pack(v, widths), widths), v), widths
    empty = np.zeros((0, record_bytes([5, 12])), np.uint8)
    assert unpack(empty, [5, 12]).shape == (0, 2)
