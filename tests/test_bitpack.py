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
