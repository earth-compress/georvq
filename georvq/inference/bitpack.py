"""Fixed-width bit packing for per-window records.

A record is a row of unsigned integer fields with known bit widths,
packed MSB-first in field order and padded with zero bits to a whole
byte. Every record of a layout has the same byte length, so a record's
offset in a section is plain arithmetic.
"""
from __future__ import annotations

import numpy as np


def record_bytes(widths) -> int:
    return (int(np.sum(widths)) + 7) // 8


def pack(values, widths) -> np.ndarray:
    """values: (N, F) non-negative ints, field f < 2**widths[f].
    Returns (N, record_bytes(widths)) uint8."""
    v = np.asarray(values, dtype=np.uint64)
    w = np.asarray(widths, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != len(w):
        raise ValueError(f"values {v.shape} do not match {len(w)} widths")
    if (w < 0).any() or (w > 32).any():
        raise ValueError("field widths must be in 0..32")
    if len(w) and (v >= (np.uint64(1) << w.astype(np.uint64))).any():
        raise ValueError("a value does not fit its field width")
    n, total = v.shape[0], int(w.sum())
    bits = np.zeros((n, record_bytes(w) * 8), dtype=np.uint8)
    pos = 0
    for f, wf in enumerate(w.tolist()):
        if wf == 0:
            continue
        shifts = np.arange(wf - 1, -1, -1, dtype=np.uint64)
        bits[:, pos:pos + wf] = ((v[:, f:f + 1] >> shifts) & 1).astype(np.uint8)
        pos += wf
    assert pos == total
    return np.packbits(bits, axis=1)


def unpack(data, widths) -> np.ndarray:
    """Inverse of `pack`: (N, record_bytes) uint8 -> (N, F) int64."""
    d = np.asarray(data, dtype=np.uint8)
    w = np.asarray(widths, dtype=np.int64)
    if d.ndim != 2 or d.shape[1] != record_bytes(w):
        raise ValueError(f"records of {d.shape[1:]} bytes, layout needs "
                         f"{record_bytes(w)}")
    bits = np.unpackbits(d, axis=1).astype(np.int64)
    out = np.zeros((d.shape[0], len(w)), dtype=np.int64)
    pos = 0
    for f, wf in enumerate(w.tolist()):
        if wf == 0:
            continue
        weights = (1 << np.arange(wf - 1, -1, -1, dtype=np.int64))
        out[:, f] = bits[:, pos:pos + wf] @ weights
        pos += wf
    return out
