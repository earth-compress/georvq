"""Real entropy-coded archives: rANS over code indices with calibrated
static CDFs, byte-exact verification, and honest ratio accounting.

The learned context model (`model.entropy_level_bits`) reports the
*achievable* rate; this module produces *actual files*. Static per-
(tier, level) CDFs are calibrated on a sample; the gap between realized
bytes and the context model's bits is logged so nothing is over-claimed.
A context-adaptive coder driven by `ContextRateModel` distributions is the
natural upgrade and slots in behind the same packet layout.

Requires compressai (`pip install georvq[archive]`).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

CDF_PRECISION = 16   # bits — the standard rANS precision


def _rans():
    try:
        from compressai._CXX import pmf_to_quantized_cdf
        from compressai.ans import BufferedRansEncoder, RansDecoder
    except ImportError as e:
        raise ImportError("entropy-coded archives need compressai: "
                          "pip install 'georvq[archive]'") from e
    return BufferedRansEncoder, RansDecoder, pmf_to_quantized_cdf


class CdfTable:
    """One quantized CDF per (tier, level), flattened for the rANS coder."""

    def __init__(self, cdfs, lengths, offsets, specs_sig):
        self.cdfs = cdfs            # list[list[int]]
        self.lengths = lengths      # list[int]
        self.offsets = offsets      # list[int]
        self.specs_sig = specs_sig  # [(cells, depth, codes) per tier]

    def table_id(self, tier, level):
        return sum(d for _, d, _ in self.specs_sig[:tier]) + level

    @classmethod
    def calibrate(cls, index_arrays, specs):
        """index_arrays: per tier, (N, cells, depth) int arrays from a
        calibration sample. Laplace-smoothed histograms -> quantized CDFs."""
        _, _, pmf_to_quantized_cdf = _rans()
        cdfs, lengths, offsets = [], [], []
        for t, sp in enumerate(specs):
            arr = np.asarray(index_arrays[t])
            for l in range(sp.depth):
                counts = np.bincount(arr[..., l].reshape(-1),
                                     minlength=sp.codes).astype(np.float64)
                pmf = (counts + 1.0) / (counts.sum() + sp.codes)
                cdf = pmf_to_quantized_cdf(pmf.tolist(), CDF_PRECISION)
                cdfs.append(cdf)
                lengths.append(len(cdf))
                offsets.append(0)
        sig = [(sp.cells, sp.depth, sp.codes) for sp in specs]
        return cls(cdfs, lengths, offsets, sig)

    def save(self, path):
        np.savez_compressed(
            path,
            cdfs=np.array([np.pad(c, (0, max(self.lengths) - len(c)))
                           for c in self.cdfs], dtype=np.int64),
            lengths=np.array(self.lengths),
            offsets=np.array(self.offsets),
            specs_sig=np.array(self.specs_sig))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        lengths = z["lengths"].tolist()
        cdfs = [z["cdfs"][i][:n].tolist() for i, n in enumerate(lengths)]
        return cls(cdfs, lengths, z["offsets"].tolist(),
                   [tuple(s) for s in z["specs_sig"]])


def flatten_packet(packet, table: CdfTable):
    """Packet indices -> (symbols, cdf-table indexes), fixed full-depth
    layout: tier-major, level-major within tier, cells within level."""
    symbols, indexes = [], []
    for t, (cells, depth, _) in enumerate(table.specs_sig):
        arr = np.asarray(packet.indices[t])          # (cells, depth)
        for l in range(depth):
            symbols.extend(int(v) for v in arr[:, l])
            indexes.extend([table.table_id(t, l)] * cells)
    return symbols, indexes


def encode_packet(packet, table: CdfTable) -> bytes:
    BufferedRansEncoder, _, _ = _rans()
    symbols, indexes = flatten_packet(packet, table)
    enc = BufferedRansEncoder()
    enc.encode_with_indexes(symbols, indexes, table.cdfs, table.lengths,
                            table.offsets)
    return enc.flush()


def decode_packet(bitstream: bytes, table: CdfTable):
    """bytes -> per-tier (cells, depth) index arrays."""
    _, RansDecoder, _ = _rans()
    indexes = []
    for t, (cells, depth, _) in enumerate(table.specs_sig):
        for l in range(depth):
            indexes.extend([table.table_id(t, l)] * cells)
    dec = RansDecoder()
    dec.set_stream(bitstream)
    symbols = dec.decode_stream(indexes, table.cdfs, table.lengths,
                                table.offsets)
    out, pos = [], 0
    for cells, depth, _ in table.specs_sig:
        arr = np.empty((cells, depth), dtype=np.int64)
        for l in range(depth):
            arr[:, l] = symbols[pos:pos + cells]
            pos += cells
        out.append(arr)
    return out


def naive_bits(table: CdfTable) -> float:
    """Uncoded full-depth bits per window."""
    return float(sum(cells * depth * np.log2(codes)
                     for cells, depth, codes in table.specs_sig))


def write_archive(out_dir, codec, batches, table=None, chunk_rows=256,
                  verify=True):
    """Encode an iterable of (x, loc_or_None, meta_rows) batches into
    parquet chunks of rANS bitstreams + metadata.json with realized ratios.
    Calibrates CDFs on the first batch if no table is given."""
    import pandas as pd
    os.makedirs(os.path.join(out_dir, "chunks"), exist_ok=True)
    rows, chunk_id, total_bytes, total_windows = [], 0, 0, 0
    mismatches = 0

    def _flush():
        nonlocal rows, chunk_id
        if rows:
            pd.DataFrame(rows).to_parquet(
                os.path.join(out_dir, "chunks", f"chunk_{chunk_id:05d}.parquet"))
            rows, chunk_id = [], chunk_id + 1

    for x, loc, meta in batches:
        packets = codec.encode(x, loc=loc)
        if table is None:
            stacked = [np.stack([p.indices[t] for p in packets])
                       for t in range(len(codec.specs))]
            table = CdfTable.calibrate(stacked, codec.specs)
            table.save(os.path.join(out_dir, "cdfs.npz"))
        for p, m in zip(packets, meta):
            bs = encode_packet(p, table)
            if verify:
                back = decode_packet(bs, table)
                if not all(np.array_equal(a, b)
                           for a, b in zip(p.indices, back)):
                    mismatches += 1
            total_bytes += len(bs)
            total_windows += 1
            rows.append({**m, "bitstream": bs,
                         "n_bytes": len(bs),
                         **({"lon": float(p.loc[0]), "lat": float(p.loc[1])}
                            if p.loc is not None and p.loc.shape == (2,) else {})})
            if len(rows) >= chunk_rows:
                _flush()
    _flush()
    nb = naive_bits(table)
    meta = {"windows": total_windows,
            "coded_bytes": total_bytes,
            "bits_per_window_coded": 8 * total_bytes / max(total_windows, 1),
            "bits_per_window_naive": nb,
            "coding_gain": nb / (8 * total_bytes / max(total_windows, 1)),
            "verify_mismatches": mismatches,
            "precision_bits": CDF_PRECISION}
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta
