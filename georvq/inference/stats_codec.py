"""Stats side-channel codec: adaptive-PCA compression of per-tile
normalization statistics.

Sources whose tiles are normalized before encoding (e.g. per-tile per-band
mean/std) must ship those stats alongside the code indices. `StatsPacker`
projects a stats vector onto a PCA basis and sends the minimum number of
8-bit-quantized components that reconstructs within tolerance (~11 bytes
for a 26-float vector), with a raw-float16 escape hatch.

Packet format: [n: 1 byte] [component bytes]; n=255 marks the raw fallback.
"""
from __future__ import annotations

import struct

import numpy as np


class StatsPacker:
    def __init__(self, pca_path=None, tolerance=0.01, data=None):
        data = data if data is not None else np.load(pca_path)
        self.components = np.asarray(data["components"])   # (N_max, D)
        self.mean = np.asarray(data["mean"])               # (D,)
        self.comp_min = np.asarray(data["comp_min"])       # (N_max,)
        self.comp_max = np.asarray(data["comp_max"])       # (N_max,)
        self.n_max = int(data["n_components"])
        self.dim = self.mean.shape[0]
        self.tolerance = tolerance

    def _quantize_components(self, projected, n):
        """Quantize n PCA components to uint8; return (quantized, dequantized)."""
        quantized = np.zeros(n, dtype=np.uint8)
        dequantized = np.zeros(n, dtype=np.float32)
        for i in range(n):
            rng = self.comp_max[i] - self.comp_min[i]
            normed = (projected[i] - self.comp_min[i]) / (rng + 1e-8)
            quantized[i] = int(np.clip(np.round(normed * 255), 0, 255))
            dequantized[i] = (quantized[i] / 255.0) * rng + self.comp_min[i]
        return quantized, dequantized

    def pack(self, stats: np.ndarray) -> bytes:
        """Compress a (D,) stats vector to bytes."""
        centered = stats - self.mean
        projected = centered @ self.components.T           # (N_max,)
        # minimum components for tolerance, checking error AFTER quantization
        for n in range(1, self.n_max + 1):
            _, dequantized = self._quantize_components(projected, n)
            recon = dequantized @ self.components[:n] + self.mean
            if np.abs(stats - recon).max() < self.tolerance:
                break
        else:
            return self.pack_raw(stats)     # PCA can't hit tolerance
        quantized, _ = self._quantize_components(projected, n)
        return struct.pack("B", n) + quantized.tobytes()

    def pack_raw(self, stats: np.ndarray) -> bytes:
        return struct.pack("B", 255) + stats.astype(np.float16).tobytes()

    def unpack(self, data: bytes) -> np.ndarray:
        """Packed stats bytes -> the (dim,) stats vector.

        `n` is the number of PCA components actually sent; `n == 255` is the
        escape hatch meaning the packet carries a raw float16 vector
        instead, used when the basis cannot represent the tile well enough.
        Inverse of `pack`.
        """
        n = struct.unpack("B", data[:1])[0]
        if n == 255:
            return np.frombuffer(data[1:1 + 2 * self.dim],
                                 dtype=np.float16).astype(np.float32)
        quantized = np.frombuffer(data[1:1 + n], dtype=np.uint8).astype(np.float32)
        projected = np.zeros(n, dtype=np.float32)
        for i in range(n):
            rng = self.comp_max[i] - self.comp_min[i]
            projected[i] = (quantized[i] / 255.0) * rng + self.comp_min[i]
        return projected @ self.components[:n] + self.mean

    def pack_batch(self, stats: np.ndarray) -> list[bytes]:
        return [self.pack(s) for s in stats]

    def unpack_batch(self, packets: list[bytes]) -> np.ndarray:
        return np.stack([self.unpack(p) for p in packets])


def fit_stats_pca(stats: np.ndarray, n_components=None) -> dict:
    """Fit the PCA basis for a StatsPacker on a (N, D) stats sample.
    Returns the dict to save as stats_pca.npz."""
    stats = np.asarray(stats, dtype=np.float64)
    mean = stats.mean(0)
    _, _, vt = np.linalg.svd(stats - mean, full_matrices=False)
    n = n_components or vt.shape[0]
    comps = vt[:n]
    proj = (stats - mean) @ comps.T
    return {"components": comps.astype(np.float32),
            "mean": mean.astype(np.float32),
            "comp_min": proj.min(0).astype(np.float32),
            "comp_max": proj.max(0).astype(np.float32),
            "n_components": n}
