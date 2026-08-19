"""Transmitting the `stats_cond` mean channel.

A model trained with `stats_cond` has its per-window mean subtracted
before the encoder and added back after the decoder, so the mean is part
of the reconstruction rather than a rescaling of it: a release must
actually send it, and must charge for it.

Training uses the exact float mean and charges `bits_per_band` as
accounting only (see `georvq.model.stats_cond`). This module is what
makes that accounting true on the wire. It quantizes against **shipped
per-band constants** — a corpus mean and a corpus scale — rather than a
per-window scale, for two reasons:

* the cost is then exactly `bands x bits_per_band` bits, the number the
  rate ladder was computed with, with no per-window side info to explain
  away; and
* the mean's distribution across windows is a property of the corpus,
  not of the window, so a per-window scale would spend bits
  re-describing something already known.

The constants travel in the release bundle beside the codebooks. They
are part of what the decoder needs, so a change to them is a change to
the bitstream.
"""
from __future__ import annotations

import numpy as np


class MeanChannel:
    """Uniform per-band quantizer for the stats_cond mean vector.

    mu0/scale: (bands,) corpus mean and scale, from `fit`.
    bits:      per band, matching the model's `stats_cond.bits_per_band`.
    clip:      half-range in units of `scale`; values beyond it saturate.
    """

    def __init__(self, mu0, scale, bits: int, clip: float = 4.0):
        self.mu0 = np.asarray(mu0, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.bits = int(bits)
        self.clip = float(clip)
        if self.mu0.shape != self.scale.shape:
            raise ValueError("mu0 and scale must have the same shape")
        if not 1 <= self.bits <= 16:
            raise ValueError("bits must be in 1..16")

    @property
    def bands(self) -> int:
        return int(self.mu0.shape[0])

    @property
    def levels(self) -> int:
        return 1 << self.bits

    def cost_bits(self) -> int:
        """Bits per window. No per-window side info, by construction."""
        return self.bands * self.bits

    # ---- fit ----

    @classmethod
    def fit(cls, means, bits: int, clip: float = 4.0) -> "MeanChannel":
        """means: (N, bands) observed window means."""
        mu = np.asarray(means, dtype=np.float32)
        sd = mu.std(axis=0)
        # a band that never varies would give a zero step and divide by
        # zero; give it the smallest scale that still round-trips
        sd = np.maximum(sd, 1e-6)
        return cls(mu.mean(axis=0), sd, bits, clip)

    # ---- code ----

    def encode(self, mu) -> np.ndarray:
        """(N, bands) float -> (N, bands) uint16 codes in [0, levels)."""
        z = (np.asarray(mu, dtype=np.float32) - self.mu0) / self.scale
        u = np.clip(z / (2 * self.clip) + 0.5, 0.0, 1.0)     # -> [0, 1]
        return np.rint(u * (self.levels - 1)).astype(np.uint16)

    def decode(self, codes) -> np.ndarray:
        """Inverse of `encode` -> (N, bands) float32."""
        u = np.asarray(codes, dtype=np.float32) / (self.levels - 1)
        z = (u - 0.5) * (2 * self.clip)
        return (z * self.scale + self.mu0).astype(np.float32)

    def round_trip(self, mu) -> np.ndarray:
        """The mean the decoder will actually see. Encode against THIS,
        not the float mean, so both ends condition on the same vector."""
        return self.decode(self.encode(mu))

    # ---- persistence ----

    def save(self, path):
        np.savez(path, mu0=self.mu0, scale=self.scale,
                 bits=np.int32(self.bits), clip=np.float32(self.clip))

    @classmethod
    def load(cls, path) -> "MeanChannel":
        z = np.load(path)
        return cls(z["mu0"], z["scale"], int(z["bits"]), float(z["clip"]))
