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

`LogOffsetChannel` codes each value instead over a fixed absolute
range, on a logarithmic scale with an offset, so that it has no corpus
constants and cannot saturate on a value that is in range but unlike the
corpus. Its parameters and decode tables travel in the bundle in the
same way.

`RotatedChannel` rotates the vector by a fixed orthogonal basis and
codes each component over a fixed range, optionally companded, so that
with hard bounds of the data (those of a mean of unit vectors, or of
values that lie in a fixed box) it cannot saturate either.

The coders here share one interface, which the store codec
(`georvq.inference.store_codec`) relies on: `kind`, `bands`,
`record_fields()` (the fields of its part of a window's base record, as
(name, count, bits)), `field_widths()`, `encode`, `decode`,
`round_trip`, `cost_bits`, `id_state()` (the arrays `bitstream_id`
hashes), `save` and `load`. `load_channel(path)` reads any kind from
its `.npz`.
"""
from __future__ import annotations

import numpy as np


class MeanChannel:
    """Uniform per-band quantizer for the stats_cond mean vector.

    mu0/scale: (bands,) corpus mean and scale, from `fit`.
    bits:      per band, matching the model's `stats_cond.bits_per_band`.
    clip:      half-range in units of `scale`; values beyond it saturate.
    """

    kind = "uniform"

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

    def field_widths(self) -> np.ndarray:
        """Bit widths of a packed record: one code per band."""
        return np.full(self.bands, self.bits, np.int64)

    def record_fields(self) -> list:
        """The channel's fields in a base record: one code per band."""
        return [("mean", self.bands, self.bits)]

    def id_state(self) -> list:
        """What a decoder must agree on, for `bitstream_id(extra=...)`."""
        return [self.mu0, self.scale, np.int32(self.bits),
                np.float32(self.clip)]

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
        z = np.load(path, allow_pickle=False)
        return cls(z["mu0"], z["scale"], int(z["bits"]), float(z["clip"]))


class LogOffsetChannel:
    """Fixed-range quantizer of each value's logarithm, with an offset.

    Value d of a window, x in model units, is coded from v = scale[d] * x
    (for example DN), clipped to [lo[d], hi[d]], as

        q = rint((ln(v + offset[d]) - ln(lo[d] + offset[d])) / step[d]),
        step[d] = (ln(hi[d] + offset[d]) - ln(lo[d] + offset[d]))
                  / (2 ** bits[d] - 1),

    so the reconstruction error is about step[d] * (v + offset[d]) / 2:
    nearly uniform below the offset and relative above it. With offset 0
    it codes ln(v) uniformly. The ranges are absolute rather than fitted
    to a corpus, so no value inside them clips, however far it lies from
    the data the codec was trained on, and every window costs exactly
    `cost_bits()`.

    scale, offset, lo, hi: (D,) float64; bits: (D,) in 1..16, with
    lo + offset > 0 and hi > lo.

    Decoding is a table lookup. The tables (`levels`, the reconstruction
    of every code of every value in model units, float32, padded to the
    widest code) are computed once in float64 and saved with the
    parameters, and a decoder reads them rather than recomputing them,
    so every backend decodes the same floats; `load` checks that the
    saved tables match the parameters. Encoding rounds in float64, so a
    value within rounding of a decision boundary may take either
    neighbouring code on different hardware, which changes the code
    sent, not how a code decodes.
    """

    kind = "log-offset"

    def __init__(self, scale, offset, lo, hi, bits, levels=None):
        self.scale = np.asarray(scale, dtype=np.float64)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self.bits = np.asarray(bits, dtype=np.int64)
        D = self.bits.shape[0] if self.bits.ndim == 1 else -1
        for name, a in (("scale", self.scale), ("offset", self.offset),
                        ("lo", self.lo), ("hi", self.hi), ("bits", self.bits)):
            if a.ndim != 1 or a.shape[0] != D:
                raise ValueError(f"{name} must have shape (D,) like bits")
        if (self.bits < 1).any() or (self.bits > 16).any():
            raise ValueError("bits must be in 1..16")
        if not (self.scale > 0).all():
            raise ValueError("scale must be positive")
        if not (self.lo + self.offset > 0).all():
            raise ValueError("lo + offset must be positive")
        if not (self.hi > self.lo).all():
            raise ValueError("hi must exceed lo")
        self.step = ((np.log(self.hi + self.offset)
                      - np.log(self.lo + self.offset))
                     / ((1 << self.bits) - 1))
        want = self._tables()
        if levels is None:
            self.levels = want
        else:
            self.levels = np.asarray(levels, dtype=np.float32)
            if self.levels.shape != want.shape:
                raise ValueError(f"levels must have shape {want.shape}")
            # saved tables are used as they are; they must describe the
            # parameters saved with them to within float32 rounding
            if not np.allclose(self.levels, want, rtol=1e-6, atol=0):
                raise ValueError("levels do not match the parameters")

    def _tables(self) -> np.ndarray:
        """(D, 2 ** max(bits)) float32 reconstruction of every code, in
        model units; entries past a value's last code repeat it."""
        n = 1 << int(self.bits.max())
        q = np.minimum(np.arange(n)[None], ((1 << self.bits) - 1)[:, None])
        a = np.log(self.lo + self.offset)[:, None]
        v = np.exp(a + q * self.step[:, None]) - self.offset[:, None]
        # the ends of the range exactly, so that lo (a window's floor,
        # often) and hi decode to themselves
        v[q == 0] = np.broadcast_to(self.lo[:, None], v.shape)[q == 0]
        top = q == ((1 << self.bits) - 1)[:, None]
        v[top] = np.broadcast_to(self.hi[:, None], v.shape)[top]
        return (v / self.scale[:, None]).astype(np.float32)

    @property
    def bands(self) -> int:
        return int(self.bits.shape[0])

    def field_widths(self) -> np.ndarray:
        """Bit widths of a packed record: one code per value."""
        return self.bits.copy()

    def cost_bits(self) -> int:
        return int(self.bits.sum())

    def record_fields(self) -> list:
        """The channel's fields in a base record: runs of values sharing a
        bit width, named by their first value."""
        out = []
        bits = [int(b) for b in self.bits]
        k = 0
        while k < len(bits):
            j = k
            while j < len(bits) and bits[j] == bits[k]:
                j += 1
            out.append((f"log_{k}", j - k, bits[k]))
            k = j
        return out

    def id_state(self) -> list:
        """What a decoder must agree on, for `bitstream_id(extra=...)`."""
        return [self.scale, self.offset, self.lo, self.hi,
                self.bits.astype(np.int32), self.levels]

    # ---- code ----

    def clipped(self, x) -> np.ndarray:
        """(N, D) bool: values outside [lo, hi] by more than float32
        rounding, which `encode` saturates."""
        v = np.asarray(x, dtype=np.float64) * self.scale
        tol = 1e-6 * np.maximum(np.abs(self.lo), np.abs(self.hi))
        return (v < self.lo - tol) | (v > self.hi + tol)

    def encode(self, x) -> np.ndarray:
        """(N, D) float -> (N, D) int64 codes, value d in [0, 2 ** bits[d])."""
        v = np.clip(np.asarray(x, dtype=np.float64) * self.scale, self.lo,
                    self.hi)
        u = (np.log(v + self.offset) - np.log(self.lo + self.offset)) \
            / self.step
        return np.clip(np.rint(u), 0, (1 << self.bits) - 1).astype(np.int64)

    def decode(self, codes) -> np.ndarray:
        """Inverse of `encode` -> (N, D) float32, by table lookup."""
        q = np.asarray(codes, dtype=np.int64)
        if q.ndim != 2 or q.shape[1] != self.bands:
            raise ValueError(f"codes must have shape (N, {self.bands})")
        if (q < 0).any() or (q >= (1 << self.bits)).any():
            raise ValueError("code out of range")
        return self.levels[np.arange(self.bands)[None], q]

    def round_trip(self, x) -> np.ndarray:
        """The values the decoder will see."""
        return self.decode(self.encode(x))

    # ---- persistence ----

    def save(self, path):
        np.savez(path, scale=self.scale, offset=self.offset, lo=self.lo,
                 hi=self.hi, bits=self.bits.astype(np.int32),
                 levels=self.levels)

    @classmethod
    def load(cls, path) -> "LogOffsetChannel":
        z = np.load(path, allow_pickle=False)
        return cls(z["scale"], z["offset"], z["lo"], z["hi"], z["bits"],
                   z["levels"])


class RotatedChannel:
    """Fixed rotation, then a fixed-range scalar quantizer per component,
    optionally companded.

    Each value x (D,) is rotated, y = (x - mu0) @ V with V orthogonal
    (columns), and component k is clipped to [lo[k], hi[k]] and coded in
    bits[k] bits as

        u = f_k(y),  q = rint((u - f_k(lo)) / step),
        step = (f_k(hi) - f_k(lo)) / (2 ** bits - 1),

    with f_k(y) = y when scale[k] is 0 (uniform steps) and
    f_k(y) = asinh(y / scale[k]) otherwise, which gives steps of about
    scale[k] * step near zero that widen logarithmically towards the ends
    of the range. When the ranges are hard bounds of the data (for unit
    vectors and their means, |y| <= 1 + |mu0| on every component), no
    value is ever clipped, and every window costs exactly `cost_bits()`.

    mu0: (D,); V: (D, D); lo, hi, scale: (D,) float64; bits: (D,) in
    1..16. V = I and scale 0 is a per-value uniform quantizer.

    Decoding looks up each component in float32 tables saved with the
    parameters (`levels`, as LogOffsetChannel does) and rotates back in
    float32, x = mu0 + y @ V.T. Encoding rounds in float64, so a value
    within rounding of a decision boundary may take either neighbouring
    code on different hardware.
    """

    kind = "rotated"

    def __init__(self, mu0, V, lo, hi, bits, scale=None, levels=None):
        self.mu0 = np.asarray(mu0, dtype=np.float32)
        self.V = np.asarray(V, dtype=np.float32)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self.bits = np.asarray(bits, dtype=np.int64)
        D = self.mu0.shape[0]
        self.scale = (np.zeros(D) if scale is None
                      else np.asarray(scale, dtype=np.float64))
        for name, a in (("lo", self.lo), ("hi", self.hi), ("bits", self.bits),
                        ("scale", self.scale)):
            if a.shape != (D,):
                raise ValueError(f"{name} must have shape ({D},)")
        if self.V.shape != (D, D):
            raise ValueError(f"V must have shape ({D}, {D})")
        if not np.allclose(self.V.astype(np.float64).T @ self.V, np.eye(D),
                           atol=1e-5):
            raise ValueError("V must be orthogonal")
        if (self.bits < 1).any() or (self.bits > 16).any():
            raise ValueError("bits must be in 1..16")
        if not (self.hi > self.lo).all():
            raise ValueError("hi must exceed lo")
        if (self.scale < 0).any():
            raise ValueError("scale must be non-negative")
        self.step = (self._f(self.hi) - self._f(self.lo)) / \
            ((1 << self.bits) - 1)
        want = self._tables()
        if levels is None:
            self.levels = want
        else:
            self.levels = np.asarray(levels, dtype=np.float32)
            if self.levels.shape != want.shape:
                raise ValueError(f"levels must have shape {want.shape}")
            if not np.allclose(self.levels, want, rtol=1e-6, atol=1e-9):
                raise ValueError("levels do not match the parameters")

    def _f(self, y):
        a = self.scale
        lin = a == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.arcsinh(y / np.where(lin, 1.0, a))
        return np.where(lin, y, u)

    def _finv(self, u):
        a = self.scale
        return np.where(a == 0, u, np.sinh(u) * a)

    def _tables(self) -> np.ndarray:
        n = 1 << int(self.bits.max())
        q = np.minimum(np.arange(n)[None], ((1 << self.bits) - 1)[:, None])
        u = self._f(self.lo)[:, None] + q * self.step[:, None]
        a = self.scale[:, None]
        y = np.where(a == 0, u, np.sinh(u) * np.where(a == 0, 1.0, a))
        top = q == ((1 << self.bits) - 1)[:, None]
        y[q == 0] = np.broadcast_to(self.lo[:, None], y.shape)[q == 0]
        y[top] = np.broadcast_to(self.hi[:, None], y.shape)[top]
        return y.astype(np.float32)

    @property
    def bands(self) -> int:
        return int(self.mu0.shape[0])

    def field_widths(self) -> np.ndarray:
        return self.bits.copy()

    def cost_bits(self) -> int:
        return int(self.bits.sum())

    def record_fields(self) -> list:
        """Runs of components sharing a bit width, named by their first."""
        out = []
        bits = [int(b) for b in self.bits]
        k = 0
        while k < len(bits):
            j = k
            while j < len(bits) and bits[j] == bits[k]:
                j += 1
            out.append((f"rot_{k}", j - k, bits[k]))
            k = j
        return out

    def id_state(self) -> list:
        return [self.mu0, self.V, self.lo, self.hi, self.bits.astype(np.int32),
                self.scale, self.levels]

    def rotate(self, x) -> np.ndarray:
        """(N, D) -> (N, D) float64 components y."""
        return (np.asarray(x, dtype=np.float64) - self.mu0) @ \
            self.V.astype(np.float64)

    def clipped(self, x) -> np.ndarray:
        """(N, D) bool: components outside [lo, hi] beyond rounding."""
        y = self.rotate(x)
        tol = 1e-6 * np.maximum(np.abs(self.lo), np.abs(self.hi))
        return (y < self.lo - tol) | (y > self.hi + tol)

    def encode(self, x) -> np.ndarray:
        y = np.clip(self.rotate(x), self.lo, self.hi)
        u = (self._f(y) - self._f(self.lo)) / self.step
        return np.clip(np.rint(u), 0, (1 << self.bits) - 1).astype(np.int64)

    def decode(self, codes) -> np.ndarray:
        q = np.asarray(codes, dtype=np.int64)
        if q.ndim != 2 or q.shape[1] != self.bands:
            raise ValueError(f"codes must have shape (N, {self.bands})")
        if (q < 0).any() or (q >= (1 << self.bits)).any():
            raise ValueError("code out of range")
        y = self.levels[np.arange(self.bands)[None], q]
        return (self.mu0 + y @ self.V.T).astype(np.float32)

    def round_trip(self, x) -> np.ndarray:
        return self.decode(self.encode(x))

    def save(self, path):
        np.savez(path, mu0=self.mu0, V=self.V, lo=self.lo, hi=self.hi,
                 bits=self.bits.astype(np.int32), scale=self.scale,
                 levels=self.levels)

    @classmethod
    def fit_pca(cls, pca_means, alloc_means, total_bits, compand=True,
                bmax=16, box=None) -> "RotatedChannel":
        """A channel for window means: mu0 and V the mean and principal
        axes of `pca_means` (N, D), every component bounded by a hard
        bound, companded by its standard deviation when `compand`, and
        `total_bits` shared greedily (each component at least 1 bit) by
        the squared error each extra bit removes on `alloc_means`.

        Without `box` the means are of unit vectors, and the bound is
        |y| <= 1 + |mu0| on every component. With `box`, a scalar or (D,),
        every value of every pixel lies in [-box, box], so the window mean
        does too, and component k of y = (x - mu0) @ V is bounded by
        +-|V[:, k]| . box - mu0 . V[:, k], computed from the float32 mu0
        and V the channel stores."""
        X = np.asarray(pca_means, dtype=np.float64)
        mu0 = X.mean(0)
        var, V = np.linalg.eigh(np.cov((X - mu0).T))
        o = np.argsort(var)[::-1]
        V, sd = V[:, o], np.sqrt(var[o])
        D = X.shape[1]
        if box is None:
            B = 1.0 + float(np.linalg.norm(mu0))
            lo, hi = -np.full(D, B), np.full(D, B)
        else:
            b = np.broadcast_to(np.asarray(box, np.float64), (D,))
            V32 = V.astype(np.float32).astype(np.float64)
            c = mu0.astype(np.float32).astype(np.float64) @ V32
            w = np.abs(V32).T @ b
            lo, hi = -w - c, w - c
        scale = sd if compand else np.zeros(D)
        Y = (np.asarray(alloc_means, dtype=np.float64) - mu0) @ V

        def dist(k, b):
            one = cls(np.zeros(1), np.eye(1), [lo[k]], [hi[k]], [b],
                      [scale[k]])
            y = Y[:, k:k + 1]
            r = one.levels[0][one.encode(y)[:, 0]]
            return float(((y[:, 0] - r) ** 2).mean())
        tab = np.array([[dist(k, b) if b else 0.0 for b in range(bmax + 1)]
                        for k in range(D)])
        bits = np.ones(D, int)
        if total_bits < D:
            raise ValueError(f"total_bits must be at least {D}")
        while bits.sum() < total_bits:
            g = [(tab[k, bits[k]] - tab[k, bits[k] + 1]) if bits[k] < bmax
                 else -1.0 for k in range(D)]
            bits[int(np.argmax(g))] += 1
        return cls(mu0, V, lo, hi, bits, scale)

    @classmethod
    def load(cls, path) -> "RotatedChannel":
        z = np.load(path, allow_pickle=False)
        return cls(z["mu0"], z["V"], z["lo"], z["hi"], z["bits"], z["scale"],
                   z["levels"])


CHANNELS = {MeanChannel.kind: MeanChannel,
            LogOffsetChannel.kind: LogOffsetChannel,
            RotatedChannel.kind: RotatedChannel}


def channel_kind(path) -> str:
    """The kind of the mean channel saved at `path`, told apart by the
    tables it holds."""
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
    if {"V", "levels", "lo", "hi"} <= names:
        return RotatedChannel.kind
    if {"levels", "offset", "lo", "hi"} <= names:
        return LogOffsetChannel.kind
    if {"mu0", "scale"} <= names:
        return MeanChannel.kind
    raise ValueError(f"{path}: not a saved mean channel (arrays "
                     f"{sorted(names)})")


def load_channel(path, kind=None):
    """A mean channel from its `.npz`. `kind`, when given (a bundle
    records it), must agree with the tables the file holds."""
    found = channel_kind(path)
    if kind is not None and kind != found:
        raise ValueError(f"{path}: holds a {found!r} mean channel, "
                         f"{kind!r} was expected")
    return CHANNELS[found].load(path)
