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

Both coders here share one interface, which the store codec
(`georvq.inference.store_codec`) relies on: `kind`, `bands`,
`record_fields()` (the fields of its part of a window's base record, as
(name, count, bits)), `field_widths()`, `encode`, `decode`,
`round_trip`, `cost_bits`, `id_state()` (the arrays `bitstream_id`
hashes), `save` and `load`. `load_channel(path)` reads either kind from
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


def _uniform(c, s, b, clip):
    """Fixed-length b-bit uniform quantizer on [-clip*s, clip*s] with
    midpoint reconstruction: (codes, reconstruction)."""
    if b == 0:
        return np.zeros(c.shape, np.int64), np.zeros_like(c)
    n = 1 << b
    lo, w = -clip * s, 2 * clip * s
    q = np.clip(np.floor((c - lo) / w * n), 0, n - 1).astype(np.int64)
    return q, (lo + (q + 0.5) * w / n).astype(c.dtype)


class RVQPCAMeanChannel:
    """Residual-VQ then PCA coder for the stats_cond mean vector.

    A few RVQ levels capture the mean's non-linear structure (land-cover
    clusters); what they leave is coded in a PCA basis fitted on that
    residual, with a fixed-length uniform quantizer per component. The
    bit allocation is fixed when the channel is built, so every window
    costs exactly `cost_bits()` and packs to `record_bytes()` bytes.

    books: (L, K, D) RVQ codebooks, K a power of two.
    mu0:   (D,) mean of the RVQ residual; V: (D, D) its PCA basis (columns,
           decreasing variance); s: (D,) per-component std.
    bits:  (D,) bits per component (0 = not sent); clip: (D,) per-component
           half-range in units of s, used at that component's bit count.

    pre_scale, pre_log: optional (D,) fixed transform into the space the
           tables were fitted in, y = log(x) * pre_scale on the dimensions
           where pre_log is set and y = x * pre_scale elsewhere. It lets one
           coder weight its dimensions unequally (a per-dimension error
           budget) or code positive quantities to a relative error. Without
           it the tables code x directly, as for AE and Tessera, and the
           saved file holds no transform arrays.

    Encode searches the RVQ levels greedily in float32; near-ties may
    resolve differently across hardware, which changes which (valid)
    codes are sent, not how they decode. Decode is a fixed float32
    computation from the codes and the shipped tables.
    """

    kind = "rvq-pca"

    def __init__(self, books, mu0, V, s, bits, clip, pre_scale=None,
                 pre_log=None):
        self.books = np.asarray(books, dtype=np.float32)
        self.mu0 = np.asarray(mu0, dtype=np.float32)
        self.V = np.asarray(V, dtype=np.float32)
        self.s = np.asarray(s, dtype=np.float32)
        self.bits = np.asarray(bits, dtype=np.int64)
        self.clip = np.asarray(clip, dtype=np.float32)
        L, K, D = self.books.shape
        if K & (K - 1):
            raise ValueError("codebook size must be a power of two")
        for name, a in (("mu0", self.mu0), ("s", self.s), ("bits", self.bits),
                        ("clip", self.clip)):
            if a.shape != (D,):
                raise ValueError(f"{name} must have shape ({D},)")
        if self.V.shape != (D, D):
            raise ValueError(f"V must have shape ({D}, {D})")
        if (self.bits < 0).any() or (self.bits > 16).any():
            raise ValueError("component bits must be in 0..16")
        self.pre_scale = self.pre_log = None
        if pre_scale is not None or pre_log is not None:
            self.pre_scale = (np.ones(D, np.float32) if pre_scale is None
                              else np.asarray(pre_scale, dtype=np.float32))
            self.pre_log = (np.zeros(D, bool) if pre_log is None
                            else np.asarray(pre_log, dtype=bool))
            if self.pre_scale.shape != (D,) or self.pre_log.shape != (D,):
                raise ValueError(f"pre_scale and pre_log must have shape "
                                 f"({D},)")
            if not (self.pre_scale > 0).all():
                raise ValueError("pre_scale must be positive")

    @property
    def bands(self) -> int:
        return int(self.books.shape[2])

    @property
    def index_bits(self) -> int:
        return int(self.books.shape[1]).bit_length() - 1

    def field_widths(self) -> np.ndarray:
        """Bit widths of a packed record: RVQ indices, then components."""
        L = self.books.shape[0]
        return np.concatenate([np.full(L, self.index_bits), self.bits])

    def cost_bits(self) -> int:
        return int(self.field_widths().sum())

    def record_bytes(self) -> int:
        from georvq.inference.bitpack import record_bytes
        return record_bytes(self.field_widths())

    def record_fields(self) -> list:
        """The channel's fields in a base record: the RVQ indices, then
        runs of PCA components sharing a bit width, named by their first
        component."""
        out = [("mean_rvq", int(self.books.shape[0]), self.index_bits)]
        bits = [int(b) for b in self.bits]
        k = 0
        while k < len(bits):
            j = k
            while j < len(bits) and bits[j] == bits[k]:
                j += 1
            out.append((f"mean_pca_{k}", j - k, bits[k]))
            k = j
        return out

    # ---- transform ----

    def to_coded(self, x) -> np.ndarray:
        """(N, D) values -> the space the tables code (identity without a
        transform)."""
        return coded_space(x, self.pre_scale, self.pre_log)

    def from_coded(self, y) -> np.ndarray:
        """Inverse of `to_coded`."""
        y = np.asarray(y, dtype=np.float32)
        if self.pre_scale is None:
            return y
        x = y / self.pre_scale
        x[:, self.pre_log] = np.exp(x[:, self.pre_log])
        return x.astype(np.float32)

    # ---- code ----

    def encode(self, mu) -> np.ndarray:
        """(N, D) float -> (N, L + D) int64 fields: RVQ indices, then
        per-component codes (0 for components given no bits)."""
        r = self.to_coded(mu)
        idx = []
        for C in self.books:
            d = ((r * r).sum(1, keepdims=True) - 2 * r @ C.T
                 + (C * C).sum(1)[None])
            j = d.argmin(1)
            idx.append(j)
            r = r - C[j]
        c = (r - self.mu0) @ self.V
        q = np.zeros(c.shape, np.int64)
        for k in range(c.shape[1]):
            q[:, k], _ = _uniform(c[:, k], self.s[k], int(self.bits[k]),
                                  float(self.clip[k]))
        return np.concatenate([np.stack(idx, 1), q], 1)

    def decode(self, fields) -> np.ndarray:
        """Inverse of `encode` -> (N, D) float32."""
        f = np.asarray(fields, dtype=np.int64)
        L = self.books.shape[0]
        out = np.zeros((f.shape[0], self.bands), np.float32)
        for l in range(L):
            out += self.books[l][f[:, l]]
        ch = np.zeros((f.shape[0], self.bands), np.float32)
        for k in range(self.bands):
            b = int(self.bits[k])
            if b:
                n = 1 << b
                lo, w = -self.clip[k] * self.s[k], 2 * self.clip[k] * self.s[k]
                ch[:, k] = lo + (f[:, L + k] + 0.5) * w / n
        return self.from_coded(out + self.mu0 + ch @ self.V.T)

    def round_trip(self, mu) -> np.ndarray:
        """The mean the decoder will see; encode the window against this."""
        return self.decode(self.encode(mu))

    def pack(self, fields) -> np.ndarray:
        from georvq.inference.bitpack import pack
        return pack(fields, self.field_widths())

    def unpack(self, data) -> np.ndarray:
        from georvq.inference.bitpack import unpack
        return unpack(data, self.field_widths())

    # ---- building the residual coder ----

    @classmethod
    def fit_residual(cls, books, residual, total_bits: int,
                     clips=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
                     bmax: int = 14, pre_scale=None,
                     pre_log=None) -> "RVQPCAMeanChannel":
        """PCA coder for `residual` (N, D), the means minus their RVQ
        reconstruction, with `total_bits - L * index_bits` bits shared
        across components greedily from an empirical distortion table
        (1- or 2-bit steps, best drop per bit). Each (component, bits)
        pair keeps the clip range with the lowest distortion. With a
        transform (`pre_scale`, `pre_log`), `books` and `residual` are in
        the coded space (`coded_space`), and the channel carries the
        transform."""
        books = np.asarray(books, dtype=np.float32)
        L, K, D = books.shape
        budget = int(total_bits) - L * (int(K).bit_length() - 1)
        if budget < 0:
            raise ValueError("total_bits is below the RVQ index cost")
        X = np.asarray(residual, dtype=np.float64)
        mu0 = X.mean(0)
        Xc = X - mu0
        var, V = np.linalg.eigh(Xc.T @ Xc / len(Xc))
        o = np.argsort(var)[::-1]
        V, var = V[:, o], np.maximum(var[o], 1e-20)
        s = np.sqrt(var)
        C = Xc @ V
        Dt = np.zeros((D, bmax + 1))
        Ct = np.ones((D, bmax + 1)) * clips[-1]
        for k in range(D):
            Dt[k, 0] = (C[:, k] ** 2).mean()
            for b in range(1, bmax + 1):
                errs = [((C[:, k] - _uniform(C[:, k], s[k], b, cl)[1]) ** 2)
                        .mean() for cl in clips]
                j = int(np.argmin(errs))
                Dt[k, b], Ct[k, b] = errs[j], clips[j]
        bits = np.zeros(D, np.int64)
        left = budget
        while left > 0:
            best = None
            for k in range(D):
                for step in (1, 2):
                    if step > left or bits[k] + step > bmax:
                        continue
                    g = (Dt[k, bits[k]] - Dt[k, bits[k] + step]) / step
                    if best is None or g > best[0]:
                        best = (g, k, step)
            if best is None or best[0] <= 0:
                break
            _, k, step = best
            bits[k] += step
            left -= step
        clip = np.array([Ct[k, bits[k]] for k in range(D)])
        return cls(books, mu0, V, s, bits, clip, pre_scale, pre_log)

    # ---- persistence ----

    def _transform_arrays(self) -> dict:
        if self.pre_scale is None:
            return {}
        return {"pre_scale": self.pre_scale,
                "pre_log": self.pre_log.astype(np.uint8)}

    def save(self, path):
        np.savez(path, books=self.books, mu0=self.mu0, V=self.V, s=self.s,
                 bits=self.bits.astype(np.int32), clip=self.clip,
                 **self._transform_arrays())

    @classmethod
    def load(cls, path) -> "RVQPCAMeanChannel":
        z = np.load(path, allow_pickle=False)
        pre = ((z["pre_scale"], z["pre_log"].astype(bool))
               if "pre_scale" in z.files else (None, None))
        return cls(z["books"], z["mu0"], z["V"], z["s"], z["bits"], z["clip"],
                   *pre)

    def id_state(self) -> list:
        """What a decoder must agree on, for `bitstream_id(extra=...)`."""
        return [self.books, self.mu0, self.V, self.s,
                self.bits.astype(np.int32), self.clip,
                *self._transform_arrays().values()]


def coded_space(x, pre_scale=None, pre_log=None) -> np.ndarray:
    """(N, D) values -> the space an `RVQPCAMeanChannel` with this
    transform fits and codes in: log on the `pre_log` dimensions, then
    times `pre_scale`. Log dimensions must be positive."""
    y = np.array(x, dtype=np.float32)
    if pre_scale is None and pre_log is None:
        return y
    if pre_log is not None:
        lg = np.asarray(pre_log, dtype=bool)
        if (y[:, lg] <= 0).any():
            raise ValueError("log-coded dimensions must be positive")
        y[:, lg] = np.log(y[:, lg])
    if pre_scale is not None:
        y = y * np.asarray(pre_scale, dtype=np.float32)
    return y.astype(np.float32)


CHANNELS = {MeanChannel.kind: MeanChannel,
            RVQPCAMeanChannel.kind: RVQPCAMeanChannel}


def channel_kind(path) -> str:
    """The kind of the mean channel saved at `path`, told apart by the
    tables it holds."""
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
    if {"books", "V"} <= names:
        return RVQPCAMeanChannel.kind
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
