"""Additive residual refinements: buying back a tier's quantizer gap.

An RVQ tier stops at some finite depth, leaving a residual the decoder
never sees. When a model's unquantized ceiling sits noticeably above its
full-depth reconstruction, that residual is worth money: scalar-quantize
it coarsely, send it alongside the indices, and the decoder adds it to
that tier's tokens before decoding. Nothing about the codebooks or the
index stream changes, so a refinement is always optional — dropping one
yields a valid decode at a lower rate.

Which tier benefits depends on the model, so the tier is a parameter
here rather than a constant. The Tessera release refines the fine tier,
where that model's slack sits: tier 2's residual is still decaying at
level 20, so it is rate-limited rather than saturated.

Two quantizer families, both with a per-window scale so the coder adapts
to outliers instead of clipping them:

`uniform`
    Symmetric mid-rise uniform quantizer, `2**bits` levels at
    (±0.5, ±1.5, ...) x step, with step set from the window's own
    standard deviation. No fitted state, so nothing travels with the
    release.

`lloyd`
    A fixed Lloyd-Max table, fitted once on held-out residuals and
    shipped with the release, applied in a **rotated** basis. After a
    deep RVQ the residual is close to isotropic quantization noise, so a
    random rotation makes every coordinate Gaussian at the same
    variance — which is exactly the regime where a Gaussian Lloyd-Max
    scalar quantizer is near-optimal, and where PCA has nothing left to
    exploit (measured: random and PCA rotations land within 1% of each
    other on a post-RVQ residual).

The rotation is derived from a seed rather than stored, so it costs
nothing in the bitstream; `rotation()` is deterministic across platforms
and devices because it runs in numpy with an explicit bit generator and
a fixed sign convention on the QR factor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RefinementSpec:
    """How one tier's post-RVQ residual is coded.

    tier:         index of the tier this refines
    bits:         bits per latent dimension
    quantizer:    "uniform" (per-window step, no fitted state) or
                  "lloyd" (shipped table, requires rotate_seed)
    sigma:        uniform only — places the outermost reconstruction
                  point at sigma standard deviations. It must be retuned
                  with `bits`: at 2 bits sigma=2.0 is good, but at 1 bit
                  it puts the two points at +-2 std where the optimum is
                  +-0.8, and the refinement adds error instead of
                  removing it. One bit per dimension is the lloyd
                  family's regime.
    rotate_seed:  seed for the fixed random rotation, or None for no
                  transform
    """
    tier: int
    bits: int
    quantizer: str = "uniform"
    sigma: float = 2.0
    rotate_seed: Optional[int] = None

    def __post_init__(self):
        if self.quantizer not in ("uniform", "lloyd"):
            raise ValueError(f"unknown quantizer {self.quantizer!r}")
        if not 1 <= self.bits <= 8:
            raise ValueError("bits must be in 1..8")
        if self.quantizer == "lloyd" and self.rotate_seed is None:
            raise ValueError(
                "the lloyd quantizer assumes rotated (Gaussian) coordinates; "
                "set rotate_seed")

    @property
    def levels(self) -> int:
        return 1 << self.bits

    def cost_bits(self, cells: int, dim: int) -> float:
        """Bits per window, including the per-window scale where one is
        sent (f16). The rotation costs nothing: it is a seed."""
        body = cells * dim * self.bits
        return float(body + (16 if self.quantizer == "uniform" else 0))

    def to_json(self) -> dict:
        return {"tier": self.tier, "bits": self.bits,
                "quantizer": self.quantizer, "sigma": self.sigma,
                "rotate_seed": self.rotate_seed}

    @classmethod
    def from_json(cls, d: dict) -> "RefinementSpec":
        return cls(tier=int(d["tier"]), bits=int(d["bits"]),
                   quantizer=d.get("quantizer", "uniform"),
                   sigma=float(d.get("sigma", 2.0)),
                   rotate_seed=d.get("rotate_seed"))


# --------------------------------------------------------------------------
# the fixed rotation
# --------------------------------------------------------------------------

def rotation(dim: int, seed: int) -> np.ndarray:
    """A deterministic orthogonal (dim, dim) matrix from a seed.

    numpy's PCG64 is specified bit-for-bit, and the sign of each QR
    column is pinned to make R's diagonal positive, which removes the
    only place LAPACK implementations are free to differ. The result is
    reproducible across platforms, devices and BLAS builds — a
    requirement, since a decoder that reconstructs a different rotation
    reconstructs garbage.
    """
    g = np.random.Generator(np.random.PCG64(seed))
    q, r = np.linalg.qr(g.standard_normal((dim, dim)))
    return np.asarray(q * np.sign(np.diag(r)), dtype=np.float32)


# --------------------------------------------------------------------------
# fitting (lloyd only) — done once, at release build time
# --------------------------------------------------------------------------

def fit_lloyd(res: np.ndarray, spec: RefinementSpec, iters: int = 30,
              sample: int = 200_000, seed: int = 0) -> np.ndarray:
    """Fit per-dimension Lloyd-Max levels on held-out residuals.

    res: (N, cells, dim) or (N, dim) residuals from the tier this spec
    refines. Returns the (dim, levels) table to ship with the release.

    Fit on residuals the shipped codebooks actually produce — a table
    fitted against a different checkpoint's residuals is fitted to the
    wrong distribution.
    """
    x = _flat(res)
    x = x @ rotation(x.shape[1], spec.rotate_seed)
    if x.shape[0] > sample:
        g = np.random.Generator(np.random.PCG64(seed))
        x = x[g.choice(x.shape[0], sample, replace=False)]
    xt = np.ascontiguousarray(x.T)                       # (dim, N)
    L = spec.levels
    qs = np.linspace(0.5 / L, 1 - 0.5 / L, L)
    q = np.quantile(xt, qs, axis=1).T.copy()             # (dim, L)
    for _ in range(iters):
        bnd = (q[:, 1:] + q[:, :-1]) / 2
        a = _bucket(xt, bnd)
        s = np.zeros_like(q)
        c = np.zeros_like(q)
        for d in range(xt.shape[0]):
            s[d] = np.bincount(a[d], weights=xt[d], minlength=L)
            c[d] = np.bincount(a[d], minlength=L)
        q = np.where(c > 0, s / np.maximum(c, 1.0), q)
    return q.astype(np.float32)


def _bucket(xt: np.ndarray, bnd: np.ndarray) -> np.ndarray:
    """Per-row searchsorted: (dim, N) values against (dim, L-1) bounds."""
    out = np.empty(xt.shape, dtype=np.int64)
    for d in range(xt.shape[0]):
        out[d] = np.searchsorted(bnd[d], xt[d])
    return out


# --------------------------------------------------------------------------
# pack / unpack
# --------------------------------------------------------------------------

def pack(res: np.ndarray, spec: RefinementSpec,
         table: Optional[np.ndarray] = None):
    """(N, cells, dim) float residuals -> (codes uint8 packed, step).

    `step` is an (N,) float16 array for the uniform quantizer and None
    for lloyd, whose scale lives in the shipped table.
    """
    r = _flat2(res)                                       # (N, cells*dim)
    if spec.quantizer == "uniform":
        half = spec.levels / 2 - 0.5
        step = (spec.sigma * r.std(axis=1, keepdims=True) / half) \
            .clip(min=1e-12)
        c = np.clip(np.round(r / step - 0.5) + 0.5, -half, half) + half
        return _packbits(c, spec.bits), step[:, 0].astype(np.float16)
    n, dim = res.shape[0], res.shape[-1]
    x = _flat(res) @ rotation(dim, spec.rotate_seed)
    bnd = (table[:, 1:] + table[:, :-1]) / 2
    c = _bucket(np.ascontiguousarray(x.T), bnd).T.astype(np.float32)
    return _packbits(c.reshape(n, -1), spec.bits), None


def unpack(codes: np.ndarray, step, spec: RefinementSpec, cells: int,
           dim: int, table: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse of `pack` -> (N, cells, dim) float32."""
    c = _unpackbits(codes, spec.bits, cells * dim)
    if spec.quantizer == "uniform":
        half = spec.levels / 2 - 0.5
        r = (c - half) * np.asarray(step, dtype=np.float32)[:, None]
        return r.reshape(-1, cells, dim)
    n = c.shape[0]
    idx = c.reshape(n * cells, dim).astype(np.int64)
    y = table[np.arange(dim)[None, :], idx]        # per-dimension levels
    r = y.astype(np.float32) @ rotation(dim, spec.rotate_seed).T
    return r.reshape(n, cells, dim)


def _flat(res: np.ndarray) -> np.ndarray:
    return np.asarray(res, dtype=np.float32).reshape(-1, res.shape[-1])


def _flat2(res: np.ndarray) -> np.ndarray:
    return np.asarray(res, dtype=np.float32).reshape(res.shape[0], -1)


def _packbits(c: np.ndarray, bits: int) -> np.ndarray:
    """(N, M) integer codes in [0, 2**bits) -> packed uint8 rows."""
    planes = (c.astype(np.uint8)[:, :, None]
              >> np.arange(bits - 1, -1, -1)) & 1
    return np.packbits(planes.reshape(c.shape[0], -1), axis=1)


def _unpackbits(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    b = np.unpackbits(packed, axis=1, count=count * bits) \
        .reshape(packed.shape[0], count, bits)
    c = np.zeros(b.shape[:2], dtype=np.float32)
    for i in range(bits):
        c = c * 2 + b[:, :, i]
    return c
