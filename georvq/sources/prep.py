"""Preps: static, fitted model-space transforms.

A Prep is a pure function pair over (tensor, meta): `forward` takes the
sampler's raw batch to model space, `inverse` takes model-space values
back to physical units. It holds no per-sample parameters and
transmits nothing; fitted state (moments, Box-Cox exponents, quantile
edges) is produced by an explicit fit step, versioned, frozen at step
0, and embedded in checkpoints. Anything measured per sample and sent
to the decoder is MODEL machinery (stats_cond), not prep.

Metrics and figure panels declare the space they are defined in
(`Metric.space`); the harness inverts through the run's prep exactly
once for the physical ones. A prep that declares `invertible = False`
withholds physical metrics at report resolution rather than at figure
time.

Phase-1 status: `SourcePrep` delegates to the Source's existing
`prep`/`normalizer` pair, so current recipes are unchanged; named
preps land per source as they migrate (era5 first).
"""
from __future__ import annotations


class Prep:
    """Static model-space transform. Subclasses implement forward and,
    unless `invertible` is False, inverse."""

    #: meta keys forward/inverse need (checked against the sampler's
    #: `provides` at config resolution)
    requires: tuple[str, ...] = ()
    invertible: bool = True

    def __init__(self, source, model_cfg):
        self.source = source
        self.cfg = model_cfg

    def forward(self, raw, meta, device):
        """raw store batch -> (x, valid, meta) in model space."""
        raise NotImplementedError

    def __call__(self, raw, meta, device):
        return self.forward(raw, meta, device)

    def inverse(self, y, meta):
        """Model-space values -> physical units. Exact, or exact up to
        a documented clamp; raise if `invertible` is False."""
        raise NotImplementedError

    def state(self):
        """Fitted constants for checkpoint embedding (json-able)."""
        return {}


class SourcePrep(Prep):
    """Delegates to the Source's existing prep/normalizer pair -- the
    compatibility default every current recipe resolves to."""

    def forward(self, raw, meta, device):
        x, valid = self.source.prep(raw, device)
        return x, valid, meta

    @property
    def invertible(self):
        return self.source.normalizer(self.cfg) is not None

    def inverse(self, y, meta):
        n = self.source.normalizer(self.cfg)
        if n is None:
            raise RuntimeError(
                f"source {self.source.name!r} declares no normalizer; "
                f"its model space has no physical inverse")
        return n.denormalize(y, None)


def build_prep(name, source, model_cfg, params=None):
    """Resolve a prep by config name; None picks the source's default
    (SourcePrep unless the source overrides `default_prep`). `params`
    (data.prep.params, a named fitted-state artifact) is passed as the
    `params` keyword, so a prep that has no fitted state rejects it."""
    if name is None:
        if hasattr(source, "default_prep"):
            make = source.default_prep
            args = (model_cfg,)
        else:
            make = SourcePrep
            args = (source, model_cfg)
    else:
        named = getattr(source, "named_preps", lambda: {})()
        if name not in named:
            raise KeyError(
                f"source {source.name!r} has no prep {name!r}; "
                f"have {sorted(named) or ['(default only)']}")
        make = named[name]
        args = (source, model_cfg)
    if params is None:
        return make(*args)
    try:
        return make(*args, params=params)
    except TypeError as e:
        raise ValueError(
            f"prep {name or 'default'!r} does not accept data.prep.params "
            f"({params!r}): {e}") from e
