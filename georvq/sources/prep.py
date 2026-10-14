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
once for the physical ones, calling `inverse(y, {})`. `invertible`
says whether that call works with an empty meta; a prep that declares
it False withholds physical metrics at report resolution rather than
at figure time.

`SourcePrep` delegates to the Source's existing `prep`/`normalizer`
pair. ERA5 `boxcox` is the only named prep so far.
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

    def load_state(self, state):
        """Restore constants a checkpoint embedded through `state()`.
        Preps whose constants are fixed in the source ignore it."""


class SourcePrep(Prep):
    """Delegates to the Source's existing prep/normalizer pair -- the
    compatibility default every current recipe resolves to.

    A normalizer with per-sample stats (DEM, S2: `per_sample = True`)
    can only be inverted with the stats of the batch being inverted.
    `forward` returns them as `meta["stats"]` and `inverse` reads them
    from there, so `invertible` is False for those sources: the
    harness inverts with an empty meta."""

    def _norm(self):
        return self.source.normalizer(self.cfg)

    def forward(self, raw, meta, device):
        x, valid = self.source.prep(raw, device)
        n = self._norm()
        if n is not None and getattr(n, "per_sample", False):
            meta = {**meta, "stats": self.source.last_stats()}
        return x, valid, meta

    @property
    def invertible(self):
        n = self._norm()
        return n is not None and not getattr(n, "per_sample", False)

    def inverse(self, y, meta):
        n = self._norm()
        if n is None:
            raise RuntimeError(
                f"source {self.source.name!r} declares no normalizer; "
                f"its model space has no physical inverse")
        if not getattr(n, "per_sample", False):
            return n.denormalize(y, None)
        stats = (meta or {}).get("stats")
        if stats is None:
            raise ValueError(
                f"source {self.source.name!r} normalizes with per-sample "
                f"stats; pass the meta that forward() returned for this "
                f"batch, which carries them as meta['stats']")
        return n.denormalize(y, stats)

    def state(self):
        st = getattr(self.source, "prep_state", None)
        return {"source": st()} if st is not None and st() else {}

    def load_state(self, state):
        if state and "source" in state and \
                hasattr(self.source, "load_prep_state"):
            self.source.load_prep_state(state["source"])


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


def prep_of(model):
    """The prep a model was trained with: `model.prep` when
    `factory.from_checkpoint` attached it, else rebuilt by name from the
    model's config (the source's default prep when the config names
    none)."""
    p = getattr(model, "prep", None)
    if p is not None:
        return p
    cfg = getattr(model, "full_config", None)
    name = cfg.data.prep.name if cfg is not None else None
    params = cfg.data.prep.params if cfg is not None else None
    return build_prep(name, model.source, model.cfg, params=params)
