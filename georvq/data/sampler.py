"""Samplers: sample geometry in one place.

A Sampler sits between the resident pool and the training loop: it
knows how to cut one atomic sample from resident slots and what
per-sample provenance travels with it. The loop sees only
`draw(batch) -> (tensor, meta)`, where meta is a dict of per-sample
tensors (level index, hour, store row, ...) that starts empty and is
filled by samplers that discard identity the tensors alone cannot
carry (the motivating failures are the era5 single-level hacks this
interface retires). Validation does not draw through a sampler: it
scores fixed tokens, and a source with sampled prep makes it
deterministic through `_prep_deterministic`.

Phase-1 status: WindowSampler delegates to ChipPool.sample_batch, so
existing recipes are bit-identical; new geometries subclass Sampler
instead of growing store flags and prep branches.
"""
from __future__ import annotations


class Sampler:
    """One atomic-sample geometry over a pool."""

    #: meta keys every draw fills (checked against the prep's
    #: requirements at config resolution)
    provides: tuple[str, ...] = ()

    def __init__(self, pool, model_cfg):
        self.pool = pool
        self.cfg = model_cfg

    def draw(self, batch, align=False, return_rows=False):
        """(tensor [B, ...], meta dict of per-sample tensors)."""
        raise NotImplementedError


class WindowSampler(Sampler):
    """The classic geometry: one random `window`-px crop per sample, all
    bands. Delegates to ChipPool.sample_batch, so it is the existing
    behaviour under the new interface (guarded by a golden test)."""

    provides = ()

    def draw(self, batch, align=False, return_rows=False):
        out = self.pool.sample_batch(batch, self.cfg.window, align=align,
                                     return_rows=return_rows)
        if return_rows:
            crop, rows = out
            return crop, {"row": rows}
        return out, {}


def build_sampler(name, pool, model_cfg, source=None, params=None):
    """Resolve a sampler by config name; None picks the source's default
    (WindowSampler unless the source overrides `default_sampler`).
    `params` (data.sampler.params) are passed to the constructor as
    keyword arguments, so a sampler that takes none rejects them."""
    params = dict(params or {})
    if name is None:
        if source is not None and hasattr(source, "default_sampler"):
            make = source.default_sampler
        else:
            make = WindowSampler
    else:
        reg = {"window": WindowSampler}
        try:
            from georvq.sources.era5 import MonoLevelSampler
            reg["mono_level"] = MonoLevelSampler
        except ImportError:
            pass
        if name not in reg:
            raise KeyError(f"unknown sampler {name!r}; have {sorted(reg)}")
        make = reg[name]
    try:
        return make(pool, model_cfg, **params)
    except TypeError as e:
        if not params:
            raise
        raise ValueError(
            f"sampler {name or 'default'!r} does not accept "
            f"data.sampler.params {sorted(params)}: {e}") from e
