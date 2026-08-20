"""Data-source adapters: everything dataset-specific in one pluggable place.

A `Source` owns the pieces of the pipeline that depend on what the data IS
(bands, dtypes, normalization, loss), while the pyramid/VQ/rate/decoder core
stays shared. Adding a new dataset = a store builder (see georvq.data.build),
a Source subclass here, and a YAML preset.
"""
from __future__ import annotations

import torch.nn as nn


class Source:
    """Interface for a data source. Subclass and register in SOURCES."""

    name: str = "base"

    def __init__(self, model_cfg):
        self.cfg = model_cfg

    def prep(self, raw, device):
        """Raw store batch -> (x, valid): model-space floats (B, C, W, W)
        and a (B, W, W) bool validity mask."""
        raise NotImplementedError

    def build_stem(self, model_cfg) -> nn.Module:
        """Raw input plane(s) -> the (B, C, W, W) plane the tokenizer eats."""
        return nn.Identity()

    def build_head(self, model_cfg, in_dim) -> nn.Module:
        """Decoder fine tokens (B, cells, in_dim) -> pixel output(s)."""
        raise NotImplementedError

    def normalizer(self, model_cfg):
        """Optional per-tile stats side-channel codec (None if the data
        needs no per-tile normalization)."""
        return None

    def recon_stats(self, out, x, valid):
        """Reconstruction loss term and quality metrics.
        Returns (loss, metrics: dict[str, float tensor])."""
        raise NotImplementedError

    def quality(self, out, x, valid):
        """Scalar validation quality (higher is better, perfect = 1); drives
        the R-D front and best-checkpoint selection. Default: per-pixel
        channel cosine, correct for unit-norm fields only — sources with
        other channel geometry must override (see dem.py)."""
        from georvq.metrics import masked_recon_stats
        return masked_recon_stats(out, x, valid)[1]


def get_source(name: str, model_cfg) -> Source:
    """Look up a `Source` adapter by its config name.

    S2 is imported lazily and skipped if its optional dependencies are
    absent, so a missing extra narrows the available sources rather than
    breaking the import of every other one.
    """
    from georvq.sources.ae import AESource
<<<<<<< Updated upstream
    from georvq.sources.tessera import TesseraSource
    SOURCES = {"ae": AESource, "tessera": TesseraSource}
=======
    from georvq.sources.dem import DEMSource
    SOURCES = {"ae": AESource, "dem": DEMSource}
>>>>>>> Stashed changes
    try:
        from georvq.sources.s2 import S2Source
        SOURCES["s2"] = S2Source
    except ImportError:
        pass
    if name not in SOURCES:
        raise KeyError(f"unknown data source {name!r}; have {sorted(SOURCES)}")
    return SOURCES[name](model_cfg)
