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

    def build_head(self, model_cfg, in_dim, n_tiers=None,
                   out_px=None) -> nn.Module:
        """Decoder fine tokens (B, cells, in_dim) -> pixel output(s).

        `n_tiers` and `out_px` are set when the head belongs to a member
        of a decoder bank, which decodes a PREFIX of the tiers to a
        smaller output than the window. Both None means the whole pyramid
        at full resolution, which is what every single-decoder model does.
        """
        raise NotImplementedError

    @staticmethod
    def head_geom(model_cfg, n_tiers=None, out_px=None):
        """(grid, patch edge) for a head over the first `n_tiers` tiers.

        The finest tier a head sees owns the token grid, and each token
        expands to a patch of `out_px // grid` pixels. For the full
        decoder this is the window and the last tier, unchanged.
        """
        fine = model_cfg.tiers[(n_tiers or len(model_cfg.tiers)) - 1]
        px = out_px or model_cfg.window
        return fine.grid, px // fine.grid

    def normalizer(self, model_cfg):
        """Optional per-tile stats side-channel codec (None if the data
        needs no per-tile normalization)."""
        return None

    def side_bits(self, cfg) -> float:
        """Per-window bits this source must ship OUTSIDE the index stream.

        A `normalizer` removes per-tile statistics before the encoder, and
        the decode is meaningless without them, so they are part of the
        rate whether or not anyone counts them. Declare the cost here and
        `PyramidCodec.side_bits` folds it into every rate figure. Sources
        that normalize nothing return 0.
        """
        return 0.0

    def recon_stats(self, out, x, valid):
        """Reconstruction loss term and quality metrics.
        Returns (loss, metrics: dict[str, float tensor])."""
        raise NotImplementedError

    def report(self, model_cfg):
        """What a run on this source should measure, plot and log.

        Returns a `georvq.sources.report.ReportSpec`. There is deliberately
        NO default: reporting inherited from whichever source came before
        is how a single-band elevation model ended up logging cosine, so a
        new source states its own surface or gets a loud error. The two
        ready-made shapes cover most cases -- `embedding_field_spec()` for
        dense per-pixel vector fields, `scalar_field_spec()` for
        single-plane scalar fields.
        """
        raise NotImplementedError(
            f"source {self.name!r} declares no ReportSpec -- see "
            f"georvq/sources/report.py; embedding_field_spec() and "
            f"scalar_field_spec() are the ready-made shapes.")


def get_source(name: str, model_cfg) -> Source:
    """Look up a `Source` adapter by its config name."""
    from georvq.sources.ae import AESource
    from georvq.sources.tessera import TesseraSource
    from georvq.sources.dem import DEMSource
    from georvq.sources.s2 import S2Source
    # "era5:<var>" carries the variable in the name so the adapter can
    # pick that variable's standardization moments
    if name.startswith("era5:"):
        from georvq.sources.era5 import Era5Source
        return Era5Source(model_cfg, var=name.split(":", 1)[1])
    SOURCES = {"ae": AESource, "tessera": TesseraSource,
               "dem": DEMSource, "s2": S2Source}
    if name not in SOURCES:
        raise KeyError(f"unknown data source {name!r}; have {sorted(SOURCES)}")
    return SOURCES[name](model_cfg)
