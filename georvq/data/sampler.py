"""Sample geometry: how one training sample is cut from the resident pool.

The loop sees only `draw(batch) -> (tensor, meta)`, where meta is a dict
of per-sample tensors (the store row when `return_rows` is set).
Validation does not draw through the sampler: it scores fixed tokens,
and a source with sampled prep makes it deterministic through
`_prep_deterministic`.
"""
from __future__ import annotations


class WindowSampler:
    """One random `window`-px crop per sample, all bands, drawn by
    ChipPool.sample_batch."""

    def __init__(self, pool, model_cfg):
        self.pool = pool
        self.cfg = model_cfg

    def draw(self, batch, align=False, return_rows=False):
        """(tensor [B, ...], meta dict of per-sample tensors)."""
        out = self.pool.sample_batch(batch, self.cfg.window, align=align,
                                     return_rows=return_rows)
        if return_rows:
            crop, rows = out
            return crop, {"row": rows}
        return out, {}
