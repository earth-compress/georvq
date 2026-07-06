"""Downstream A/B evaluation of AlphaEarth compression codecs.

Protocol follows AlphaEarth Foundations (arXiv:2507.22291): frozen 64-d
per-pixel embeddings -> lightweight probe (kNN k=1/k=3, linear) -> balanced
accuracy (classification) or R^2 (regression). We run the *identical* probe on
the original published embedding and on its compressed->decompressed
reconstruction; the delta is the downstream cost of compression.

Eval samples are drawn from AEF source tiles the compressor never trained on
(see extract.held_out_tile_pool) to avoid train/test contamination, and the
codec is fed real spatial windows so reconstruction is faithful to deployment.
"""
from . import extract, probes, benchmarks  # noqa: F401
