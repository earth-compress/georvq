"""AE v2 release bundle: the files a decoder of v2 stores needs.

The bundle format, its int8 codebooks and its checks are those of
georvq.inference.bundle, for the release `georvq.ae.v2.AE_V2`; the mean
channel file holds `RVQPCAMeanChannel` tables. Loading a bundle gives an
`AEV2Codec`.
"""
from __future__ import annotations

from georvq.ae.v2 import AE_V2, FORMAT, AEV2Codec  # noqa: F401
from georvq.inference import bundle as _bundle
from georvq.inference.bundle import (BOOK_SUFFIX, BUNDLE_VERSION,  # noqa: F401
                                     FILES, NEG_ZERO, SCALE_SEARCH_ULPS,
                                     TRAINING_STATE, book_from_int,
                                     book_to_int, shippable_config)


def write_bundle(out_dir, model, cfg, channel, source=None, notes="",
                 onnx=False) -> dict:
    """Write a v2 bundle for a loaded model and its mean channel. Returns
    the bundle.json payload. `source` is descriptive metadata about the
    checkpoint the weights came from. `onnx` also writes the model's
    ONNX export under onnx/, so that one bundle serves both backends."""
    return _bundle.write_bundle(out_dir, model, cfg, channel, AE_V2,
                                source=source, notes=notes, onnx=onnx)


def build_bundle(ckpt, mean_npz, out_dir, notes="", onnx=False) -> dict:
    """Bundle a training checkpoint and its mean channel tables; `onnx`
    as for write_bundle."""
    return _bundle.build_bundle(ckpt, mean_npz, out_dir, AE_V2, notes=notes,
                                onnx=onnx)


def write_decoder_bundle(bundle_dir, out_dir=None, verify=True) -> dict:
    """The decoder-only bundle of the full bundle at `bundle_dir`, in
    `out_dir` (default `bundle_dir/decoder`); build_bundle writes it
    (georvq.inference.bundle.write_decoder_bundle)."""
    return _bundle.write_decoder_bundle(bundle_dir, out_dir, AE_V2,
                                        verify=verify)


def load_model(bundle_dir, device="cpu", verify=True):
    """(model, bundle.json payload) of a bundle, the model in eval mode.
    `verify` checks the file hashes recorded in bundle.json."""
    return _bundle.load_model(bundle_dir, AE_V2, device, verify)


def load_bundle(bundle_dir, device="cpu", bf16=None, verify=True) -> AEV2Codec:
    """An AEV2Codec from a bundle directory. Raises if the ids recomputed
    from the loaded tensors differ from those in bundle.json."""
    return _bundle.load_bundle(bundle_dir, AE_V2, device, bf16, verify)
