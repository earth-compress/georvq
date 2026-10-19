"""The Tessera release codecs.

Everything here is a response to the specifics of the Tessera embedding
field and to the policies of the Tessera releases: the int8 + per-pixel
scale reparametrization, the transmitted patch mean, the rate ladder, and
the choice of which tier carries a residual refinement. The generic,
source-agnostic machinery (Codec/Packet, refinements, release bundles,
bitstream ids, the store codec and reader) lives in `georvq.inference`.

`georvq.tessera.v2` is the Tessera v2 release, one georvq Zarr store per
geotessera tile, opened with `georvq.open`. The names exported here are
those of the beta release (packets in memory, `release.py`), and are
imported on first use so that reading v2 stores with the ONNX backend
does not import torch.

The per-window mean is transmitted: Tessera vectors are not unit-norm,
and the mean is 85.5% of a window's energy, cheap to send and worth more
per bit than pyramid depth. `georvq.inference.mean_codec` turns the
model's rate accounting into real bits. The beta's optional refinement
sits on the fine tier; see `REFINEMENT` in `release.py` for the
measurement behind that choice.

Use `georvq.ae` and this package as worked examples of the same
contract, not as templates to inherit blindly.
"""
import importlib

_NAMES = {"decode_tile": "io", "encode_tile": "io", "from_windows": "io",
          "prep": "io", "requantize": "io", "to_windows": "io",
          "LADDER": "release", "REFINEMENT": "release",
          "TesseraRelease": "release", "build_bundle": "release",
          "load": "release"}

__all__ = ["LADDER", "REFINEMENT", "TesseraRelease", "build_bundle", "load",
           "prep", "encode_tile", "decode_tile", "to_windows", "from_windows",
           "requantize"]


def __getattr__(name):
    mod = _NAMES.get(name)
    if mod is None:
        raise AttributeError(f"module 'georvq.tessera' has no attribute "
                             f"{name!r}")
    return getattr(importlib.import_module(f"georvq.tessera.{mod}"), name)
