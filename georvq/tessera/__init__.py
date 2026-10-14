"""The Tessera release codec.

Everything here is a response to the specifics of the Tessera embedding
field and to the policies of the Tessera release: the int8 + per-pixel
scale reparametrization, the transmitted patch mean, the rate ladder, and
the choice of which tier carries a residual refinement. The generic,
source-agnostic machinery (Codec/Packet, refinements, release bundles,
bitstream ids) lives in `georvq.inference`.

The per-window mean is transmitted: Tessera vectors are not unit-norm,
and the mean is 85.5% of a window's energy, cheap to send and worth more
per bit than pyramid depth. `georvq.inference.mean_codec` turns the
model's rate accounting into real bits. The optional refinement sits on
the fine tier; see `REFINEMENT` in `release.py` for the measurement
behind that choice.

Use `georvq.ae` and this package as two worked examples of the same
contract, not as templates to inherit blindly.
"""
from georvq.tessera.io import (decode_tile, encode_tile, from_windows, prep,
                               requantize, to_windows)
from georvq.tessera.release import (LADDER, REFINEMENT, TesseraRelease,
                                    build_bundle, load)

__all__ = ["LADDER", "REFINEMENT", "TesseraRelease", "build_bundle", "load",
           "prep", "encode_tile", "decode_tile", "to_windows", "from_windows",
           "requantize"]
