"""The Tessera release codec.

Everything here is a response to the specifics of the Tessera embedding
field and to the policies of the Tessera release: the int8 + per-pixel
scale reparametrization, the transmitted patch mean and the rate ladder.
The generic, source-agnostic machinery (Codec/Packet, bundles, bitstream
ids, the store codec and reader) lives in `georvq.inference`.

`georvq.tessera.v2` is the Tessera v2 release, one georvq Zarr store per
geotessera tile, opened with `georvq.open`. `prep` (from `io`) turns
geotessera's int8 codes and scale plane into the model's input, and is
imported on first use so that reading v2 stores with the ONNX backend
does not import torch.

The per-window mean is transmitted: Tessera vectors are not unit-norm,
and the mean is 85.5% of a window's energy, cheap to send and worth more
per bit than pyramid depth. `georvq.inference.mean_codec` turns the
model's rate accounting into real bits.

Use `georvq.ae` and this package as worked examples of the same
contract, not as templates to inherit blindly.
"""
import importlib

_NAMES = {"prep": "io"}

__all__ = ["prep"]


def __getattr__(name):
    mod = _NAMES.get(name)
    if mod is None:
        raise AttributeError(f"module 'georvq.tessera' has no attribute "
                             f"{name!r}")
    return getattr(importlib.import_module(f"georvq.tessera.{mod}"), name)
