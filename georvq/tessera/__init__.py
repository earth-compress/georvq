"""The Tessera release codec.

Everything here is a response to the specifics of the Tessera embedding
field and to the policies of the Tessera release: the int8 + per-pixel
scale reparametrization, the transmitted patch mean and the rate ladder.
The generic, source-agnostic machinery (Codec/Packet, bundles, bitstream
ids, the store codec and reader) lives in `georvq.inference`.

`georvq.tessera.v1_1` is the Tessera v1.1 release, one georvq Zarr store
per 4096 px shard of the v1.1 embeddings' zone arrays (the global run
produced by dClimate), opened with `georvq.open`; `georvq.tessera.tiles`
reads shards and tiles from the v1.1 Zarr store, and
`georvq.tessera.index` makes the release's index. `read_tile` (from
`index`) reads a geotessera tile from the stores an index lists, and
`prep` (from `io`) turns the int8 codes and scale plane into the model's
input; both are imported on first use, so that reading stores with the
ONNX backend does not import torch.

The per-window mean is transmitted: Tessera vectors are not unit-norm,
and a window's mean carries much of its energy, so it is sent rather
than coded by the pyramid. `georvq.inference.mean_codec` turns the
model's rate accounting into real bits.

Use `georvq.ae` and this package as worked examples of the same
contract, not as templates to inherit blindly.
"""
import importlib

_NAMES = {"prep": "io", "read_tile": "index", "find_tile": "index"}

__all__ = ["prep", "read_tile", "find_tile"]


def __getattr__(name):
    mod = _NAMES.get(name)
    if mod is None:
        raise AttributeError(f"module 'georvq.tessera' has no attribute "
                             f"{name!r}")
    return getattr(importlib.import_module(f"georvq.tessera.{mod}"), name)
