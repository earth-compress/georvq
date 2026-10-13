"""The AlphaEarth release codec.

Everything here is a response to the specifics of the AlphaEarth
Foundations embedding tiles (64-band unit-norm int8, -128 nodata,
1024 px-blocked COG sources) and the policies of the frozen AE release:
pixel prep, the 2-bit coarse-residual sidecar, the quality ladder, the
COG asset container, and the executables. The generic, source-agnostic
machinery (Codec/Packet, ONNX export, release bundles, bitstream ids)
lives in georvq.inference; new sources should build their own release
layer on top of that, using this package as the worked example — not as
a template to inherit blindly.
"""


def open(path_or_url, **kw):
    """Open an AE v2 .gvq file as a rasterio-like dataset; see
    georvq.ae.v2.open_dataset."""
    from georvq.ae.v2 import open_dataset
    return open_dataset(path_or_url, **kw)
