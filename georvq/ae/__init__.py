"""The AlphaEarth release codec (ae-v2).

This package holds the policy specific to the AlphaEarth Foundations
annual embedding tiles (64-band int8, -128 nodata, 8192 px tiles with
bottom-up rows): pixel prep and requantization, the transmitted window
mean, the rung ladder and quality column, one georvq Zarr store per
source tile, bundles and their resolution, and the tile index. The
source-agnostic parts (the coded records and their Zarr arrays, the mean
coder, the ONNX export, bitstream ids) are in georvq.inference.
"""


def open(store, **kw):
    """Open an AE v2 georvq Zarr store (a path or URL, or an open
    zarr.Group) as a rasterio-like dataset; see
    georvq.ae.v2.open_dataset."""
    from georvq.ae.v2 import open_dataset
    return open_dataset(store, **kw)
