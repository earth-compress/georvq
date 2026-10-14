"""The AlphaEarth release codec (ae-v2).

This package holds the policy specific to the AlphaEarth Foundations
annual embedding tiles (64-band int8, -128 nodata, 8192 px tiles with
bottom-up rows): pixel prep and requantization, the transmitted window
mean, the rung ladder and quality column, one georvq GeoParquet file per
source tile, bundles and their resolution, and the tile index. The
source-agnostic parts (the GeoParquet container, the mean coder, the
ONNX export, bitstream ids) are in georvq.inference.
"""


def open(path_or_url, **kw):
    """Open an AE v2 georvq GeoParquet file as a rasterio-like dataset; see
    georvq.ae.v2.open_dataset."""
    from georvq.ae.v2 import open_dataset
    return open_dataset(path_or_url, **kw)
