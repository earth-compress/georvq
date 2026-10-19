"""The AlphaEarth release codec (ae-v2).

This package holds the policy specific to the AlphaEarth Foundations
annual embedding tiles (64-band int8, -128 nodata, 8192 px tiles with
bottom-up rows): pixel prep and requantization, the transmitted window
mean, the rung ladder and quality column, one georvq Zarr store per
source tile, bundles and their resolution, and the tile index. The
source-agnostic parts (the coded records and their Zarr arrays, the
store codec and reader, the mean coder, bundles, the ONNX export,
bitstream ids) are in georvq.inference. Stores are opened with
`georvq.open`.
"""
