"""The Sentinel-2 L1C release codec (s2-v1).

This package holds the policy specific to Sentinel-2 L1C products: the
13 bands on their three native grids (10, 20 and 60 m), the per-window
per-band mean and standard deviation the model normalises by and the
release transmits, the rung ladder and its DN RMSE quality statistics,
one georvq Zarr store per SAFE product with the rest of the product kept
as a zip, and the reconstruction of a SAFE product from a store
(`georvq.s2.v1`, with the SAFE reading and writing in
`georvq.s2.safe`). The source-agnostic parts are in georvq.inference.
Stores are opened with `georvq.open`.
"""
