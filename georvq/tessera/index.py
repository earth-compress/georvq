"""The tile index of Tessera v1.1 stores.

The columns every release's index starts with
(georvq.inference.store_index.CORE) are followed by Tessera's own, taken
from each store's header, since the v1.1 run publishes no index of
0.1 degree tiles to join: `year`, `tile_lon` and `tile_lat` (the cell's
centre), `utm_zone`, `zone_row` and `zone_col` (the tile's first pixel
in the zone's grid), `valid_fraction` (the share of the tile's pixels
with an embedding, from its mask) and `embeddings_version`.
"""
from __future__ import annotations

import os


def source_row(product, valid_fraction) -> dict:
    w = product.get("zone_window") or {}
    return {"year": int(product["year"]),
            "tile_lon": float(product["lon"]), "tile_lat": float(product["lat"]),
            "utm_zone": int(product["utm_zone"]),
            "zone_row": w.get("row"), "zone_col": w.get("col"),
            "valid_fraction": float(valid_fraction),
            "embeddings_version": (product.get("embeddings") or {}).get(
                "version")}


def build_index(stores, out_path=None, url_prefix=None, root=None):
    """Index of Tessera v1.1 stores, one row per store in the stores'
    order. `stores` is a list of stores or a directory holding them.
    `path` is the store, `url_prefix` joined to its path relative to
    `root` when given, or that relative path alone when `url_prefix` is
    "". Writes GeoParquet to `out_path` when given; returns the table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from georvq.inference.records import RecordReader
    from georvq.inference.store_index import (find_stores, index_table,
                                              tile_record)
    from georvq.tessera.v1_1 import KIND
    if isinstance(stores, (str, os.PathLike)) and os.path.isdir(stores) \
            and not os.fspath(stores).rstrip("/").endswith(".georvq.zarr"):
        root = stores if root is None else root
        stores = find_stores(stores)
    stores = [os.fspath(f) for f in stores]
    paths, recs, rows, keys = [], [], [], set()
    for f in stores:
        rd = RecordReader(f)
        p = rd.header["product"]
        if p.get("kind") != KIND:
            raise ValueError(f"{f}: product kind {p.get('kind')!r}, not {KIND}")
        key = (int(p["year"]), round(float(p["lon"]), 2),
               round(float(p["lat"]), 2))
        if key in keys:
            raise ValueError(f"{f}: a second store for {key}")
        keys.add(key)
        mask = rd.read_mask(0, int(p["height"]), 0, int(p["width"]))
        recs.append(tile_record(f))
        rows.append(source_row(p, mask.mean()))
        path = os.path.abspath(f)
        if url_prefix is not None:
            prefix = url_prefix.rstrip("/")
            path = (prefix + "/" if prefix else "") + os.path.relpath(
                f, os.fspath(root)).replace(os.sep, "/")
        paths.append(path)
    schema = pa.schema([("year", pa.int16()), ("tile_lon", pa.float64()),
                        ("tile_lat", pa.float64()), ("utm_zone", pa.int16()),
                        ("zone_row", pa.int64()), ("zone_col", pa.int64()),
                        ("valid_fraction", pa.float64()),
                        ("embeddings_version", pa.string())])
    out = index_table(paths, recs, pa.Table.from_pylist(rows, schema=schema))
    if out_path is not None:
        pq.write_table(out, out_path, compression="zstd")
    return out
