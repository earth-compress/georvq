"""Tile index of an AE v2 release: `aef_georvq_index.parquet`.

The index has the columns every release's index has (georvq.inference.
store_index: path, product id, source URL, CRS, bounds, footprint, ids,
sizes and the rung byte table), one row per `.georvq.zarr` store,
followed by the row of the source AlphaEarth index (`aef_index.parquet`
on source.coop: fid, crs, path, year, utm_zone, UTM and WGS84 bounds,
geom, geom_bbox, location) for the tile the store codes, matched on the
product id (the source file's stem, unique across the source index).
The source's `path` becomes `source_path`; its `crs`, which must equal
the store's, and its `geom` and `geom_bbox`, which the index's own
`geometry` and `bbox` replace, are left out. The other source columns,
`location` included, describe the source tile and are copied unchanged.
Rows follow the source index's order.

Only metadata and stored sizes are read, so the stores must be on a
filesystem zarr-python can list, such as a local directory.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

from georvq.inference.records import SUFFIX
from georvq.inference.store_index import (RUNG_TYPE, find_stores,  # noqa: F401
                                          index_table, rung_bytes,
                                          stem as _stem, tile_record)

INDEX_NAME = "aef_georvq_index.parquet"
# source columns the index's own columns replace
_REPLACED = ("crs", "geom", "geom_bbox")


def build_index(stores, source_index, out_path=None, root=None, url_prefix=None,
                workers=16, **reader_kw) -> pa.Table:
    """Index of coded AE v2 stores against the source index.

    stores: a local directory, searched for `*.georvq.zarr`, or an
    iterable of local store paths. source_index: path of
    `aef_index.parquet`. `path` is `url_prefix` joined to the store's
    path relative to `root` (the directory when `stores` is one) when
    `url_prefix` is given (that relative path alone when `url_prefix` is
    ""), and otherwise the store's absolute path. Rows
    follow the source index's order. Raises if a store's product id is
    not in the source index, two stores code the same product, or a
    store's CRS differs from the source's. Writes GeoParquet to
    `out_path` when given; returns the table."""
    if isinstance(stores, (str, os.PathLike)) and os.path.isdir(stores) \
            and not os.fspath(stores).rstrip("/").endswith(SUFFIX):
        root = stores if root is None else root
        stores = find_stores(stores)
    stores = [os.fspath(f) for f in stores]
    if url_prefix is not None and root is None:
        raise ValueError("url_prefix needs root")

    def name(f):
        if url_prefix is not None:
            rel = os.path.relpath(f, os.fspath(root)).replace(os.sep, "/")
            prefix = url_prefix.rstrip("/")
            return (prefix + "/" if prefix else "") + rel
        return os.path.abspath(f)

    with ThreadPoolExecutor(max(1, int(workers))) as ex:
        recs = list(ex.map(lambda f: tile_record(f, **reader_kw), stores))

    src = pq.ParquetFile(source_index)
    src_paths = src.read(columns=["path"]).column("path").to_pylist()
    row_of = {_stem(p): i for i, p in enumerate(src_paths)}
    if len(row_of) != len(src_paths):
        raise ValueError(f"{source_index}: product ids are not unique")
    rows, seen = {}, {}
    for f, r in zip(stores, recs):
        pid = r["product_id"]
        if pid not in row_of:
            raise ValueError(f"{f}: product {pid!r} is not in {source_index}")
        if pid in seen:
            raise ValueError(f"{f} and {seen[pid]} both code {pid!r}")
        seen[pid] = f
        rows[row_of[pid]] = (name(f), r)
    order = sorted(rows)

    table = src.read().take(pa.array(order, pa.int64()))
    table = table.replace_schema_metadata(None)
    recs = [rows[i][1] for i in order]
    if "crs" in table.schema.names:
        for r, c in zip(recs, table.column("crs").to_pylist()):
            if r["crs"] != c:
                raise ValueError(f"{r['product_id']}: store CRS {r['crs']}, "
                                 f"source index {c}")
    table = table.drop([c for c in _REPLACED if c in table.schema.names])
    table = table.rename_columns(["source_path" if c == "path" else c
                                  for c in table.schema.names])
    out = index_table([rows[i][0] for i in order], recs, table)
    if out_path is not None:
        pq.write_table(out, out_path, compression="zstd")
    return out
