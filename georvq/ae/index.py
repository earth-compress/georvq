"""Tile index of an AE v2 release: `aef_georvq_index.parquet`.

One row per `.georvq.zarr` store, carrying the row of the source
AlphaEarth index (`aef_index.parquet` on source.coop: fid, crs, path,
year, utm_zone, UTM and WGS84 bounds, geom, geom_bbox, location) for the
tile the store codes, matched on the product id (the source file's stem,
unique across the source index). `path` is rewritten to the coded store
and the source's path moves to `source_path`. The other source columns,
`location` included, describe the source tile and are copied unchanged.
Added columns, as georvq.inference.store_index describes them:
`bitstream_id`, `decoder_id`, `store_bytes`, `metadata_bytes`,
`quality_bytes` and `rungs`.

Only metadata and stored sizes are read, so the stores must be on a
filesystem zarr-python can list, such as a local directory. The output
is GeoParquet with the source's `geo` metadata, its bbox narrowed to the
indexed tiles and a 1.1 bbox covering on `geom_bbox` added.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from georvq import __version__
from georvq.inference.records import SUFFIX
from georvq.inference.store_index import (find_stores, rung_bytes,  # noqa: F401
                                          stem as _stem, tile_record)

INDEX_NAME = "aef_georvq_index.parquet"
_BBOX = ("xmin", "ymin", "xmax", "ymax")

RUNG_TYPE = pa.list_(pa.struct([("rung", pa.string()),
                                 ("bits_px", pa.float64()),
                                 ("bytes", pa.int64())]))


def _geo(source_geo, table, column):
    """The source's geo metadata for the indexed rows."""
    geo = json.loads(source_geo)
    gc = geo["columns"][column]
    if table.num_rows:
        bb = table.column("geom_bbox").combine_chunks()
        gc["bbox"] = [
            float(pc.min(bb.field("xmin")).as_py()),
            float(pc.min(bb.field("ymin")).as_py()),
            float(pc.max(bb.field("xmax")).as_py()),
            float(pc.max(bb.field("ymax")).as_py())]
    gc["covering"] = {"bbox": {k: ["geom_bbox", k] for k in _BBOX}}
    geo["version"] = "1.1.0"
    geo["creator"] = {"library": "georvq", "version": __version__}
    return json.dumps(geo)


def build_index(stores, source_index, out_path=None, root=None, url_prefix=None,
                workers=16, **reader_kw) -> pa.Table:
    """Index of coded AE v2 stores against the source index.

    stores: a local directory, searched for `*.georvq.zarr`, or an
    iterable of local store paths. source_index: path of
    `aef_index.parquet`. `path` is `url_prefix` joined to the store's
    path relative to `root` (the directory when `stores` is one) when
    `url_prefix` is given, and otherwise the store's absolute path. Rows
    follow the source index's order. Raises if a store's product id is
    not in the source index or two stores code the same product. Writes
    GeoParquet to `out_path` when given; returns the table."""
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
            return url_prefix.rstrip("/") + "/" + rel
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
    paths = [rows[i][0] for i in order]
    recs = [rows[i][1] for i in order]
    k = table.schema.get_field_index("path")
    table = table.set_column(k, "path", pa.array(paths, pa.string()))
    table = table.add_column(k + 1, "source_path", pa.array(
        [src_paths[i] for i in order], pa.string()))
    for c, t in (("bitstream_id", pa.string()), ("decoder_id", pa.string()),
                 ("store_bytes", pa.int64()), ("metadata_bytes", pa.int64()),
                 ("quality_bytes", pa.int64())):
        table = table.append_column(c, pa.array([r[c] for r in recs], t))
    table = table.append_column("rungs", pa.array([r["rungs"] for r in recs],
                                                  RUNG_TYPE))
    meta = src.schema_arrow.metadata or {}
    if b"geo" in meta:
        geo = json.loads(meta[b"geo"])
        table = table.replace_schema_metadata(
            {"geo": _geo(meta[b"geo"], table, geo["primary_column"])})
    if out_path is not None:
        pq.write_table(table, out_path, compression="zstd")
    return table
