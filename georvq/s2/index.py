"""The tile index of S2 v1 stores.

The columns every release's index starts with
(georvq.inference.store_index.CORE) are followed by S2's own, taken from
each store's header, since there is no source index to join for SAFE
products: `safe` (the product's directory name), `mgrs_tile` (such as
"33TTG"), `relative_orbit`, `spacecraft`, `sensing_start` (UTC) and
`processing_baseline`.
"""
from __future__ import annotations

import os
import re

_NAME = re.compile(r"^S2[A-D]_MSIL1C_\d{8}T\d{6}_N\d{4}_R(\d{3})_T(\w{5})_")


def source_row(product) -> dict:
    """S2's own index columns of a header's `product`."""
    from datetime import datetime
    m = _NAME.match(str(product["id"]))
    t = product.get("sensing_start")
    return {"safe": product.get("safe"),
            "mgrs_tile": m.group(2) if m else None,
            "relative_orbit": int(m.group(1)) if m else None,
            "spacecraft": product.get("spacecraft"),
            "sensing_start": datetime.fromisoformat(t.replace("Z", "+00:00"))
            if t else None,
            "processing_baseline": product.get("processing_baseline")}


def build_index(stores, out_path=None, url_prefix=None, root=None):
    """Index of S2 v1 stores, one row per store in the stores' order.
    `stores` is a list of stores or a directory holding them. `path` is
    the store, `url_prefix` joined to its path relative to `root` when
    given, or that relative path alone when `url_prefix` is "". Writes
    GeoParquet to `out_path` when given; returns the table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from georvq.inference.records import RecordReader
    from georvq.inference.store_index import (find_stores, index_table,
                                              tile_record)
    from georvq.s2.v1 import KIND
    if isinstance(stores, (str, os.PathLike)) and os.path.isdir(stores) \
            and not os.fspath(stores).rstrip("/").endswith(".georvq.zarr"):
        root = stores if root is None else root
        stores = find_stores(stores)
    stores = [os.fspath(f) for f in stores]
    paths, recs, rows = [], [], []
    for f in stores:
        p = RecordReader(f).header["product"]
        if p.get("kind") != KIND:
            raise ValueError(f"{f}: product kind {p.get('kind')!r}, not {KIND}")
        recs.append(tile_record(f))
        rows.append(source_row(p))
        path = os.path.abspath(f)
        if url_prefix is not None:
            prefix = url_prefix.rstrip("/")
            path = (prefix + "/" if prefix else "") + os.path.relpath(
                f, os.fspath(root)).replace(os.sep, "/")
        paths.append(path)
    schema = pa.schema([("safe", pa.string()), ("mgrs_tile", pa.string()),
                        ("relative_orbit", pa.int32()),
                        ("spacecraft", pa.string()),
                        ("sensing_start", pa.timestamp("ms", tz="UTC")),
                        ("processing_baseline", pa.string())])
    out = index_table(paths, recs, pa.Table.from_pylist(rows, schema=schema))
    if out_path is not None:
        pq.write_table(out, out_path)
    return out
