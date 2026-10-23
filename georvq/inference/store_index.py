"""The tile index of a release: one row per store, the same columns for
every release, followed by the source collection's own columns.

A release's index is GeoParquet 1.1 with one row per store. Its first
columns (`CORE`) are the same for every release and are computed here
from each store's metadata and stored sizes; the release layer then
appends the row of the source collection's own index for the product
the store codes (georvq.ae.index matches source.coop's
`aef_index.parquet`, georvq.tessera.v2 geotessera's manifest). The
core columns are:

- `path`: the store, as a URL or a path relative to the index;
- `product_id`: the header's `product.id`;
- `source_url`: the header's `product.source`, where the source product
  was read from;
- `release`: the header's `codec.release.name`;
- `crs`: "EPSG:<code>" of the coded grid, as the store's `proj:code`;
- `width`, `height`: the product's size in pixels;
- `bounds`: {xmin, ymin, xmax, ymax} of the product in its CRS;
- `bbox`: the same in longitude and latitude (WGS 84), the GeoParquet
  covering of `geometry`;
- `geometry`: the product's footprint in longitude and latitude as a
  WKB polygon, each edge densified to 21 points;
- `bitstream_id`, `decoder_id`: from the store's codec header;
- `store_bytes`: the bytes stored by the store's arrays, each array's
  zarr.json included (zarr-python's `nbytes_stored`);
- `metadata_bytes`: the group zarr.json, which a reader fetches on open
  and which holds the consolidated metadata of the arrays;
- `quality_bytes`: the stored bytes of the store's quality arrays, 0
  when it has none;
- `rungs`: per rung of the header's rung table, in order, `rung`
  ("8-8-2"), the header's nominal `bits_px`, and `bytes`, the bytes a
  whole-product read at that rung reads: the stored (compressed) chunks
  of base and of each tier's levels up to the rung's depth, the shard
  indexes of the tiers read, and the mask array's stored bytes; quality
  excluded.

Only metadata and stored sizes are read, so the stores must be on a
filesystem zarr-python can list, such as a local directory.
"""
from __future__ import annotations

import json
import os
import struct

import pyarrow as pa

from georvq.inference.records import SUFFIX, RecordReader

_BOX = pa.struct([(k, pa.float64()) for k in ("xmin", "ymin", "xmax", "ymax")])
RUNG_TYPE = pa.list_(pa.struct([("rung", pa.string()),
                                ("bits_px", pa.float64()),
                                ("bytes", pa.int64())]))
CORE = pa.schema([
    ("path", pa.string()), ("product_id", pa.string()),
    ("source_url", pa.string()), ("release", pa.string()),
    ("crs", pa.string()), ("width", pa.int64()), ("height", pa.int64()),
    ("bounds", _BOX), ("bbox", _BOX), ("geometry", pa.binary()),
    ("bitstream_id", pa.string()), ("decoder_id", pa.string()),
    ("store_bytes", pa.int64()), ("metadata_bytes", pa.int64()),
    ("quality_bytes", pa.int64()), ("rungs", RUNG_TYPE)])
EDGE_POINTS = 21


def rung_bytes(reader: RecordReader, mask_bytes: int) -> list:
    """Per header rung: {rung, bits_px, bytes}, with bytes the stored
    bytes a whole-product read at the rung reads."""
    col = reader.column_bytes()
    index = reader.shard_index_bytes()
    out = []
    for r in reader.header["rungs"]:
        t = [int(v) for v in r["tuple"]]
        cols = reader.columns_for(t, mask=False)
        idx = sum(index[tier.name] for tier, L in zip(reader.layout.tiers, t)
                  if not tier.in_base and L > 0)
        out.append({"rung": "-".join(map(str, t)),
                    "bits_px": float(r["bits_px"]),
                    "bytes": int(sum(col.get(c, 0) for c in cols) + idx
                                 + mask_bytes)})
    return out


def footprint(product) -> dict:
    """`crs`, `bounds`, `bbox` and `geometry` of a header's product (see
    the module docstring); None for each when it has no CRS."""
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp
    a, b, c, d, e, f = (float(v) for v in product["transform"][:6])
    w, h = int(product["width"]), int(product["height"])
    n = EDGE_POINTS - 1
    ring = ([(w * i / n, 0) for i in range(n)] + [(w, h * i / n) for i in range(n)]
            + [(w * (n - i) / n, h) for i in range(n)]
            + [(0, h * (n - i) / n) for i in range(n)])
    xs = [a * px + b * py + c for px, py in ring]
    ys = [d * px + e * py + f for px, py in ring]
    bounds = {"xmin": min(xs), "ymin": min(ys), "xmax": max(xs), "ymax": max(ys)}
    epsg, wkt = product.get("epsg"), product.get("crs_wkt")
    if not epsg and not wkt:
        return {"crs": None, "bounds": bounds, "bbox": None, "geometry": None}
    crs = CRS.from_epsg(int(epsg)) if epsg else CRS.from_wkt(wkt)
    lon, lat = warp(crs, CRS.from_epsg(4326), xs, ys)
    if (max(lon) - min(lon)) > 180:
        raise ValueError(f"product {product.get('id')}: its footprint "
                         "crosses the antimeridian")
    pts = list(zip(lon, lat)) + [(lon[0], lat[0])]
    wkb = struct.pack("<bII", 1, 3, 1) + struct.pack("<I", len(pts)) + b"".join(
        struct.pack("<dd", x, y) for x, y in pts)
    return {"crs": f"EPSG:{int(epsg)}" if epsg else None, "bounds": bounds,
            "bbox": {"xmin": min(lon), "ymin": min(lat),
                     "xmax": max(lon), "ymax": max(lat)},
            "geometry": wkb}


def tile_record(path, **reader_kw) -> dict:
    """Metadata facts of one coded store: the core columns but `path`.
    `reader_kw` go to RecordReader."""
    rd = RecordReader(path, **reader_kw)
    h = rd.header
    codec = h.get("codec") or {}
    rel = codec.get("release")
    sizes = rd.stored_bytes()
    meta = os.path.join(os.fspath(path), "zarr.json")
    p = h["product"]
    return {"product_id": str(p["id"]),
            "source_url": p.get("source"),
            "release": rel.get("name") if isinstance(rel, dict) else None,
            "width": int(p["width"]), "height": int(p["height"]),
            **footprint(p),
            "bitstream_id": codec.get("bitstream_id"),
            "decoder_id": codec.get("decoder_id"),
            "store_bytes": int(sum(sizes.values())),
            "metadata_bytes": int(os.path.getsize(meta)),
            "quality_bytes": int(sum(b for k, b in sizes.items()
                                     if k.startswith("quality/"))),
            "rungs": rung_bytes(rd, sizes.get("mask", 0))}


def index_table(paths, records, source=None) -> pa.Table:
    """The index of stores at `paths` with their `tile_record`s: the core
    columns, then the columns of `source` (a table with one row per
    store, in the same order), whose names must not be core names."""
    rows = [dict(r, path=p) for p, r in zip(paths, records)]
    t = pa.Table.from_pylist(rows, schema=CORE)
    if source is not None:
        if source.num_rows != t.num_rows:
            raise ValueError(f"{source.num_rows} source rows for "
                             f"{t.num_rows} stores")
        clash = sorted(set(source.schema.names) & set(CORE.names))
        if clash:
            raise ValueError(f"source columns {clash} are core column names")
        for name in source.schema.names:
            t = t.append_column(source.schema.field(name), source.column(name))
    return t.replace_schema_metadata({"geo": json.dumps(geo_metadata(t))})


def geo_metadata(table) -> dict:
    """GeoParquet 1.1 metadata of an index: `geometry` the primary column
    (WKB polygons in longitude and latitude, the default CRS), with
    `bbox` its covering."""
    from georvq import __version__
    bb = [b for b in table.column("bbox").to_pylist() if b is not None]
    col = {"encoding": "WKB", "geometry_types": ["Polygon"],
           "covering": {"bbox": {k: ["bbox", k]
                                 for k in ("xmin", "ymin", "xmax", "ymax")}}}
    if bb:
        col["bbox"] = [min(b["xmin"] for b in bb), min(b["ymin"] for b in bb),
                       max(b["xmax"] for b in bb), max(b["ymax"] for b in bb)]
    return {"version": "1.1.0", "primary_column": "geometry",
            "columns": {"geometry": col},
            "creator": {"library": "georvq", "version": __version__}}


def find_stores(root) -> list:
    """`*.georvq.zarr` stores under a local directory, sorted."""
    out = []
    for d, names, _ in os.walk(os.fspath(root)):
        hits = [n for n in names if n.endswith(SUFFIX)]
        out += [os.path.join(d, n) for n in hits]
        names[:] = [n for n in names if not n.endswith(SUFFIX)]
    return sorted(out)


def stem(p: str) -> str:
    """The product stem of a store or source path."""
    name = p.rstrip("/").rsplit("/", 1)[-1]
    return name[:-len(SUFFIX)] if name.endswith(SUFFIX) else os.path.splitext(name)[0]
