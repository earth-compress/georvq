"""The index of Tessera v1.1 stores, and reading a geotessera tile from it.

The columns every release's index starts with
(georvq.inference.store_index.CORE) are followed by Tessera's own, taken
from each store's header: `year`, `utm_zone`, `shard_row` and
`shard_col` (the shard on the zone's 4096 px shard grid), `zone_row` and
`zone_col` (the store's first pixel in the zone's grid), `zone_west`,
`zone_top` and `zone_px` (the zone grid's origin and pixel size),
`valid_fraction` (the share of the store's pixels with an embedding,
from its mask), `embeddings_version`, and `tiles`, the geotessera tiles
whose window touches the store: per tile its `name`, centre `lon` and
`lat`, and its window in the zone's grid (`row`, `col`, `height`,
`width`), as georvq.tessera.tiles.tiles_in_region computes it. A tile
that straddles shards is listed in each of their stores.

`find_tile` and `read_tile` use only the index: a tile's window is read
from the stores it touches and assembled, so the result has the shape,
georeferencing and nodata of the tile, as a store per tile gave before.
The stores' windows are on the shard grid rather than the tile's, so the
decoded values are those of the shard stores' windows.
"""
from __future__ import annotations

import io
import os
import posixpath
import re

def _tiles_type():
    import pyarrow as pa
    return pa.list_(pa.struct([
        ("name", pa.string()), ("lon", pa.float64()), ("lat", pa.float64()),
        ("row", pa.int64()), ("col", pa.int64()), ("height", pa.int64()),
        ("width", pa.int64())]))


def source_row(product, valid_fraction) -> dict:
    from georvq.tessera.tiles import tiles_in_region
    w, o = product["zone_window"], product["zone_origin"]
    sh = product.get("shard") or {}
    zone = int(product["utm_zone"])
    return {"year": int(product["year"]), "utm_zone": zone,
            "shard_row": sh.get("row"), "shard_col": sh.get("col"),
            "zone_row": int(w["row"]), "zone_col": int(w["col"]),
            "zone_west": float(o["west"]), "zone_top": float(o["top"]),
            "zone_px": float(o["px"]),
            "valid_fraction": float(valid_fraction),
            "embeddings_version": (product.get("embeddings") or {}).get(
                "version"),
            "tiles": tiles_in_region(zone, int(product["epsg"]), o["west"],
                                     o["top"], o["px"], w["row"], w["col"],
                                     w["height"], w["width"])}


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
        key = (int(p["year"]), int(p["utm_zone"]), p["zone_window"]["row"],
               p["zone_window"]["col"])
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
    schema = pa.schema([("year", pa.int16()), ("utm_zone", pa.int16()),
                        ("shard_row", pa.int32()), ("shard_col", pa.int32()),
                        ("zone_row", pa.int64()), ("zone_col", pa.int64()),
                        ("zone_west", pa.float64()), ("zone_top", pa.float64()),
                        ("zone_px", pa.float64()),
                        ("valid_fraction", pa.float64()),
                        ("embeddings_version", pa.string()),
                        ("tiles", _tiles_type())])
    out = index_table(paths, recs, pa.Table.from_pylist(rows, schema=schema))
    if out_path is not None:
        pq.write_table(out, out_path, compression="zstd")
    return out


# --------------------------------------------------------------------------
# reading a tile
# --------------------------------------------------------------------------

_INDEX_CACHE = {}


def _is_url(p) -> bool:
    return isinstance(p, str) and "://" in p


def load_index(index, storage_options=None):
    """(table, base) of an index: a pyarrow table, or a path or URL of the
    GeoParquet file, with `base` the directory its relative paths are
    relative to. URLs are read once per process."""
    import pyarrow.parquet as pq
    if hasattr(index, "column_names"):
        return index, None
    key = os.fspath(index)
    if key not in _INDEX_CACHE:
        if _is_url(key):
            import fsspec
            with fsspec.open(key, "rb", **(storage_options or {})) as f:
                t = pq.read_table(io.BytesIO(f.read()))
            base = key.rsplit("/", 1)[0]
        else:
            t = pq.read_table(key)
            base = os.path.dirname(os.path.abspath(key))
        _INDEX_CACHE[key] = (t, base)
    return _INDEX_CACHE[key]


def _tile_args(lon, lat, year):
    """(lon, lat) from numbers or a tile name such as "grid_26.15_44.45"
    (or "2024/grid_26.15_44.45", which also gives the year)."""
    if isinstance(lon, str):
        m = re.fullmatch(r"(?:(\d{4})/)?grid_(-?[\d.]+)_(-?[\d.]+)", lon)
        if m is None:
            raise ValueError(f"{lon!r} is not a tile name like grid_26.15_44.45")
        if m.group(1):
            year = int(m.group(1))
        lat = float(m.group(3))
        lon = float(m.group(2))
    if year is None or lat is None:
        raise ValueError("a tile needs lon, lat and year")
    return round(float(lon), 2), round(float(lat), 2), int(year)


def find_tile(index, lon, lat=None, year=None, storage_options=None) -> dict:
    """Where the tile centred at (lon, lat) in `year` is, from the index
    alone: {name, year, zone, epsg, window (row, col, height, width) in
    the zone's grid, transform (north-up, of the tile), stores: [{path,
    row, col, height, width}] with each store's window in the zone's
    grid}. `lon` may be a tile name. Raises KeyError for a tile no store
    touches."""
    from georvq.tessera.tiles import tile_name, utm_zone
    lon, lat, year = _tile_args(lon, lat, year)
    t, base = load_index(index, storage_options)
    name, zone = tile_name(lon, lat), utm_zone(lon)
    rows = [r for r in t.select(["path", "year", "utm_zone", "zone_row",
                                 "zone_col", "width", "height", "crs",
                                 "zone_west", "zone_top", "zone_px",
                                 "tiles"]).to_pylist()
            if r["year"] == year and r["utm_zone"] == zone
            and any(x["name"] == name for x in r["tiles"])]
    if not rows:
        raise KeyError(f"no store of the index holds {year}/{name}")
    win = next(x for x in rows[0]["tiles"] if x["name"] == name)
    px, west, top = rows[0]["zone_px"], rows[0]["zone_west"], rows[0]["zone_top"]
    stores = []
    for r in rows:
        p = r["path"]
        if base is not None and not (_is_url(p) or os.path.isabs(p)):
            p = (posixpath.join(base, p) if _is_url(base)
                 else os.path.join(base, p))
        stores.append({"path": p, "row": r["zone_row"], "col": r["zone_col"],
                       "height": r["height"], "width": r["width"]})
    return {"name": name, "year": year, "zone": zone,
            "epsg": int(rows[0]["crs"].split(":")[1]),
            "window": (win["row"], win["col"], win["height"], win["width"]),
            "transform": (px, 0.0, west + win["col"] * px, 0.0, -px,
                          top - win["row"] * px),
            "stores": stores}


def read_tile(index, lon, lat=None, year=None, quality=None, window=None,
              codec=None, storage_options=None, **open_kw):
    """The tile centred at (lon, lat) in `year`, read from the stores of
    the index at `quality` (a rung or a bound, as for `georvq.open`'s
    `read`) and assembled: float32 (128, height, width) in the release's
    output units, zero vectors at nodata and where no store covers the
    tile. `lon` may be a tile name ("grid_26.15_44.45", or
    "2024/grid_26.15_44.45"). `window`, ((row0, row1), (col0, col1)) in
    the tile's pixels, reads only that part of it. `codec` and `open_kw`
    go to `georvq.open`; the codec the first store loads is reused for
    the others. The tile's georeferencing is `find_tile`'s."""
    import numpy as np

    import georvq
    info = find_tile(index, lon, lat, year, storage_options)
    r0, c0, h, w = info["window"]
    if window is not None:
        (a, b), (c, d) = window
        if not (0 <= a < b <= h and 0 <= c < d <= w):
            raise ValueError(f"window {window} is outside the {h} x {w} tile")
        r0, c0, h, w = r0 + a, c0 + c, b - a, d - c
    out = None
    for st in info["stores"]:
        y0, y1 = max(r0, st["row"]), min(r0 + h, st["row"] + st["height"])
        x0, x1 = max(c0, st["col"]), min(c0 + w, st["col"] + st["width"])
        if y0 >= y1 or x0 >= x1:
            continue
        kw = dict(open_kw)
        if storage_options is not None and _is_url(st["path"]):
            kw["storage_options"] = storage_options
        with georvq.open(st["path"], codec=codec, **kw) as ds:
            a = ds.read(window=((y0 - st["row"], y1 - st["row"]),
                                (x0 - st["col"], x1 - st["col"])),
                        quality=quality)
            codec = ds.codec
        if out is None:
            out = np.zeros((a.shape[0], h, w), a.dtype)
        out[:, y0 - r0:y1 - r0, x0 - c0:x1 - c0] = a
    return out
