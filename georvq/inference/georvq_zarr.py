"""Prototype: georvq coded products as Zarr v3 groups.

One group per product, `{stem}.georvq.zarr`, holds the same records as a
georvq GeoParquet file (docs/GEORVQ_PARQUET.md) as uint8 arrays over the
window grid, plus the pixel validity mask:

    zarr.json           group: the georvq header, the aux metadata,
                        GeoZarr proj: and spatial: attributes, the store
                        manifest, and (by default) consolidated metadata
    mask    (H, W) bool             one chunk per mask-kind chunk, zstd;
                                    chunks with every pixel valid omitted
    base    (wy, wx, base bytes)    one chunk per base chunk
    <tier>  (wy, wx, levels, bytes) sharding_indexed, one shard for the
                                    array, inner chunks of one level of
                                    one tier chunk, no compression
    quality (wy, wx, quality bytes) chunked as base

The chunks are those of the header's layout, so a store has the chunk
sizes of the GeoParquet file it is written from. An inner chunk of a
tier holds the same bytes as the GeoParquet value of that level column
and chunk, edge chunks included (zarr pads them with the fill value 0,
as the GeoParquet writer pads with zero records), so the two containers
decode to the same values. zarr-python writes the shards: inner chunks
in its Morton order, the index at the end with the default index codecs
(bytes, crc32c), and inner chunks of zeros, among them the chunks the
header lists as empty, left out and read as zeros.

Consolidated metadata is a zarr-python convention rather than part of
the v3 specification. `parquet_to_zarr(..., consolidated=False)` leaves
it out, and the reader then fetches the zarr.json of each array.

`ZarrReader` reads a store from a directory or an http(s) URL with the
range machinery of georvq_parquet (RangeSource): it fetches the group
zarr.json (and the array zarr.json files in one parallel round when the
metadata is not consolidated), then the shard index of each tier a read
needs and has not seen (one parallel round), then the merged data
ranges. It reuses RVQParquetReader's planning, cache and unpacking, so
its `read` returns what RVQParquetReader.read returns.

The writer needs zarr-python; the reader needs numpy and, for masks,
numcodecs.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from georvq.inference.georvq_parquet import (
    BACKOFF, CACHE_BYTES, CONCURRENCY, FIRST_READ, GAP, RETRIES, TIMEOUT,
    RangeSource, RVQParquetReader, check_header, merge_ranges, span)

SUFFIX = ".georvq.zarr"
STORE_KEY = "georvq_store"
STORE_FORMAT = "georvq-zarr"
STORE_MAJOR, STORE_MINOR = 0, 2
MASK_ZSTD_LEVEL = 19
_MISSING = 2 ** 64 - 1
_ZSTD_MAGIC = 0xFD2FB528

PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/proj/"
                  "refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/proj/blob/v0.1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj",
    "description": "Coordinate reference system information for geospatial data"}
SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/"
                  "refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v0.1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial",
    "description": "Spatial coordinate information"}


def zarr_name(source) -> str:
    """`{stem}.georvq.zarr` for a source product or a .georvq.parquet."""
    name = os.path.basename(os.fspath(source).rstrip("/"))
    for ext in (".georvq.parquet", SUFFIX, ".tiff", ".tif", ".SAFE", ".nc"):
        if name.endswith(ext):
            return name[:-len(ext)] + SUFFIX
    return name + SUFFIX


def is_zarr(path_or_url) -> bool:
    return os.fspath(path_or_url).rstrip("/").endswith(".zarr")


# --------------------------------------------------------------------------
# shard index
# --------------------------------------------------------------------------

def _crc32c_table():
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        t.append(c)
    return t


_CRC_TABLE = None


def crc32c(data: bytes) -> int:
    try:
        import google_crc32c
        return int(google_crc32c.value(bytes(data)))
    except ImportError:
        pass
    global _CRC_TABLE
    if _CRC_TABLE is None:
        _CRC_TABLE = _crc32c_table()
    c, t = 0xFFFFFFFF, _CRC_TABLE
    for b in data:
        c = (c >> 8) ^ t[(c ^ b) & 0xFF]
    return c ^ 0xFFFFFFFF


def index_bytes(n_chunks: int) -> int:
    """Bytes of a shard index of `n_chunks` (uint64 pairs and a crc32c)."""
    return 16 * int(n_chunks) + 4


def encode_index(offsets_and_lengths) -> bytes:
    """Shard index with the default index codecs (bytes little-endian,
    crc32c), from a (..., 2) array of (offset, length), missing as
    2**64 - 1."""
    raw = np.ascontiguousarray(offsets_and_lengths, "<u8").tobytes()
    return raw + struct.pack("<I", crc32c(raw))


def decode_index(buf: bytes, chunks_per_shard) -> np.ndarray:
    n = int(np.prod(chunks_per_shard))
    if len(buf) != index_bytes(n):
        raise ValueError(f"shard index of {len(buf)} bytes, expected "
                         f"{index_bytes(n)}")
    raw, crc = buf[:-4], struct.unpack("<I", buf[-4:])[0]
    if crc32c(raw) != crc:
        raise ValueError("shard index fails its crc32c")
    return np.frombuffer(raw, "<u8").reshape(tuple(chunks_per_shard) + (2,))


def zstd_content_size(buf: bytes):
    """Frame_Content_Size from the header of the zstd frame at the start
    of `buf`, or None when the header does not declare it."""
    if len(buf) < 5 or struct.unpack("<I", buf[:4])[0] != _ZSTD_MAGIC:
        raise ValueError("not a zstd frame")
    fhd = buf[4]
    fcs_flag, single, did_flag = fhd >> 6, (fhd >> 5) & 1, fhd & 3
    at = 5 + (0 if single else 1) + (0, 1, 2, 4)[did_flag]
    n = (1 if single else 0, 2, 4, 8)[fcs_flag]
    if n == 0:
        return None
    if len(buf) < at + n:
        raise ValueError("truncated zstd frame header")
    v = int.from_bytes(buf[at:at + n], "little")
    return v + 256 if n == 2 else v


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------

def _window_transform(t, W):
    a, b, c, d, e, f = (float(v) for v in t[:6])
    return [a * W, b * W, c, d * W, e * W, f]


def _bbox(t, h, w):
    a, b, c, d, e, f = (float(v) for v in t[:6])
    xs = [a * x + b * y + c for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
    ys = [d * x + e * y + f for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
    return [min(xs), min(ys), max(xs), max(ys)]


def _spatial(dims, shape, transform):
    if transform is None:
        return {}
    return {"spatial:dimensions": list(dims), "spatial:shape": list(shape),
            "spatial:transform": [float(v) for v in transform],
            "spatial:bbox": _bbox(transform, *shape),
            "spatial:registration": "pixel"}


def _north_transform(header, geo, h, w):
    """The product's north-up affine from the header, or from the
    GeoParquet bbox of an axis-aligned product; None when neither."""
    t = (header.get("product") or {}).get("transform")
    if t is not None:
        return [float(v) for v in t[:6]]
    bb = ((geo or {}).get("columns", {}).get("geometry", {}) or {}).get("bbox")
    if bb is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in bb)
    return [(x1 - x0) / w, 0.0, x0, 0.0, -(y1 - y0) / h, y1]


def _crs_attrs(product, geo):
    out = {}
    if product.get("epsg"):
        out["proj:code"] = f"EPSG:{int(product['epsg'])}"
    wkt = product.get("crs_wkt")
    if wkt:
        try:
            import pyproj
            out["proj:wkt2"] = pyproj.CRS.from_wkt(wkt).to_wkt("WKT2_2019")
        except ImportError:
            pass
    crs = ((geo or {}).get("columns", {}).get("geometry", {}) or {}).get("crs")
    if crs:
        out["proj:projjson"] = crs
    return out


def _chunk_path(name, coords):
    return f"{name}/c/" + "/".join(str(int(c)) for c in coords)


def parquet_to_zarr(src, out, consolidated=True) -> str:
    """Write the Zarr store of a georvq GeoParquet file `src` (a path or
    an open RVQParquetReader) at `out`, a directory that must not exist.
    The records are copied, not re-encoded, and the chunks are those of
    the file. `consolidated` adds zarr-python's consolidated metadata to
    the group zarr.json. Returns `out`."""
    import zarr
    from zarr.codecs import ZstdCodec

    rd = src if isinstance(src, RVQParquetReader) else RVQParquetReader(src)
    out = os.fspath(out)
    if os.path.exists(out):
        raise FileExistsError(out)
    lay, hdr = rd.layout, rd.header
    gy, gx = rd.window_grid
    H, Wd = rd.height, rd.width
    W = lay.window_px
    north = _north_transform(hdr, rd.geo, H, Wd)
    crs = _crs_attrs(hdr["product"], rd.geo)

    def chunks(kind):
        ny, nx = rd.chunk_grids[kind]
        return [(i, j) for i in range(ny) for j in range(nx)
                if (i, j) not in rd.empty[kind]]

    needs = []
    for c in lay.column_order:
        kind = lay.kind_of[c]
        needs += [(c, i, j) for i, j in chunks(kind)
                  if c != "mask" or (i, j) in rd.masked]
    vals = rd.values(needs)

    root = zarr.open_group(out, mode="w-", zarr_format=3)
    win_attrs = {"zarr_conventions": [PROJ_CONVENTION, SPATIAL_CONVENTION],
                 **({"proj:code": crs["proj:code"]} if "proj:code" in crs else {}),
                 **_spatial(("wy", "wx"), (gy, gx),
                            None if north is None else _window_transform(north, W))}
    store_info = {"format": STORE_FORMAT,
                  "version": f"{STORE_MAJOR}.{STORE_MINOR}",
                  "status": "prototype", "arrays": {}, "objects": {}}

    def records(col, kind):
        g = lay.groups[kind]
        n, rb = g.chunk_windows, lay.columns[col].record_bytes
        ny, nx = rd.chunk_grids[kind]
        a = np.zeros((ny * n, nx * n, rb), np.uint8)
        for i, j in chunks(kind):
            a[i * n:(i + 1) * n, j * n:(j + 1) * n] = np.frombuffer(
                vals[(col, i, j)], np.uint8).reshape(n, n, rb)
        return a

    # base and quality: plain chunked arrays, empty chunks not written
    for col in ["base"] + (["quality"] if lay.quality is not None else []):
        kind = lay.kind_of[col]
        n = lay.groups[kind].chunk_windows
        rb = lay.columns[col].record_bytes
        arr = root.create_array(
            col, shape=(gy, gx, rb), dtype="uint8", chunks=(n, n, rb),
            compressors=None, filters=None, fill_value=0,
            dimension_names=["wy", "wx", f"{col}_byte"], attributes=win_attrs,
            config={"write_empty_chunks": True})
        a = records(col, kind)
        for i, j in chunks(kind):
            y1, x1 = min(gy, (i + 1) * n), min(gx, (j + 1) * n)
            arr[i * n:y1, j * n:x1] = a[i * n:y1, j * n:x1]
        store_info["arrays"][col] = {"kind": kind, "columns": [col]}

    # tiers outside base: one shard, one level of one chunk per inner chunk
    for t in lay.tiers:
        if t.in_base:
            continue
        g = lay.groups[t.name]
        n = g.chunk_windows
        rb = lay.columns[t.columns[0]].record_bytes
        ny, nx = rd.chunk_grids[t.name]
        arr = root.create_array(
            t.name, shape=(gy, gx, t.levels, rb), dtype="uint8",
            chunks=(n, n, 1, rb), shards=(ny * n, nx * n, t.levels, rb),
            compressors=None, filters=None, fill_value=0,
            dimension_names=["wy", "wx", f"{t.name}_level", f"{t.name}_byte"],
            attributes=win_attrs, config={"write_empty_chunks": False})
        full = np.stack([records(c, t.name) for c in t.columns], 2)
        arr[...] = full[:gy, :gx]
        store_info["arrays"][t.name] = {"kind": t.name,
                                        "columns": list(t.columns)}

    # mask: bool, zstd; all-valid chunks omitted, empty chunks all False
    mk = lay.mask_kind
    p = lay.groups[mk].chunk_px
    mask = root.create_array(
        "mask", shape=(H, Wd), dtype="bool", chunks=(p, p),
        compressors=ZstdCodec(level=MASK_ZSTD_LEVEL), filters=None,
        fill_value=True, dimension_names=["y", "x"],
        attributes={"zarr_conventions": [PROJ_CONVENTION, SPATIAL_CONVENTION],
                    **({"proj:code": crs["proj:code"]} if "proj:code" in crs
                       else {}),
                    **_spatial(("y", "x"), (H, Wd), north)},
        config={"write_empty_chunks": False})
    ny, nx = rd.chunk_grids[mk]
    for i in range(ny):
        for j in range(nx):
            y0, x0 = i * p, j * p
            bh, bw = min(p, H - y0), min(p, Wd - x0)
            if (i, j) in rd.empty[mk]:
                mask[y0:y0 + bh, x0:x0 + bw] = np.zeros((bh, bw), bool)
            elif (i, j) in rd.masked:
                mask[y0:y0 + bh, x0:x0 + bw] = rd._decode_mask(
                    vals[("mask", i, j)], (bh, bw))
    store_info["arrays"]["mask"] = {"kind": mk, "columns": ["mask"]}

    for dirpath, _, files in os.walk(out):
        for f in files:
            if f != "zarr.json":
                full = os.path.join(dirpath, f)
                store_info["objects"][os.path.relpath(full, out)] = \
                    os.path.getsize(full)
    store_info["objects"] = dict(sorted(store_info["objects"].items()))

    attrs = {"georvq": hdr,
             "georvq_aux": None if rd.aux is None else json.loads(rd.aux),
             "zarr_conventions": [PROJ_CONVENTION, SPATIAL_CONVENTION],
             **crs, **_spatial(("y", "x"), (H, Wd), north),
             STORE_KEY: store_info}
    root.update_attributes(attrs)
    if consolidated:
        # zarr-python warns on every write that consolidated metadata is
        # not part of the v3 specification
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Consolidated metadata")
            zarr.consolidate_metadata(out)
    return out


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

class ZarrReader(RVQParquetReader):
    """Reads a georvq Zarr store from a directory or an http(s) URL,
    with the interface of RVQParquetReader (`read`, `read_mask`,
    `values`, `needs`, `stats`, `trace`, `header`, `aux`, `layout`).

    Opening fetches the group zarr.json (one read of its first
    `first_read` bytes, a second when it is longer) and, when it holds
    no consolidated metadata, the zarr.json of each array in one
    parallel round. A read lists the (column, chunk) values it needs
    exactly as RVQParquetReader does, leaves out the cached ones,
    fetches the shard index of each tier it needs and has not seen (all
    in one parallel round), maps each value to a byte range of its
    object, merges ranges per object whose gap is at most `gap` bytes,
    and fetches them in parallel. Each object's size is checked against
    the store manifest, and its ETag or Last-Modified against the first
    one seen. The cache key of the store is its URL or path and the
    hash of its metadata. `headers` go with every request, as in
    RVQParquetReader."""

    driver = "GeoRVQZarr"

    def __init__(self, path_or_url, first_read=FIRST_READ, gap=GAP,
                 concurrency=CONCURRENCY, cache=None, cache_bytes=CACHE_BYTES,
                 cache_dir=None, trace=None, timeout=TIMEOUT, retries=RETRIES,
                 backoff=BACKOFF, headers=None):
        self.source = os.fspath(path_or_url).rstrip("/")
        self.trace = trace
        self.headers = dict(headers or {})
        self.gap = int(gap)
        self.concurrency = max(1, int(concurrency))
        self.timeout = None if timeout is None else float(timeout)
        self.retries = max(0, int(retries))
        self.backoff = float(backoff)
        self._url = self.source.startswith(("http://", "https://"))
        self._lock = threading.Lock()
        self.stats = {"requests": 0, "bytes": 0, "retries": 0,
                      "cache_hits": 0, "cache_bytes": 0}
        self.version_tag = None
        self.size = None
        self._fd = None
        self._objects = {}
        self._index = {}
        try:
            self._open_store(int(first_read), cache, cache_bytes, cache_dir)
        except BaseException:
            self.close()
            raise

    # ---- objects ----

    def _object(self, path) -> RangeSource:
        with self._lock:
            o = self._objects.get(path)
        if o is not None:
            return o
        src = (f"{self.source}/{path}" if self._url
               else os.path.join(self.source, *path.split("/")))
        o = RangeSource(src, self.trace, self.timeout, self.retries,
                        self.backoff, self.concurrency, stats=self.stats,
                        lock=self._lock, size=self._sizes.get(path),
                        headers=self.headers)
        o.adopt_tag = True
        with self._lock:
            o = self._objects.setdefault(path, o)
        return o

    def close(self) -> None:
        for o in list(getattr(self, "_objects", {}).values()):
            o.close()

    @property
    def identity(self) -> list:
        return [self.source if self._url else os.path.realpath(self.source),
                getattr(self, "_meta_sha", None)]

    # ---- open ----

    def _json(self, path, first_read):
        """Bytes of a whole metadata object, by one prefix read (two
        when it is longer than `first_read`)."""
        o = self._object(path)
        data, size = o._prefix(first_read)
        o.size = size
        if size > len(data):
            data += o._read((len(data), size - len(data)))
        return o, data

    def _open_store(self, first_read, cache, cache_bytes, cache_dir):
        self._sizes = {}
        meta, data = self._json("zarr.json", first_read)
        self.version_tag = meta.version_tag
        with span(self.trace, "cpu", "parse zarr.json"):
            z = json.loads(data)
        if z.get("zarr_format") != 3 or z.get("node_type") != "group":
            raise ValueError(f"{self.source}: not a Zarr v3 group")
        attrs = z.get("attributes") or {}
        if "georvq" not in attrs or STORE_KEY not in attrs:
            raise ValueError(f"{self.source}: no georvq header or store manifest")
        info = self.store_info = attrs[STORE_KEY]
        if info.get("format") != STORE_FORMAT or int(
                str(info.get("version", "")).split(".")[0]) != STORE_MAJOR:
            raise ValueError(f"{self.source}: store {info.get('format')!r} "
                             f"version {info.get('version')!r}")
        self.attrs = attrs
        self.header = attrs["georvq"]
        check_header(self.header, self.source)
        aux = attrs.get("georvq_aux")
        self.aux = None if aux is None else json.dumps(aux)
        self.geo = None
        self._from_header()
        self._sizes = {str(k): int(v) for k, v in info["objects"].items()}
        names = list(info["arrays"])
        cm = (z.get("consolidated_metadata") or {}).get("metadata") or {}
        sha = hashlib.sha1(data)
        if not all(n in cm for n in names):
            def one(name):
                return self._json(f"{name}/zarr.json", first_read)[1]
            if self._url and len(names) > 1:
                with ThreadPoolExecutor(min(self.concurrency, len(names))) as ex:
                    raw = list(ex.map(one, names))
            else:
                raw = [one(n) for n in names]
            cm = {}
            for name, d in zip(names, raw):
                md = json.loads(d)
                if md.get("zarr_format") != 3 or md.get("node_type") != "array":
                    raise ValueError(f"{self.source}: {name} is not a Zarr v3 "
                                     f"array")
                cm[name] = md
                sha.update(d)
        self.consolidated = bool(z.get("consolidated_metadata"))
        self.arrays = {n: self._check_array(n, cm[n]) for n in names}
        self._meta_sha = sha.hexdigest()
        self._set_cache(json.dumps(self.identity), cache, cache_bytes, cache_dir)

    def _check_array(self, name, md) -> dict:
        """The array's chunk shape (inner chunk for a tier), checked
        against the layout and the codecs this reader decodes."""
        lay = self.layout
        kind = self.store_info["arrays"][name]["kind"]
        cs = md["chunk_grid"]["configuration"]["chunk_shape"]
        cke = md.get("chunk_key_encoding") or {}
        if cke.get("name") != "default" or (cke.get("configuration") or {}).get(
                "separator", "/") != "/":
            raise ValueError(f"{name}: chunk key encoding {cke}")
        codecs = md["codecs"]
        names = [c["name"] for c in codecs]
        if name == "mask":
            p = lay.groups[kind].chunk_px
            if md["data_type"] != "bool" or cs != [p, p] or names != [
                    "bytes", "zstd"]:
                raise ValueError(f"mask: {md['data_type']} {cs} {names}")
            return {"chunk_shape": cs}
        if md["data_type"] != "uint8":
            raise ValueError(f"{name}: data type {md['data_type']}")
        n = lay.groups[kind].chunk_windows
        if name in ("base", "quality"):
            rb = lay.columns[name].record_bytes
            if cs != [n, n, rb] or names != ["bytes"]:
                raise ValueError(f"{name}: chunks {cs}, codecs {names}")
            return {"chunk_shape": cs}
        t = next(t for t in lay.tiers if t.name == name)
        rb = lay.columns[t.columns[0]].record_bytes
        ny, nx = self.chunk_grids[kind]
        if names != ["sharding_indexed"]:
            raise ValueError(f"{name}: codecs {names}")
        conf = codecs[0]["configuration"]
        inner = [c["name"] for c in conf["codecs"]]
        idx = [c["name"] for c in conf["index_codecs"]]
        if (cs != [ny * n, nx * n, t.levels, rb]
                or conf["chunk_shape"] != [n, n, 1, rb] or inner != ["bytes"]
                or idx != ["bytes", "crc32c"]
                or conf.get("index_location", "end") != "end"):
            raise ValueError(f"{name}: shard {cs}, inner {conf['chunk_shape']}, "
                             f"codecs {inner}, index {idx}")
        return {"chunk_shape": conf["chunk_shape"],
                "chunks_per_shard": (ny, nx, t.levels, 1),
                "chunk_bytes": n * n * rb,
                "object": _chunk_path(name, (0, 0, 0, 0))}

    def _check_index(self, kind, idx) -> np.ndarray:
        a = self.arrays[kind]
        size = self._sizes.get(a["object"])
        have = idx[..., 0] != _MISSING
        end = (idx[..., 0][have] + idx[..., 1][have]).astype(np.int64)
        if (idx[..., 1][have] != a["chunk_bytes"]).any() or (
                size is not None and end.size and end.max() > size - index_bytes(
                    int(np.prod(a["chunks_per_shard"])))):
            raise ValueError(f"{self.source}: {kind} shard index does not fit "
                             f"its object")
        return idx

    def _load_indices(self, kinds) -> None:
        """Fetch the shard index of each tier in `kinds`, in parallel."""
        kinds = [k for k in kinds if k not in self._index]
        if not kinds:
            return

        def one(kind):
            a = self.arrays[kind]
            n = index_bytes(int(np.prod(a["chunks_per_shard"])))
            o = self._object(a["object"])
            if o.size is not None:
                buf = o._read((o.size - n, n))
            else:
                buf, o.size = o._suffix(n)
            return kind, self._check_index(
                kind, decode_index(buf, a["chunks_per_shard"]))
        if self._url and len(kinds) > 1:
            with ThreadPoolExecutor(min(self.concurrency, len(kinds))) as ex:
                got = list(ex.map(one, kinds))
        else:
            got = [one(k) for k in kinds]
        with self._lock:
            for kind, idx in got:
                self._index[kind] = idx

    # ---- ranges ----

    def _locate(self, need):
        """(object path, offset, length) of a value, or None for an inner
        chunk absent from its shard."""
        col, i, j = need
        if col == "mask":
            path = _chunk_path("mask", (i, j))
            return path, 0, self._sizes[path]
        if col in ("base", "quality"):
            path = _chunk_path(col, (i, j, 0))
            return path, 0, self._sizes[path]
        kind = self.layout.kind_of[col]
        t = next(t for t in self.layout.tiers if t.name == kind)
        level = t.columns.index(col)
        o, ln = (int(v) for v in self._index[kind][i, j, level, 0])
        if o == _MISSING:
            return None
        return self.arrays[kind]["object"], o, ln

    def _tier_kinds(self, needs):
        return {self.layout.kind_of[n[0]] for n in needs
                if n[0] not in ("base", "mask", "quality")}

    def plan(self, wy0, wy1, wx0, wx1, depths=None, base=True, mask=True,
             quality=False) -> list:
        """(object, offset, length) requests a read would make now, after
        the shard indices it needs (fetched here if missing)."""
        cols = self.columns_for(depths, base, mask, quality)
        needs = [n for n in self.needs(wy0, wy1, wx0, wx1, cols)
                 if not self.cache.has(self._key(n), self._disk)]
        self._load_indices(self._tier_kinds(needs))
        return [(p, o, ln) for p, rs in self._object_ranges(needs).items()
                for o, ln in rs]

    def _object_ranges(self, needs) -> dict:
        by = {}
        for n in needs:
            loc = self._locate(n)
            if loc is not None:
                by.setdefault(loc[0], []).append(loc[1:])
        return {p: merge_ranges(rs, self.gap) for p, rs in by.items()}

    # ---- values ----

    def _decode_mask(self, v, shape) -> np.ndarray:
        if len(v) == 0:
            return np.ones(shape, bool)
        from numcodecs.zstd import Zstd
        p = self.layout.groups[self.layout.mask_kind].chunk_px
        # refuse before decompressing a chunk that declares another size
        declared = zstd_content_size(v)
        if declared != p * p:
            raise ValueError(f"mask chunk declares {declared} bytes, expected "
                             f"{p * p}")
        raw = Zstd().decode(v)
        if len(raw) != p * p:
            raise ValueError(f"mask chunk of {len(raw)} bytes, expected {p * p}")
        a = np.frombuffer(raw, np.uint8).reshape(p, p)
        return a[:shape[0], :shape[1]].astype(bool)

    def values(self, needs) -> dict:
        """{(column, chunk_row, chunk_col): value bytes} for `needs`, from
        the cache where held and otherwise in merged range requests."""
        out, missing, hits, hit_bytes = {}, [], 0, 0
        for n in needs:
            v = self.cache.get(self._key(n), self._disk)
            if v is not None and self._value_error(n, v) is not None:
                v = None
            if v is None:
                missing.append(n)
            else:
                out[n] = v
                hits += 1
                hit_bytes += len(v)
        with self._lock:
            self.stats["cache_hits"] += hits
            self.stats["cache_bytes"] += hit_bytes
        if not missing:
            return out
        self._load_indices(self._tier_kinds(missing))
        with span(self.trace, "cpu", "plan ranges"):
            ranges = self._object_ranges(missing)
            jobs = [(p, r) for p, rs in ranges.items() for r in rs]

        def get(job):
            return self._object(job[0])._read(job[1])
        if self._url and self.concurrency > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(min(self.concurrency, len(jobs))) as ex:
                fetched = list(ex.map(get, jobs))
        else:
            fetched = [get(j) for j in jobs]
        got = {}
        for (p, (o, ln)), data in zip(jobs, fetched):
            got.setdefault(p, []).append((o, data))
        with span(self.trace, "cpu", "slice values"):
            for n in missing:
                loc = self._locate(n)
                if loc is None:
                    # an inner chunk absent from its shard reads as zeros
                    kind = self.layout.kind_of[n[0]]
                    v = bytes(self.arrays[kind]["chunk_bytes"])
                else:
                    p, o, ln = loc
                    v = next(d[o - a:o - a + ln] for a, d in got[p]
                             if a <= o and o + ln <= a + len(d))
                err = self._value_error(n, v)
                if err is not None:
                    raise ValueError(f"{self.source}: {n} holds {err}")
                out[n] = v
                self.cache.put(self._key(n), v, self._disk)
        return out

    def kind_bytes(self) -> dict:
        """Bytes of each array's objects."""
        out = {}
        for path, n in self._sizes.items():
            name = path.split("/")[0]
            out[name] = out.get(name, 0) + n
        return out


def open_reader(path_or_url, **kw):
    """ZarrReader for a .zarr path or URL, RVQParquetReader otherwise."""
    if is_zarr(path_or_url):
        return ZarrReader(path_or_url, **kw)
    return RVQParquetReader(path_or_url, **kw)
