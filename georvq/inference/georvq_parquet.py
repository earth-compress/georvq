"""Reader and writer for georvq GeoParquet files (docs/GEORVQ_PARQUET.md).

A file holds one source product. Its windows' fixed-width code records
are grouped into square spatial chunks, with a chunk size chosen per
kind: `base` (the base record, for example a transmitted mean and the
coarsest tier's levels), each finer tier by its name, and `quality`
(per-window quality bytes, chunked as base). Each row, and each row
group, is one chunk of one kind: its footprint as a WKB polygon with a
GeoParquet 1.1 bbox covering, and its windows' records as binary
values, one column per tier level. The pixel validity mask is a column
of the base chunks, after `base`. A chunk none of whose windows has a
valid pixel is empty: its data columns are null and the header lists
it. Row groups are ordered by kind, then row-major over that kind's
chunk grid. The codec header is JSON in the file's key-value metadata
under `georvq`.

Code columns are written uncompressed and without dictionary encoding,
so each column chunk is one data page around one value and its byte
range follows from the footer. The reader opens a file with one suffix
read, plans the byte ranges a window region needs at a depth tuple from
the footer, merges adjacent ranges and small gaps, fetches them (in
parallel over HTTP), and keeps the decoded values per (chunk, column)
in a cache, so later reads fetch only what they lack. A read fetches
no empty chunk, and no chunk of another kind whose windows in the read
all lie in empty chunks. The file is a
standard Parquet file, so pyarrow or DuckDB can read it without this
module.

This module is source-agnostic and depends only on numpy, pyarrow and
the standard library, so a reader can be installed without torch.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import struct
import threading
import urllib.request
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from georvq.inference.bitpack import pack, record_bytes, unpack

FORMAT = "georvq-parquet"
MAJOR = 1
MINOR = 1
STATUS = "placeholder"
SUFFIX = ".georvq.parquet"
FIRST_READ = 256 * 1024
GAP = 16 * 1024
CONCURRENCY = 16
CACHE_BYTES = 512 * 1024 * 1024
MASK_CODEC = "packbits-zlib"
HEADER_KEY = "georvq"
AUX_KEY = "georvq_aux"
GEO_KEY = "geo"
_BBOX = ("xmin", "ymin", "xmax", "ymax")
_STAT_COLUMNS = ["kind", "product_id", "chunk_row", "chunk_col"] + [
    f"bbox.{k}" for k in _BBOX]
_WRITER_KEYS = ("format", "version", "status", "layout")
_RESERVED = ("base", "mask", "quality")


def tier_names(n_tiers: int) -> list:
    """Column-name stems of the tiers, coarsest first."""
    if n_tiers == 3:
        return ["coarse", "mid", "fine"]
    return [f"tier{i}" for i in range(n_tiers)]


def output_name(source) -> str:
    """`{source stem}.georvq.parquet` for a source product path."""
    stem = os.path.basename(os.fspath(source).rstrip("/"))
    for ext in (".tiff", ".tif", ".SAFE", ".zarr", ".nc"):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    return stem + SUFFIX


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Column:
    """A code column. `fields` is a list of (name, count, bits); a record
    is one window's values, packed MSB-first and padded to a byte."""
    name: str
    fields: tuple

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(
            (str(n), int(c), int(b)) for n, c, b in self.fields))

    @property
    def widths(self) -> list:
        return [b for _, c, b in self.fields for _ in range(c)]

    @property
    def n_values(self) -> int:
        return sum(c for _, c, _ in self.fields)

    @property
    def record_bytes(self) -> int:
        return record_bytes(self.widths)


@dataclass(frozen=True)
class Tier:
    """A tier's levels and the columns holding them. A tier stored in
    the base record has `columns == ("base",)` and is always read whole."""
    name: str
    levels: int
    columns: tuple

    @property
    def in_base(self) -> bool:
        return self.columns == ("base",)


@dataclass(frozen=True)
class Group:
    """One kind of row group: its chunk side in pixels and windows, and
    its data columns in file order."""
    kind: str
    chunk_px: int
    chunk_windows: int
    columns: tuple


class Layout:
    """Columns and chunk geometry of a file.

    window_px: window side in pixels. chunk_px: chunk side in pixels, a
    multiple of window_px, either one int for every kind or a dict with
    an entry for `base` and for each tier outside base; quality chunks
    follow base. base: fields of the base record. tiers: list of (name,
    levels, fields), fields None for a tier carried in base (only the
    first tier can be). quality: fields of the quality record, or None
    for no quality column. mask_kind: the kind whose chunks hold the
    pixel validity mask, `base` or a tier outside base.

    Kinds, in file order, are `base`, each tier outside base, and
    `quality`. The mask follows `base` in the base kind, and is the first
    column of a tier's kind, so either way a read of the kind's leading
    columns covers it in the same range."""

    def __init__(self, window_px, chunk_px, base, tiers, quality=None,
                 mask_kind="base"):
        self.window_px = W = int(window_px)
        self.base = Column("base", base)
        self.tiers, self.level_columns = [], []
        for i, (name, levels, fields) in enumerate(tiers):
            if name in _RESERVED or name in [t.name for t in self.tiers]:
                raise ValueError(f"tier name {name!r} is reserved or repeated")
            if fields is None:
                if i:
                    raise ValueError("only the first tier can live in base")
                self.tiers.append(Tier(name, int(levels), ("base",)))
                continue
            cols = [Column(f"{name}_{l}", fields) for l in range(1, int(levels) + 1)]
            self.level_columns += cols
            self.tiers.append(Tier(name, int(levels), tuple(c.name for c in cols)))
        self.quality = None if quality is None else Column("quality", quality)
        self.columns = {c.name: c for c in [self.base] + self.level_columns
                        + ([self.quality] if self.quality else [])}

        kinds = ["base"] + [t.name for t in self.tiers if not t.in_base]
        if isinstance(chunk_px, dict):
            extra = set(chunk_px) - set(kinds)
            missing = [k for k in kinds if k not in chunk_px]
            if extra or missing:
                raise ValueError(f"chunk_px keys {sorted(chunk_px)}, expected {kinds}")
            px = {k: int(chunk_px[k]) for k in kinds}
        else:
            px = {k: int(chunk_px) for k in kinds}
        for k, p in px.items():
            if p <= 0 or p % W:
                raise ValueError(f"{k}: chunk_px {p} is not a positive multiple "
                                 f"of window_px {W}")
        if mask_kind not in kinds:
            raise ValueError(f"mask_kind {mask_kind!r} not in {kinds}")
        self.mask_kind = mask_kind
        groups = []
        for k in kinds:
            if k == "base":
                cols = ("base", "mask") if k == mask_kind else ("base",)
            else:
                cols = next(t.columns for t in self.tiers if t.name == k)
                if k == mask_kind:
                    cols = ("mask",) + cols
            groups.append(Group(k, px[k], px[k] // W, cols))
        if self.quality is not None:
            groups.append(Group("quality", px["base"], px["base"] // W,
                                ("quality",)))
        self.groups = OrderedDict((g.kind, g) for g in groups)
        self.kind_of = {c: g.kind for g in groups for c in g.columns}

    @property
    def column_order(self) -> list:
        """Data columns in file order."""
        return [c for g in self.groups.values() for c in g.columns]

    def chunk_grid(self, kind, height, width) -> tuple:
        p = self.groups[kind].chunk_px
        return (-(-int(height) // p), -(-int(width) // p))

    def window_grid(self, height, width) -> tuple:
        w = self.window_px
        return (-(-int(height) // w), -(-int(width) // w))

    def to_json(self) -> dict:
        def col(c):
            return {"name": c.name, "record_bytes": c.record_bytes,
                    "fields": [{"name": n, "count": k, "bits": b}
                               for n, k, b in c.fields]}
        return {"window_px": self.window_px,
                "chunk_order": "row-major", "window_order": "row-major",
                "groups": [{"kind": g.kind, "chunk_px": g.chunk_px,
                            "chunk_windows": g.chunk_windows,
                            "columns": list(g.columns)}
                           for g in self.groups.values()],
                "mask": {"column": "mask", "kind": self.mask_kind,
                         "codec": MASK_CODEC},
                "columns": [col(self.columns[n]) for n in self.column_order
                            if n != "mask"],
                "tiers": [{"name": t.name, "levels": t.levels,
                           "columns": list(t.columns)} for t in self.tiers]}

    @classmethod
    def from_json(cls, d: dict) -> "Layout":
        cols = {c["name"]: c for c in d["columns"]}

        def fields(name):
            c = cols[name]
            f = [(x["name"], x["count"], x["bits"]) for x in c["fields"]]
            if Column(name, f).record_bytes != int(c["record_bytes"]):
                raise ValueError(f"column {name}: record_bytes "
                                 f"{c['record_bytes']} disagrees with its fields")
            return f
        tiers = []
        for t in d["tiers"]:
            if t["columns"] == ["base"]:
                tiers.append((t["name"], t["levels"], None))
            else:
                tiers.append((t["name"], t["levels"], fields(t["columns"][0])))
        px = {g["kind"]: g["chunk_px"] for g in d["groups"] if g["kind"] != "quality"}
        lay = cls(d["window_px"], px, fields("base"), tiers,
                  fields("quality") if "quality" in cols else None,
                  d["mask"]["kind"])
        if [g["kind"] for g in d["groups"]] != list(lay.groups) or any(
                list(lay.groups[g["kind"]].columns) != g["columns"]
                or lay.groups[g["kind"]].chunk_px != g["chunk_px"]
                for g in d["groups"]):
            raise ValueError("layout groups disagree with the columns and tiers")
        return lay


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def encode_mask(valid) -> bytes:
    """Mask value of one chunk: b"" when every pixel is valid, otherwise
    zlib of numpy.packbits of the row-major bits (1 = valid)."""
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return b""
    return zlib.compress(np.packbits(valid.ravel()).tobytes(), 9)


def decode_mask(data: bytes, shape) -> np.ndarray:
    if not data:
        return np.ones(shape, bool)
    bits = np.frombuffer(zlib.decompress(data), dtype=np.uint8)
    n = int(shape[0]) * int(shape[1])
    return np.unpackbits(bits, count=n).reshape(shape).astype(bool)


def wkb_polygon(ring) -> bytes:
    """Little-endian WKB of a polygon with one closed ring of (x, y)."""
    return (struct.pack("<BII", 1, 3, 1) + struct.pack("<I", len(ring))
            + b"".join(struct.pack("<dd", x, y) for x, y in ring))


def chunk_footprint(transform, cy, cx, chunk_px, height, width):
    """Ring and bbox of chunk (cy, cx) under the affine `transform`
    (a, b, c, d, e, f), clipped to the product."""
    a, b, c, d, e, f = (float(v) for v in transform[:6])
    r0, r1 = cy * chunk_px, min((cy + 1) * chunk_px, height)
    c0, c1 = cx * chunk_px, min((cx + 1) * chunk_px, width)
    corners = [(c0, r1), (c1, r1), (c1, r0), (c0, r0)]
    pts = [(a * col + b * row + c, d * col + e * row + f) for col, row in corners]
    if e > 0:                      # keep the ring counter-clockwise
        pts = pts[::-1]
    ring = pts + [pts[0]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return ring, (min(xs), min(ys), max(xs), max(ys))


def merge_ranges(ranges, gap=0) -> list:
    """Merge (offset, length) ranges whose gap is at most `gap` bytes;
    the gap bytes are fetched with them."""
    out = []
    for o, n in sorted(ranges):
        if out and o - (out[-1][0] + out[-1][1]) <= gap:
            out[-1][1] = max(out[-1][1], o + n - out[-1][0])
        else:
            out.append([o, n])
    return [(o, n) for o, n in out]


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------

class RVQParquetWriter:
    """Writes one product. Window regions can be added in any order,
    from independent encoders; `close` writes the row groups in file
    order, so the file does not depend on the order of `add_windows`
    calls. Every window of the product must be added. A chunk none of
    whose windows has a valid pixel is written with null data columns,
    so records given for those windows are not stored.

    header: the codec, product, rungs and quality entries of the
    `georvq` metadata. transform: the north-up affine (a, b, c, d, e, f)
    of the coded pixel grid. crs: PROJJSON dict, or None if unknown.
    aux: a str (or JSON-able object) stored under `georvq_aux`."""

    def __init__(self, path, header: dict, layout: Layout, product_id: str,
                 transform, height: int, width: int, crs=None, aux=None):
        clash = [k for k in _WRITER_KEYS if k in header]
        if clash:
            raise ValueError(f"header keys {clash} are set by the writer")
        self.path = os.fspath(path)
        self.header = dict(header)
        self.layout = layout
        self.product_id = str(product_id)
        self.transform = [float(v) for v in transform[:6]]
        self.height, self.width = int(height), int(width)
        self.crs = crs
        self.aux = aux if aux is None or isinstance(aux, str) else json.dumps(
            aux, separators=(",", ":"), default=str)
        self.window_grid = layout.window_grid(height, width)
        self.chunk_grids = {k: layout.chunk_grid(k, height, width)
                            for k in layout.groups}
        # packed records of each code column over its kind's padded grid
        self._rec = {}
        for name, col in layout.columns.items():
            g = layout.groups[layout.kind_of[name]]
            ny, nx = self.chunk_grids[g.kind]
            self._rec[name] = np.zeros((ny * g.chunk_windows, nx * g.chunk_windows,
                                        col.record_bytes), np.uint8)
        self._filled = np.zeros(self.window_grid, bool)
        self._valid = None
        self._closed = False

    def _tier(self, name):
        for t in self.layout.tiers:
            if t.name == name and not t.in_base:
                return t
        return None

    def add_windows(self, window_row: int, window_col: int, records: dict,
                    valid=None) -> None:
        """records: {column: (h, w, n_values) ints} for the windows
        [window_row:+h, window_col:+w]; a tier name instead of a column
        takes (h, w, levels, n_values). Every array gives the same
        (h, w), inside the window grid. Columns left out are zero.
        valid: pixel validity of the region clipped to the product,
        None for fully valid."""
        wy, wx = int(window_row), int(window_col)
        cols = {}
        for key, arr in records.items():
            arr = np.asarray(arr)
            t = self._tier(key)
            if t is not None:
                if arr.ndim != 4 or arr.shape[2] != t.levels:
                    raise ValueError(f"tier {key}: expected (h, w, {t.levels}, "
                                     f"values), got {arr.shape}")
                for l, name in enumerate(t.columns):
                    cols[name] = arr[:, :, l]
            elif key in self.layout.columns:
                cols[key] = arr
            else:
                raise ValueError(f"no column or tier named {key!r}")
        if not cols:
            raise ValueError("no records given")
        shapes = {a.shape[:2] for a in cols.values()}
        if len(shapes) != 1:
            raise ValueError(f"records cover different window regions {shapes}")
        h, w = shapes.pop()
        gy, gx = self.window_grid
        if not (0 <= wy and 0 <= wx and wy + h <= gy and wx + w <= gx):
            raise ValueError(f"windows [{wy}:{wy + h}, {wx}:{wx + w}] outside "
                             f"the {gy}x{gx} grid")
        for name, arr in cols.items():
            col = self.layout.columns[name]
            if arr.ndim != 3 or arr.shape[2] != col.n_values:
                raise ValueError(f"{name}: expected (h, w, {col.n_values}), got "
                                 f"{arr.shape}")
            rec = pack(arr.reshape(h * w, -1), col.widths)
            self._rec[name][wy:wy + h, wx:wx + w] = rec.reshape(h, w, -1)
        W = self.layout.window_px
        py, px = wy * W, wx * W
        shape = (min(h * W, self.height - py), min(w * W, self.width - px))
        if valid is not None:
            valid = np.asarray(valid, dtype=bool)
            if valid.shape != shape:
                raise ValueError(f"windows at ({wy}, {wx}): mask {valid.shape}, "
                                 f"expected {shape}")
            if self._valid is None:
                self._valid = np.ones((self.height, self.width), bool)
            self._valid[py:py + shape[0], px:px + shape[1]] = valid
        elif self._valid is not None:
            self._valid[py:py + shape[0], px:px + shape[1]] = True
        self._filled[wy:wy + h, wx:wx + w] = True

    def _live(self) -> np.ndarray:
        """(window rows, window cols) True where a window has a valid
        pixel."""
        gy, gx = self.window_grid
        if self._valid is None:
            return np.ones((gy, gx), bool)
        W = self.layout.window_px
        v = np.zeros((gy * W, gx * W), bool)
        v[:self.height, :self.width] = self._valid
        return v.reshape(gy, W, gx, W).any((1, 3))

    def _rows(self):
        """Row dicts in file order, the chunks with a non-empty mask, and
        the empty chunks of each kind."""
        lay, rows, masked, empty = self.layout, [], [], {}
        live = self._live()
        for g in lay.groups.values():
            ny, nx = self.chunk_grids[g.kind]
            n, p = g.chunk_windows, g.chunk_px
            empty[g.kind] = []
            for i in range(ny):
                for j in range(nx):
                    ring, bb = chunk_footprint(self.transform, i, j, p,
                                               self.height, self.width)
                    row = {"kind": g.kind, "product_id": self.product_id,
                           "chunk_row": i, "chunk_col": j,
                           "geometry": wkb_polygon(ring),
                           "bbox": dict(zip(_BBOX, bb))}
                    if not live[i * n:(i + 1) * n, j * n:(j + 1) * n].any():
                        empty[g.kind].append([i, j])
                        rows.append(row)
                        continue
                    for c in g.columns:
                        if c == "mask":
                            v = b"" if self._valid is None else encode_mask(
                                self._valid[i * p:(i + 1) * p, j * p:(j + 1) * p])
                            if v:
                                masked.append([i, j])
                        else:
                            v = self._rec[c][i * n:(i + 1) * n,
                                             j * n:(j + 1) * n].tobytes()
                        row[c] = v
                    rows.append(row)
        return rows, masked, empty

    def _schema(self, masked, empty):
        bbox_t = pa.struct([(k, pa.float64()) for k in _BBOX])
        fields = [pa.field("kind", pa.string(), nullable=False),
                  pa.field("product_id", pa.string(), nullable=False),
                  pa.field("chunk_row", pa.int32(), nullable=False),
                  pa.field("chunk_col", pa.int32(), nullable=False),
                  pa.field("geometry", pa.binary(), nullable=False),
                  pa.field("bbox", bbox_t, nullable=False)]
        fields += [pa.field(n, pa.binary()) for n in self.layout.column_order]
        _, ext = chunk_footprint(self.transform, 0, 0,
                                 max(self.height, self.width), self.height,
                                 self.width)
        geo_col = {"encoding": "WKB", "geometry_types": ["Polygon"],
                   "crs": self.crs, "bbox": list(ext),
                   "covering": {"bbox": {k: ["bbox", k] for k in _BBOX}}}
        geo = {"version": "1.1.0", "primary_column": "geometry",
               "columns": {"geometry": geo_col}}
        lay = self.layout.to_json()
        first = 0
        for g in lay["groups"]:
            ny, nx = self.chunk_grids[g["kind"]]
            g["chunk_grid"] = [ny, nx]
            g["row_groups"] = [first, ny * nx]
            g["empty_chunks"] = empty[g["kind"]]
            first += ny * nx
        lay["mask"]["chunks"] = masked
        lay["window_grid"] = list(self.window_grid)
        lay["product_px"] = [self.height, self.width]
        header = dict(self.header)
        header.update({"format": FORMAT, "version": f"{MAJOR}.{MINOR}",
                       "status": STATUS, "layout": lay})
        meta = {GEO_KEY: json.dumps(geo, separators=(",", ":")),
                HEADER_KEY: json.dumps(header, separators=(",", ":"))}
        if self.aux is not None:
            meta[AUX_KEY] = self.aux
        return pa.schema(fields), meta

    def close(self) -> None:
        if self._closed:
            return
        if not self._filled.all():
            ys, xs = np.nonzero(~self._filled)
            raise ValueError(f"{len(ys)} windows were not added, first "
                             f"({ys[0]}, {xs[0]})")
        rows, masked, empty = self._rows()
        schema, meta = self._schema(masked, empty)
        names = schema.names
        largest = max([len(r[c]) for r in rows for c in self.layout.column_order
                       if c in r] + [0])
        with pq.ParquetWriter(
                self.path, schema, compression="NONE", use_dictionary=False,
                write_statistics=_STAT_COLUMNS,
                data_page_size=max(1 << 20, 2 * largest),
                store_schema=False, write_page_index=False) as w:
            for row in rows:
                w.write_table(pa.Table.from_pydict(
                    {n: [row.get(n)] for n in names}, schema=schema))
            # store_schema=False drops schema metadata, so the key-value
            # metadata is added to the footer directly.
            w.add_key_value_metadata(meta)
        self._rec = {}
        self._valid = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

class RecordCache:
    """Values of column chunks, keyed by (file key, column, chunk_row,
    chunk_col) with file key = (bitstream id, file identity hash).

    An in-memory LRU bounded by `max_bytes`, backed optionally by files
    under `cache_dir`/<bitstream id>/<file identity hash>/. Readers can
    share one cache."""

    def __init__(self, max_bytes=CACHE_BYTES, cache_dir=None):
        self.max_bytes = int(max_bytes)
        self.dir = None if cache_dir is None else os.path.expanduser(
            os.fspath(cache_dir))
        self._d = OrderedDict()
        self.nbytes = 0
        self._lock = threading.Lock()

    def _path(self, key):
        (bid, fkey), col, r, c = key
        safe = re.sub(r"[^\w.-]", "_", str(bid)) or "_"
        return os.path.join(self.dir, safe, fkey, f"{col}-{r}-{c}.bin")

    def has(self, key) -> bool:
        with self._lock:
            if key in self._d:
                return True
        return self.dir is not None and os.path.exists(self._path(key))

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        if self.dir is None:
            return None
        try:
            with open(self._path(key), "rb") as f:
                v = f.read()
        except FileNotFoundError:
            return None
        self._remember(key, v)
        return v

    def put(self, key, value: bytes) -> None:
        self._remember(key, value)
        if self.dir is not None:
            p = self._path(key)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                f.write(value)
            os.replace(tmp, p)

    def _remember(self, key, value):
        if len(value) > self.max_bytes:
            return
        with self._lock:
            if key in self._d:
                self.nbytes -= len(self._d.pop(key))
            self._d[key] = value
            self.nbytes += len(value)
            while self.nbytes > self.max_bytes:
                _, v = self._d.popitem(last=False)
                self.nbytes -= len(v)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.nbytes = 0


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

class _Sparse(io.RawIOBase):
    """A file of known size holding only the byte ranges fetched so far,
    for pyarrow to parse pages from. The tail stays; other pieces are
    dropped by `clear`. A read outside the fetched ranges raises, so a
    reader cannot fetch more than it planned."""

    def __init__(self, size, tail_offset, tail):
        self.size, self.pos = int(size), 0
        self.tail = (int(tail_offset), tail)
        self.pieces = []

    def covers(self, o, n) -> bool:
        to, td = self.tail
        return to <= o and o + n <= to + len(td)

    def add(self, o, data):
        self.pieces.append((int(o), data))

    def clear(self):
        self.pieces = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, p, whence=0):
        self.pos = {0: 0, 1: self.pos, 2: self.size}[whence] + p
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        for o, d in [self.tail] + self.pieces:
            if o <= self.pos and end <= o + len(d):
                out = d[self.pos - o:end - o]
                self.pos = end
                return out
        raise OSError(f"bytes {self.pos}-{end} were not fetched")

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


class RVQParquetReader:
    """Reads a georvq GeoParquet file from a local path or an http(s)
    URL (HTTP Range requests).

    Opening costs one suffix read of `first_read` bytes, and a second
    when the footer is longer. A read plans the column chunks it needs,
    leaves out those in `cache` or in the fetched tail, merges ranges
    whose gap is at most `gap` bytes, and fetches them, up to
    `concurrency` at a time over HTTP and in turn from a local file.
    `cache` defaults to a RecordCache of `cache_bytes`, on disk under
    `cache_dir` when one is given. `stats` counts requests, bytes,
    values served from the cache and their bytes."""

    def __init__(self, path_or_url, first_read=FIRST_READ, gap=GAP,
                 concurrency=CONCURRENCY, cache=None, cache_bytes=CACHE_BYTES,
                 cache_dir=None):
        self.source = os.fspath(path_or_url)
        self.gap = int(gap)
        self.concurrency = max(1, int(concurrency))
        self._url = self.source.startswith(("http://", "https://"))
        self._fd = None if self._url else os.open(self.source, os.O_RDONLY)
        self._lock = threading.Lock()
        self.stats = {"requests": 0, "bytes": 0, "cache_hits": 0,
                      "cache_bytes": 0}
        self.version_tag = None
        tail, size = self._suffix(int(first_read))
        if len(tail) < 12 or tail[-4:] != b"PAR1":
            raise ValueError(f"{self.source}: not a Parquet file")
        need = struct.unpack("<I", tail[-8:-4])[0] + 8
        if need > len(tail):
            more = self._read((size - need, need - len(tail)))
            tail = more + tail
        self.size = size
        self._buf = _Sparse(size, size - len(tail), tail)
        # The footer is parsed from the fetched bytes alone: pyarrow's own
        # open reads a fixed-size suffix that may extend past them.
        footer = tail[len(tail) - need:]
        md = pq.read_metadata(pa.BufferReader(b"PAR1" + footer))
        self._pf = pq.ParquetFile(self._buf, metadata=md)
        self.metadata = md
        kv = md.metadata or {}
        if HEADER_KEY.encode() not in kv:
            raise ValueError(f"{self.source}: no {HEADER_KEY!r} metadata")
        self.header = json.loads(kv[HEADER_KEY.encode()])
        if self.header.get("format") != FORMAT:
            raise ValueError(f"{self.source}: format {self.header.get('format')!r}")
        major = int(str(self.header["version"]).split(".")[0])
        if major != MAJOR:
            raise ValueError(f"{self.source}: {FORMAT} major version {major}, "
                             f"this reader handles {MAJOR}")
        self.geo = json.loads(kv[GEO_KEY.encode()]) if GEO_KEY.encode() in kv else None
        aux = kv.get(AUX_KEY.encode())
        self.aux = None if aux is None else aux.decode("utf-8")
        lay = self.header["layout"]
        self.layout = Layout.from_json(lay)
        self.window_grid = tuple(int(v) for v in lay["window_grid"])
        self.chunk_grids, self._first, self.empty = {}, {}, {}
        for g in lay["groups"]:
            self.chunk_grids[g["kind"]] = tuple(int(v) for v in g["chunk_grid"])
            self._first[g["kind"]] = int(g["row_groups"][0])
            self.empty[g["kind"]] = {tuple(c) for c in g.get("empty_chunks", [])}
        # windows of empty chunks have no valid pixel, whatever their kind
        self.dead = np.zeros(self.window_grid, bool)
        for kind, chunks in self.empty.items():
            n = self.layout.groups[kind].chunk_windows
            for i, j in chunks:
                self.dead[i * n:(i + 1) * n, j * n:(j + 1) * n] = True
        self.masked = {tuple(c) for c in lay["mask"].get("chunks", [])}
        self.height, self.width = (int(v) for v in lay["product_px"])
        n_rg = sum(a * b for a, b in self.chunk_grids.values())
        if md.num_row_groups != n_rg:
            raise ValueError(f"{self.source}: {md.num_row_groups} row groups, "
                             f"the layout has {n_rg}")
        self._col = {md.schema.column(k).path: k for k in range(md.num_columns)}
        bid = (self.header.get("codec") or {}).get("bitstream_id", "")
        self.file_key = (str(bid), hashlib.sha1(json.dumps(
            self.identity, default=str).encode()).hexdigest()[:20])
        self.cache = cache if cache is not None else RecordCache(cache_bytes,
                                                                 cache_dir)

    # ---- io ----

    @property
    def identity(self) -> list:
        """What distinguishes this file's bytes from another version: the
        URL with its ETag or Last-Modified, or the path, size and mtime."""
        if self._url:
            return [self.source, self.version_tag]
        st = os.fstat(self._fd)
        return [os.path.realpath(self.source), st.st_size, st.st_mtime_ns]

    def _count(self, n):
        with self._lock:
            self.stats["requests"] += 1
            self.stats["bytes"] += n

    @staticmethod
    def _tag(resp):
        return resp.headers.get("ETag") or resp.headers.get("Last-Modified")

    def _suffix(self, n):
        if self._url:
            req = urllib.request.Request(self.source,
                                         headers={"Range": f"bytes=-{n}"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                self.version_tag = self._tag(resp)
                if resp.status == 206:
                    size = int(resp.headers["Content-Range"].rsplit("/", 1)[1])
                elif resp.status == 200:
                    size = len(data)
                    data = data[-n:]
                else:
                    raise OSError(f"{self.source}: suffix request answered "
                                  f"{resp.status}")
        else:
            size = os.fstat(self._fd).st_size
            data = self._pread(max(0, size - n), min(n, size))
        self._count(len(data))
        return data, size

    def _pread(self, offset, length):
        parts, got = [], 0
        while got < length:
            b = os.pread(self._fd, length - got, offset + got)
            if not b:
                break
            parts.append(b)
            got += len(b)
        return b"".join(parts)

    def _read(self, rng) -> bytes:
        offset, length = rng
        if length <= 0:
            return b""
        if self._url:
            req = urllib.request.Request(
                self.source, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
            with urllib.request.urlopen(req) as resp:
                if resp.status != 206:
                    raise OSError(f"{self.source}: range request answered "
                                  f"{resp.status}, expected 206")
                tag = self._tag(resp)
                if self.version_tag is not None and tag != self.version_tag:
                    raise OSError(f"{self.source}: changed since it was opened "
                                  f"({self.version_tag} -> {tag})")
                data = resp.read()
        else:
            data = self._pread(offset, length)
        if len(data) != length:
            raise OSError(f"{self.source}: short read at {offset}")
        self._count(len(data))
        return data

    def _fetch(self, ranges) -> list:
        if self._url and self.concurrency > 1 and len(ranges) > 1:
            with ThreadPoolExecutor(min(self.concurrency, len(ranges))) as ex:
                return list(ex.map(self._read, ranges))
        return [self._read(r) for r in ranges]

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- metadata ----

    def row_group(self, kind: str, chunk_row: int, chunk_col: int) -> int:
        ny, nx = self.chunk_grids[kind]
        if not (0 <= chunk_row < ny and 0 <= chunk_col < nx):
            raise ValueError(f"{kind} chunk ({chunk_row}, {chunk_col}) outside "
                             f"the {ny}x{nx} grid")
        return self._first[kind] + int(chunk_row) * nx + int(chunk_col)

    def chunk_range(self, column, chunk_row, chunk_col) -> tuple:
        """(offset, length) of one column chunk of its kind's chunk."""
        rg = self.row_group(self.layout.kind_of[column], chunk_row, chunk_col)
        cc = self.metadata.row_group(rg).column(self._col[column])
        o = (cc.dictionary_page_offset if cc.has_dictionary_page
             else cc.data_page_offset)
        return int(o), int(cc.total_compressed_size)

    def column_bytes(self) -> dict:
        """Total column-chunk bytes per top-level column."""
        out = {}
        md = self.metadata
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            for k in range(md.num_columns):
                name = md.schema.column(k).path.split(".")[0]
                out[name] = out.get(name, 0) + rg.column(k).total_compressed_size
        return out

    def kind_bytes(self) -> dict:
        """Total bytes of each kind's row groups, every column included."""
        md, out = self.metadata, {}
        for kind, (ny, nx) in self.chunk_grids.items():
            first = self._first[kind]
            out[kind] = sum(md.row_group(first + i).column(k).total_compressed_size
                            for i in range(ny * nx) for k in range(md.num_columns))
        return out

    def columns_for(self, depths=None, base=True, mask=True,
                    quality=False) -> list:
        """Columns read for a depth tuple (one entry per tier; a tier in
        base is read whole whatever its entry). None reads every level."""
        tiers = self.layout.tiers
        if depths is None:
            depths = [t.levels for t in tiers]
        if len(depths) != len(tiers):
            raise ValueError(f"depths {tuple(depths)} for {len(tiers)} tiers")
        cols = (["base"] if base else []) + (["mask"] if mask else [])
        for t, L in zip(tiers, depths):
            if t.in_base:
                continue
            if not 0 <= int(L) <= t.levels:
                raise ValueError(f"tier {t.name}: depth {L} not in 0..{t.levels}")
            cols += list(t.columns[:int(L)])
        if quality:
            if self.layout.quality is None:
                raise ValueError(f"{self.source}: no quality column")
            cols.append("quality")
        return cols

    def chunks_of(self, kind, wy0, wy1, wx0, wx1) -> list:
        """Chunks of `kind` holding windows [wy0:wy1, wx0:wx1]."""
        rows, cols = self.window_grid
        if not (0 <= wy0 < wy1 <= rows and 0 <= wx0 < wx1 <= cols):
            raise ValueError(f"windows [{wy0}:{wy1}, {wx0}:{wx1}] outside "
                             f"the {rows}x{cols} grid")
        n = self.layout.groups[kind].chunk_windows
        return [(i, j) for i in range(wy0 // n, (wy1 - 1) // n + 1)
                for j in range(wx0 // n, (wx1 - 1) // n + 1)]

    def skipped(self, kind, i, j, wy0, wy1, wx0, wx1) -> bool:
        """True when chunk (i, j) of `kind` is empty, or when none of its
        windows inside [wy0:wy1, wx0:wx1] has a valid pixel as known from
        the empty chunks of any kind. A read does not fetch it."""
        if (i, j) in self.empty[kind]:
            return True
        n = self.layout.groups[kind].chunk_windows
        return bool(self.dead[max(wy0, i * n):min(wy1, (i + 1) * n),
                              max(wx0, j * n):min(wx1, (j + 1) * n)].all())

    def needs(self, wy0, wy1, wx0, wx1, columns) -> list:
        """(column, chunk_row, chunk_col) values that a read of `columns`
        over the windows needs. Masks known to be empty, and chunks that
        `skipped` rules out, are left out."""
        out = []
        for c in columns:
            kind = self.layout.kind_of[c]
            for ch in self.chunks_of(kind, wy0, wy1, wx0, wx1):
                if c == "mask" and ch not in self.masked:
                    continue
                if self.skipped(kind, *ch, wy0, wy1, wx0, wx1):
                    continue
                out.append((c,) + ch)
        return out

    def _key(self, need):
        return (self.file_key,) + tuple(need)

    def plan(self, wy0, wy1, wx0, wx1, depths=None, base=True, mask=True,
             quality=False) -> list:
        """Merged (offset, length) requests a read would make now: the
        needed column chunks not in the cache or the fetched tail."""
        cols = self.columns_for(depths, base, mask, quality)
        return self._ranges([n for n in self.needs(wy0, wy1, wx0, wx1, cols)
                             if not self.cache.has(self._key(n))])

    def _ranges(self, needs):
        ranges = []
        for n in needs:
            o, ln = self.chunk_range(*n)
            if not self._buf.covers(o, ln):
                ranges.append((o, ln))
        return merge_ranges(ranges, self.gap)

    # ---- values ----

    def values(self, needs) -> dict:
        """{(column, chunk_row, chunk_col): value bytes} for `needs`, from
        the cache where held and otherwise in merged requests."""
        out, missing = {}, []
        for n in needs:
            v = self.cache.get(self._key(n))
            if v is None:
                missing.append(n)
            else:
                out[n] = v
                self.stats["cache_hits"] += 1
                self.stats["cache_bytes"] += len(v)
        if not missing:
            return out
        ranges = self._ranges(missing)
        for r, data in zip(ranges, self._fetch(ranges)):
            self._buf.add(r[0], data)
        try:
            by_rg = OrderedDict()
            for n in missing:
                rg = self.row_group(self.layout.kind_of[n[0]], n[1], n[2])
                by_rg.setdefault(rg, []).append(n)
            for rg, ns in by_rg.items():
                tb = self._pf.read_row_group(rg, columns=[n[0] for n in ns],
                                             use_threads=False)
                for n in ns:
                    v = tb.column(n[0])[0].as_py()
                    if v is None:
                        raise ValueError(f"{self.source}: {n} is null")
                    out[n] = v
                    self.cache.put(self._key(n), v)
        finally:
            self._buf.clear()
        return out

    def read(self, wy0, wy1, wx0, wx1, depths=None, base=True, mask=True,
             quality=False) -> dict:
        """Records of windows [wy0:wy1, wx0:wx1].

        Returns {"base": (h, w, n), tier name: (h, w, L, n) for each
        tier outside base read at L > 0, "quality": (h, w, n), "valid":
        (h * W, w * W) pixel validity, False beyond the product}, with
        only the parts asked for. Windows without a valid pixel that were
        not fetched have zero records."""
        if depths is None:
            depths = [t.levels for t in self.layout.tiers]
        cols = self.columns_for(depths, base, mask, quality)
        vals = self.values(self.needs(wy0, wy1, wx0, wx1, cols))
        gh, gw = wy1 - wy0, wx1 - wx0
        out = {}

        def records(name):
            col = self.layout.columns[name]
            kind = self.layout.kind_of[name]
            n = self.layout.groups[kind].chunk_windows
            arr = np.zeros((gh, gw, col.n_values), np.int64)
            for i, j in self.chunks_of(kind, wy0, wy1, wx0, wx1):
                if (name, i, j) not in vals:
                    continue
                y0, x0 = i * n, j * n
                sy0, sy1 = max(wy0, y0), min(wy1, y0 + n)
                sx0, sx1 = max(wx0, x0), min(wx1, x0 + n)
                raw = np.frombuffer(vals[(name, i, j)], np.uint8).reshape(
                    n, n, col.record_bytes)[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
                arr[sy0 - wy0:sy1 - wy0, sx0 - wx0:sx1 - wx0] = unpack(
                    raw.reshape(-1, col.record_bytes), col.widths).reshape(
                    sy1 - sy0, sx1 - sx0, -1)
            return arr

        if base:
            out["base"] = records("base")
        for t, L in zip(self.layout.tiers, depths):
            if t.in_base or int(L) == 0:
                continue
            out[t.name] = np.stack([records(c) for c in t.columns[:int(L)]], 2)
        if quality:
            out["quality"] = records("quality")
        if mask:
            W = self.layout.window_px
            kind = self.layout.mask_kind
            p = self.layout.groups[kind].chunk_px
            valid = np.zeros((gh * W, gw * W), bool)
            y0, x0 = wy0 * W, wx0 * W
            for i, j in self.chunks_of(kind, wy0, wy1, wx0, wx1):
                py0, px0 = i * p, j * p
                bh, bw = min(p, self.height - py0), min(p, self.width - px0)
                sy0, sy1 = max(y0, py0), min(y0 + gh * W, py0 + bh)
                sx0, sx1 = max(x0, px0), min(x0 + gw * W, px0 + bw)
                if sy1 <= sy0 or sx1 <= sx0:
                    continue
                if self.skipped(kind, i, j, wy0, wy1, wx0, wx1):
                    continue
                if (i, j) in self.masked:
                    blk = decode_mask(vals[("mask", i, j)], (bh, bw))
                    valid[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                        blk[sy0 - py0:sy1 - py0, sx0 - px0:sx1 - px0]
                else:
                    valid[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = True
            out["valid"] = valid
        return out

    def read_mask(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Validity (True = valid) of pixels [y0:y1, x0:x1]."""
        if not (0 <= y0 < y1 <= self.height and 0 <= x0 < x1 <= self.width):
            raise ValueError(f"pixels [{y0}:{y1}, {x0}:{x1}] outside the "
                             f"{self.height}x{self.width} product")
        W = self.layout.window_px
        wy0, wx0 = y0 // W, x0 // W
        v = self.read(wy0, -(-y1 // W), wx0, -(-x1 // W), base=False,
                      depths=[0] * len(self.layout.tiers))["valid"]
        return v[y0 - wy0 * W:y1 - wy0 * W, x0 - wx0 * W:x1 - wx0 * W]
