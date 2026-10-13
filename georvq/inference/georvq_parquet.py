"""Reader and writer for georvq GeoParquet files (docs/GEORVQ_PARQUET.md).

A file holds one source product. Each row, and each row group, is one
square spatial block of the product: its footprint as a WKB polygon with
a GeoParquet 1.1 bbox covering, and its windows' fixed-width code
records as binary values, one column per tier level. The base column
holds the windows' base records (for example a transmitted mean and the
coarsest tier's levels), `mask` holds the block's pixel validity, and an
optional `quality` column holds per-window quality bytes. The codec
header is JSON in the file's key-value metadata under `georvq`.

Code columns are written uncompressed and without dictionary encoding,
so each column chunk is one data page around one value and its byte
range follows from the footer. The reader opens a file with one suffix
read, then fetches the column chunks a block range needs at a given
depth, merging adjacent ranges, and small gaps between them, into single
requests. The file is a standard Parquet file, so pyarrow, DuckDB or
GDAL can read it without this module.

This module is source-agnostic and depends only on numpy, pyarrow and
the standard library, so a reader can be installed without torch.
"""
from __future__ import annotations

import io
import json
import os
import struct
import urllib.request
import zlib
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from georvq.inference.bitpack import pack, record_bytes, unpack

FORMAT = "georvq-parquet"
MAJOR = 1
MINOR = 0
STATUS = "placeholder"
SUFFIX = ".georvq.parquet"
FIRST_READ = 256 * 1024
GAP = 16 * 1024
MASK_CODEC = "packbits-zlib"
HEADER_KEY = "georvq"
AUX_KEY = "georvq_aux"
GEO_KEY = "geo"
_BBOX = ("xmin", "ymin", "xmax", "ymax")
_STAT_COLUMNS = ["product_id", "block_row", "block_col"] + [f"bbox.{k}" for k in _BBOX]
_WRITER_KEYS = ("format", "version", "status", "layout")


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


class Layout:
    """Columns and block geometry of a file.

    window_px: window side in pixels; block_px: block side in pixels, a
    multiple of window_px. base: fields of the base record. tiers: list
    of (name, levels, fields), fields None for a tier carried in base
    (only the first tier can be). quality: fields of the quality record,
    or None for no quality column."""

    def __init__(self, window_px, block_px, base, tiers, quality=None):
        self.window_px, self.block_px = int(window_px), int(block_px)
        if self.block_px % self.window_px:
            raise ValueError(f"block_px {block_px} is not a multiple of "
                             f"window_px {window_px}")
        self.block_windows = self.block_px // self.window_px
        self.base = Column("base", base)
        self.tiers, self.level_columns = [], []
        for i, (name, levels, fields) in enumerate(tiers):
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

    @property
    def column_order(self) -> list:
        """Code columns in file order, `mask` after `base`."""
        return (["base", "mask"] + [c.name for c in self.level_columns]
                + (["quality"] if self.quality else []))

    def block_grid(self, height, width) -> tuple:
        b = self.block_px
        return (-(-int(height) // b), -(-int(width) // b))

    def window_grid(self, height, width) -> tuple:
        w = self.window_px
        return (-(-int(height) // w), -(-int(width) // w))

    def to_json(self) -> dict:
        def col(c):
            return {"name": c.name, "record_bytes": c.record_bytes,
                    "fields": [{"name": n, "count": k, "bits": b}
                               for n, k, b in c.fields]}
        return {"window_px": self.window_px, "block_px": self.block_px,
                "block_windows": [self.block_windows] * 2,
                "block_order": "row-major", "window_order": "row-major",
                "mask": {"column": "mask", "codec": MASK_CODEC},
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
        return cls(d["window_px"], d["block_px"], fields("base"), tiers,
                   fields("quality") if "quality" in cols else None)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def encode_mask(valid) -> bytes:
    """Mask value of one block: b"" when every pixel is valid, otherwise
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


def block_footprint(transform, by, bx, block_px, height, width):
    """Ring and bbox of block (by, bx) under the affine `transform`
    (a, b, c, d, e, f), clipped to the product."""
    a, b, c, d, e, f = (float(v) for v in transform[:6])
    r0, r1 = by * block_px, min((by + 1) * block_px, height)
    c0, c1 = bx * block_px, min((bx + 1) * block_px, width)
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
    """Writes one product. Blocks can be added in any order, from
    independent block encoders; `close` writes one row group per block
    in row-major block order, so the file does not depend on the order
    of `add_block` calls. Every block of the grid must be added.

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
        self.block_grid = layout.block_grid(height, width)
        self.window_grid = layout.window_grid(height, width)
        self._blocks = {}
        self._closed = False

    def _tier(self, name):
        for t in self.layout.tiers:
            if t.name == name and not t.in_base:
                return t
        return None

    def add_block(self, block_row: int, block_col: int, records: dict,
                  valid=None) -> None:
        """records: {column: (h, w, n_values) ints} for the block's
        windows, h, w <= block_windows (an edge block may give only the
        windows inside the product). A tier name instead of a column
        takes (h, w, levels, n_values). Columns left out are zero.
        valid: the block's pixel validity clipped to the product, None
        for fully valid."""
        by, bx = int(block_row), int(block_col)
        nby, nbx = self.block_grid
        if not (0 <= by < nby and 0 <= bx < nbx):
            raise ValueError(f"block ({by}, {bx}) outside the {nby}x{nbx} grid")
        n = self.layout.block_windows
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
        row = {}
        for name, col in self.layout.columns.items():
            full = np.zeros((n, n, col.n_values), dtype=np.int64)
            if name in cols:
                arr = cols[name]
                if arr.ndim != 3 or arr.shape[2] != col.n_values \
                        or arr.shape[0] > n or arr.shape[1] > n:
                    raise ValueError(f"{name}: expected (<= {n}, <= {n}, "
                                     f"{col.n_values}), got {arr.shape}")
                full[:arr.shape[0], :arr.shape[1]] = arr
            row[name] = pack(full.reshape(n * n, -1), col.widths).tobytes()
        b = self.layout.block_px
        shape = (min(b, self.height - by * b), min(b, self.width - bx * b))
        if valid is None:
            row["mask"] = b""
        else:
            valid = np.asarray(valid, dtype=bool)
            if valid.shape != shape:
                raise ValueError(f"block ({by}, {bx}) mask {valid.shape}, "
                                 f"expected {shape}")
            row["mask"] = encode_mask(valid)
        self._blocks[(by, bx)] = row

    def _schema(self, bboxes):
        bbox_t = pa.struct([(k, pa.float64()) for k in _BBOX])
        fields = [pa.field("product_id", pa.string(), nullable=False),
                  pa.field("block_row", pa.int32(), nullable=False),
                  pa.field("block_col", pa.int32(), nullable=False),
                  pa.field("geometry", pa.binary(), nullable=False),
                  pa.field("bbox", bbox_t, nullable=False)]
        fields += [pa.field(n, pa.binary(), nullable=False)
                   for n in self.layout.column_order]
        geo_col = {"encoding": "WKB", "geometry_types": ["Polygon"],
                   "crs": self.crs,
                   "bbox": [min(b[0] for b in bboxes), min(b[1] for b in bboxes),
                            max(b[2] for b in bboxes), max(b[3] for b in bboxes)],
                   "covering": {"bbox": {k: ["bbox", k] for k in _BBOX}}}
        geo = {"version": "1.1.0", "primary_column": "geometry",
               "columns": {"geometry": geo_col}}
        header = dict(self.header)
        header.update({
            "format": FORMAT, "version": f"{MAJOR}.{MINOR}", "status": STATUS,
            "layout": dict(self.layout.to_json(),
                           block_grid=list(self.block_grid),
                           window_grid=list(self.window_grid))})
        meta = {GEO_KEY: json.dumps(geo, separators=(",", ":")),
                HEADER_KEY: json.dumps(header, separators=(",", ":"))}
        if self.aux is not None:
            meta[AUX_KEY] = self.aux
        return pa.schema(fields), meta

    def close(self) -> None:
        if self._closed:
            return
        nby, nbx = self.block_grid
        missing = [(i, j) for i in range(nby) for j in range(nbx)
                   if (i, j) not in self._blocks]
        if missing:
            raise ValueError(f"{len(missing)} blocks were not added, first "
                             f"{missing[0]}")
        foot = {(i, j): block_footprint(self.transform, i, j,
                                        self.layout.block_px, self.height,
                                        self.width)
                for i in range(nby) for j in range(nbx)}
        schema, meta = self._schema([bb for _, bb in foot.values()])
        largest = max(len(v) for r in self._blocks.values() for v in r.values())
        with pq.ParquetWriter(
                self.path, schema, compression="NONE", use_dictionary=False,
                write_statistics=_STAT_COLUMNS,
                data_page_size=max(1 << 20, 2 * largest),
                store_schema=False, write_page_index=False) as w:
            for i in range(nby):
                for j in range(nbx):
                    ring, bb = foot[(i, j)]
                    row = {"product_id": [self.product_id], "block_row": [i],
                           "block_col": [j], "geometry": [wkb_polygon(ring)],
                           "bbox": [dict(zip(_BBOX, bb))]}
                    row.update({n: [v] for n, v in self._blocks[(i, j)].items()})
                    w.write_table(pa.Table.from_pydict(row, schema=schema))
            # store_schema=False drops schema metadata, so the key-value
            # metadata is added to the footer directly.
            w.add_key_value_metadata(meta)
        self._blocks = {}
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

class _Sparse(io.RawIOBase):
    """A file of known size holding only the byte ranges fetched so far.
    The tail stays; other pieces are dropped by `clear`. A read outside
    the fetched ranges raises, so a reader cannot fetch more than it
    asked for."""

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
    URL (HTTP Range requests). `stats` counts requests and bytes.

    Opening costs one suffix read of `first_read` bytes, and a second
    when the footer is longer. Reads merge column-chunk ranges whose gap
    is at most `gap` bytes."""

    def __init__(self, path_or_url, first_read=FIRST_READ, gap=GAP):
        self.source = os.fspath(path_or_url)
        self.gap = int(gap)
        self._url = self.source.startswith(("http://", "https://"))
        self._file = None if self._url else open(self.source, "rb")
        self.stats = {"requests": 0, "bytes": 0}
        tail, size = self._suffix(int(first_read))
        if len(tail) < 12 or tail[-4:] != b"PAR1":
            raise ValueError(f"{self.source}: not a Parquet file")
        need = struct.unpack("<I", tail[-8:-4])[0] + 8
        if need > len(tail):
            more = self._read(size - need, need - len(tail))
            tail = more + tail
        self.size = size
        self._buf = _Sparse(size, size - len(tail), tail)
        # The footer is parsed from the fetched bytes alone: pyarrow's own
        # open reads a fixed-size suffix that may extend past them.
        footer = tail[len(tail) - need:]
        md = pq.read_metadata(pa.BufferReader(b"PAR1" + footer))
        self._pf = pq.ParquetFile(self._buf, metadata=md)
        self.metadata = md
        kv = self.metadata.metadata or {}
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
        self.block_grid = tuple(int(v) for v in lay["block_grid"])
        self.window_grid = tuple(int(v) for v in lay["window_grid"])
        p = self.header.get("product", {})
        self.height = int(p.get("height", self.window_grid[0] * self.layout.window_px))
        self.width = int(p.get("width", self.window_grid[1] * self.layout.window_px))
        md = self.metadata
        if md.num_row_groups != self.block_grid[0] * self.block_grid[1]:
            raise ValueError(f"{self.source}: {md.num_row_groups} row groups for "
                             f"a {self.block_grid} block grid")
        self._col = {md.schema.column(k).path: k for k in range(md.num_columns)}

    # ---- io ----

    def _suffix(self, n):
        if self._url:
            req = urllib.request.Request(self.source,
                                         headers={"Range": f"bytes=-{n}"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                if resp.status == 206:
                    size = int(resp.headers["Content-Range"].rsplit("/", 1)[1])
                elif resp.status == 200:
                    size = len(data)
                    data = data[-n:]
                else:
                    raise OSError(f"{self.source}: suffix request answered "
                                  f"{resp.status}")
        else:
            size = os.fstat(self._file.fileno()).st_size
            self._file.seek(max(0, size - n))
            data = self._file.read(n)
        self.stats["requests"] += 1
        self.stats["bytes"] += len(data)
        return data, size

    def _read(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        if self._url:
            req = urllib.request.Request(
                self.source, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
            with urllib.request.urlopen(req) as resp:
                if resp.status != 206:
                    raise OSError(f"{self.source}: range request answered "
                                  f"{resp.status}, expected 206")
                data = resp.read()
        else:
            self._file.seek(offset)
            data = self._file.read(length)
        if len(data) != length:
            raise OSError(f"{self.source}: short read at {offset}")
        self.stats["requests"] += 1
        self.stats["bytes"] += len(data)
        return data

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- metadata ----

    def row_group(self, block_row: int, block_col: int) -> int:
        return int(block_row) * self.block_grid[1] + int(block_col)

    def chunk_range(self, block_row, block_col, column) -> tuple:
        """(offset, length) of one column chunk."""
        cc = self.metadata.row_group(self.row_group(block_row, block_col)) \
            .column(self._col[column])
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

    def blocks_of(self, wy0, wy1, wx0, wx1) -> list:
        rows, cols = self.window_grid
        if not (0 <= wy0 < wy1 <= rows and 0 <= wx0 < wx1 <= cols):
            raise ValueError(f"windows [{wy0}:{wy1}, {wx0}:{wx1}] outside "
                             f"the {rows}x{cols} grid")
        n = self.layout.block_windows
        return [(i, j) for i in range(wy0 // n, (wy1 - 1) // n + 1)
                for j in range(wx0 // n, (wx1 - 1) // n + 1)]

    def byte_ranges(self, blocks, columns) -> list:
        """Merged (offset, length) requests that fetch `columns` of
        `blocks`, leaving out what the open already holds."""
        ranges = []
        for by, bx in blocks:
            for c in columns:
                o, n = self.chunk_range(by, bx, c)
                if not self._buf.covers(o, n):
                    ranges.append((o, n))
        return merge_ranges(ranges, self.gap)

    # ---- values ----

    def fetch(self, blocks, columns) -> dict:
        """{column: [value bytes per block]} for `blocks` (a list of
        (block_row, block_col)), fetched in merged requests."""
        blocks = list(blocks)
        for o, n in self.byte_ranges(blocks, columns):
            self._buf.add(o, self._read(o, n))
        try:
            rgs = [self.row_group(*b) for b in blocks]
            tb = self._pf.read_row_groups(rgs, columns=list(columns),
                                          use_threads=False)
            return {c: tb.column(c).to_pylist() for c in columns}
        finally:
            self._buf.clear()

    def read(self, wy0, wy1, wx0, wx1, depths=None, base=True, mask=True,
             quality=False) -> dict:
        """Records of windows [wy0:wy1, wx0:wx1] in one fetch.

        Returns {"base": (h, w, n), tier name: (h, w, L, n) for each
        tier outside base read at L > 0, "quality": (h, w, n), "valid":
        (h * W, w * W) pixel validity, False beyond the product}, with
        only the parts asked for."""
        cols = self.columns_for(depths, base, mask, quality)
        blocks = self.blocks_of(wy0, wy1, wx0, wx1)
        vals = self.fetch(blocks, cols)
        n = self.layout.block_windows
        gh, gw = wy1 - wy0, wx1 - wx0
        out = {}

        def place(arr, by, bx, block):
            y0, x0 = by * n, bx * n
            sy0, sy1 = max(wy0, y0), min(wy1, y0 + n)
            sx0, sx1 = max(wx0, x0), min(wx1, x0 + n)
            arr[sy0 - wy0:sy1 - wy0, sx0 - wx0:sx1 - wx0, ...] = \
                block[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, ...]

        def records(name):
            col = self.layout.columns[name]
            arr = np.zeros((gh, gw, col.n_values), np.int64)
            for (by, bx), v in zip(blocks, vals[name]):
                raw = np.frombuffer(v, np.uint8).reshape(n * n, col.record_bytes)
                place(arr, by, bx, unpack(raw, col.widths).reshape(n, n, -1))
            return arr

        if base:
            out["base"] = records("base")
        for t, L in zip(self.layout.tiers, depths or [t.levels for t in self.layout.tiers]):
            if t.in_base or int(L) == 0:
                continue
            out[t.name] = np.stack([records(c) for c in t.columns[:int(L)]], 2)
        if quality:
            out["quality"] = records("quality")
        if mask:
            W, b = self.layout.window_px, self.layout.block_px
            valid = np.zeros((gh * W, gw * W), bool)
            for (by, bx), v in zip(blocks, vals["mask"]):
                py0, px0 = by * b, bx * b
                bh, bw = min(b, self.height - py0), min(b, self.width - px0)
                blk = decode_mask(v, (bh, bw))
                y0, x0 = wy0 * W, wx0 * W
                sy0, sy1 = max(y0, py0), min(y0 + gh * W, py0 + bh)
                sx0, sx1 = max(x0, px0), min(x0 + gw * W, px0 + bw)
                valid[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                    blk[sy0 - py0:sy1 - py0, sx0 - px0:sx1 - px0]
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
