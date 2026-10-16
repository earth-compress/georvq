"""Coded records of a georvq product, stored as a Zarr v3 group.

A product's windows each have fixed-width code records: a `base` record
(for example a transmitted mean and the coarsest tier's levels), one
record per level of each finer tier, and optionally a quality record.
Records are packed MSB-first and padded to a byte (georvq.inference.
bitpack), and grouped into square spatial chunks with a chunk size per
kind. A store `{stem}.georvq.zarr` holds them as uint8 arrays over the
window grid, written and read by zarr-python:

    zarr.json            group: the codec header under `georvq`, source
                         metadata under `georvq_aux`, proj: and spatial:
                         attributes, and zarr-python's consolidated
                         metadata
    base   (wy, wx, base bytes)     one chunk per base chunk
    <tier> (wy, wx, levels, bytes)  sharding_indexed, one shard for the
                                    array, inner chunks of one level of
                                    one tier chunk
    mask   (H, W) bool              one chunk per mask-kind chunk, zstd;
                                    chunks with every pixel valid omitted
    quality/<decoder_id> (wy, wx, quality bytes)  chunked as base

Code arrays have no compression codec, so each chunk holds the packed
records of its windows. A chunk none of whose windows has a valid pixel
is empty: the header lists it, its records are zero, and the tiers do
not store it. Quality values measure one decoder's error at its rungs,
so each array under `quality/` is named by the `decoder_id` it was
measured with, and its attributes (`georvq_quality`) carry that
decoder's ids, release, rungs in record order, record fields and the
definition of the error. The `quality` group's attributes list the
decoders it holds; a later decoder of the same bitstream adds its array
and its entry there.

`RecordWriter` writes a store; `RecordReader` reads records and masks
of window regions from an open zarr.Group, one zarr-python read per
array over the bounding box of the chunks a read needs, and keeps the
values per (column, chunk) in a cache. `open_store` opens a path or URL.
Reading only needs numpy and zarr-python 3.3 or later (earlier versions
fetch the inner chunks of a shard one at a time).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from georvq.inference.bitpack import pack, record_bytes, unpack

FORMAT = "georvq-zarr"
MAJOR = 0
MINOR = 4
STATUS = "prototype"
SUFFIX = ".georvq.zarr"
HEADER_KEY = "georvq"
AUX_KEY = "georvq_aux"
QUALITY_GROUP = "quality"
QUALITY_KEY = "georvq_quality"
CACHE_BYTES = 512 * 1024 * 1024
MASK_ZSTD_LEVEL = 19
_WRITER_KEYS = ("format", "version", "status", "layout")
_RESERVED = ("base", "mask", "quality")
_TIER_NAME = re.compile(r"[A-Za-z0-9_]+")

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


def tier_names(n_tiers: int) -> list:
    """Column-name stems of the tiers, coarsest first."""
    if n_tiers == 3:
        return ["coarse", "mid", "fine"]
    return [f"tier{i}" for i in range(n_tiers)]


def output_name(source) -> str:
    """`{source stem}.georvq.zarr` for a source product path."""
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
    """One kind of chunk: its side in pixels and windows, and its data
    columns."""
    kind: str
    chunk_px: int
    chunk_windows: int
    columns: tuple


class Layout:
    """Columns and chunk geometry of a product.

    window_px: window side in pixels. chunk_px: chunk side in pixels, a
    multiple of window_px, either one int for every kind or a dict with
    an entry for `base` and for each tier outside base; quality chunks
    follow base. base: fields of the base record. tiers: list of (name,
    levels, fields), fields None for a tier carried in base (only the
    first tier can be). quality: fields of the quality record, or None
    for no quality column. mask_kind: the kind whose chunks the pixel
    validity mask is chunked by, `base` or a tier outside base.

    Kinds are `base`, each tier outside base, and `quality`."""

    def __init__(self, window_px, chunk_px, base, tiers, quality=None,
                 mask_kind="base"):
        self.window_px = W = int(window_px)
        self.base = Column("base", base)
        self.tiers, self.level_columns = [], []
        for i, (name, levels, fields) in enumerate(tiers):
            if not isinstance(name, str) or not _TIER_NAME.fullmatch(name):
                raise ValueError(f"tier name {name!r} is not made of "
                                 f"[A-Za-z0-9_]")
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
        """Data columns, base first."""
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
                "window_order": "row-major",
                "groups": [{"kind": g.kind, "chunk_px": g.chunk_px,
                            "chunk_windows": g.chunk_windows,
                            "columns": list(g.columns)}
                           for g in self.groups.values()],
                "mask": {"column": "mask", "kind": self.mask_kind},
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


def check_header(header, source="") -> None:
    """Refuse a header of another format or major version."""
    if header.get("format") != FORMAT:
        raise ValueError(f"{source}: format {header.get('format')!r}, not "
                         f"{FORMAT!r}")
    major = int(str(header["version"]).split(".")[0])
    if major != MAJOR:
        raise ValueError(f"{source}: {FORMAT} major version {major}, "
                         f"this reader handles {MAJOR}")


# --------------------------------------------------------------------------
# tracing and cache
# --------------------------------------------------------------------------

class Trace:
    """Timed events of a read, for profiling: each a dict with `kind`
    ("get" for a zarr-python read, "cpu" for local work, "decode" for
    codec stages), `label`, `t0` and `t1` (time.perf_counter seconds) and
    `bytes`. Reads made in parallel are recorded from their own threads,
    so their overlap is kept. Pass one as `trace=` to RecordReader (or
    georvq.ae.open); tracing is off without one."""

    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def add(self, kind, label, t0, t1, nbytes=0):
        with self._lock:
            self.events.append({"kind": kind, "label": label, "t0": t0,
                                "t1": t1, "bytes": int(nbytes)})

    @contextlib.contextmanager
    def span(self, kind, label, nbytes=0, sync=None):
        """Records the enclosed block. `sync`, when given, is called
        before the end time is taken (e.g. torch.cuda.synchronize), so
        asynchronous device work is counted in the block that issued it."""
        t0 = time.perf_counter()
        yield
        if sync is not None:
            sync()
        self.add(kind, label, t0, time.perf_counter(), nbytes)


def span(trace, kind, label, nbytes=0, sync=None):
    """`trace.span(...)`, or a no-op context when `trace` is None."""
    if trace is None:
        return contextlib.nullcontext()
    return trace.span(kind, label, nbytes, sync)


class RecordCache:
    """Values of (column, chunk) records, keyed by (store key, column,
    chunk_row, chunk_col): an in-memory LRU bounded by `max_bytes`.
    Readers can share one."""

    def __init__(self, max_bytes=CACHE_BYTES):
        self.max_bytes = int(max_bytes)
        self._d = OrderedDict()
        self.nbytes = 0
        self._lock = threading.Lock()

    def has(self, key) -> bool:
        with self._lock:
            return key in self._d

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value: bytes) -> None:
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
    return {"spatial:dimensions": list(dims), "spatial:shape": list(shape),
            "spatial:transform": [float(v) for v in transform],
            "spatial:bbox": _bbox(transform, *shape),
            "spatial:registration": "pixel"}


def _crs_attrs(product, projjson):
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
    if projjson:
        out["proj:projjson"] = projjson
    return out


def quality_info(header) -> dict:
    """The `georvq_quality` attributes of the quality array of the
    header's decoder: its ids and release, its rungs in record order,
    the record fields and the definition of the error (the header's
    `quality`)."""
    codec = header.get("codec") or {}
    did = codec.get("decoder_id")
    if not did:
        raise ValueError("the header names no codec.decoder_id to tie its "
                         "quality values to")
    col = next(c for c in header["layout"]["columns"] if c["name"] == "quality")
    q = {k: v for k, v in (header.get("quality") or {}).items() if k != "column"}
    return {"decoder_id": did, "bitstream_id": codec.get("bitstream_id"),
            "release": codec.get("release"), "rungs": header.get("rungs", []),
            "fields": col["fields"], "quality": q}


class RecordWriter:
    """Writes one product as a Zarr store at `path`. Window regions can be
    added in any order, from independent encoders; `close` writes the
    store, so it does not depend on the order of `add_windows` calls.
    Every window of the product must be added. Records given for windows
    of an empty chunk (no valid pixel in any of its windows) are not
    stored.

    header: the codec, product, rungs and quality entries of the
    `georvq` attribute; with a quality column, `codec.decoder_id` names
    the decoder it was measured with. transform: the north-up affine
    (a, b, c, d, e, f) of the coded pixel grid. crs: PROJJSON dict, or
    None if unknown. aux: a JSON string or JSON-able object stored under
    `georvq_aux`. consolidated: add zarr-python's consolidated metadata,
    which saves readers a request per array on open."""

    def __init__(self, path, header: dict, layout: Layout, product_id: str,
                 transform, height: int, width: int, crs=None, aux=None,
                 consolidated=True):
        clash = [k for k in _WRITER_KEYS if k in header]
        if clash:
            raise ValueError(f"header keys {clash} are set by the writer")
        self.path = os.fspath(path).rstrip("/")
        self.header = dict(header)
        self.layout = layout
        self.product_id = str(product_id)
        self.transform = [float(v) for v in transform[:6]]
        self.height, self.width = int(height), int(width)
        self.crs = crs
        self.aux = aux if aux is None or not isinstance(aux, str) else json.loads(aux)
        self.consolidated = bool(consolidated)
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

    def _header(self):
        """The `georvq` attribute: the header with the layout, its empty
        chunks and the mask chunks holding an invalid pixel. Zeroes the
        records of empty chunks."""
        lay = self.layout.to_json()
        live = self._live()
        empty_of = {}
        for g in lay["groups"]:
            kind = g["kind"]
            ny, nx = self.chunk_grids[kind]
            n = self.layout.groups[kind].chunk_windows
            empty = [[i, j] for i in range(ny) for j in range(nx)
                     if not live[i * n:(i + 1) * n, j * n:(j + 1) * n].any()]
            for i, j in empty:
                for c in self.layout.groups[kind].columns:
                    if c != "mask":
                        self._rec[c][i * n:(i + 1) * n, j * n:(j + 1) * n] = 0
            g["chunk_grid"] = [ny, nx]
            g["empty_chunks"] = empty
            empty_of[kind] = {tuple(c) for c in empty}
        masked = []
        if self._valid is not None:
            mk = self.layout.mask_kind
            p = self.layout.groups[mk].chunk_px
            ny, nx = self.chunk_grids[mk]
            for i in range(ny):
                for j in range(nx):
                    blk = self._valid[i * p:(i + 1) * p, j * p:(j + 1) * p]
                    if not blk.all() and (i, j) not in empty_of[mk]:
                        masked.append([i, j])
        lay["mask"]["chunks"] = masked
        lay["window_grid"] = list(self.window_grid)
        lay["product_px"] = [self.height, self.width]
        header = dict(self.header)
        header.update({"format": FORMAT, "version": f"{MAJOR}.{MINOR}",
                       "status": STATUS, "layout": lay})
        return header

    def _write(self, out, header):
        import warnings
        import zarr
        from zarr.codecs import ZstdCodec
        lay = self.layout
        gy, gx = self.window_grid
        H, Wd = self.height, self.width
        crs = _crs_attrs(header.get("product") or {}, self.crs)
        conv = {"zarr_conventions": [PROJ_CONVENTION, SPATIAL_CONVENTION],
                **({"proj:code": crs["proj:code"]} if "proj:code" in crs else {})}
        win_attrs = {**conv, **_spatial(("wy", "wx"), (gy, gx), _window_transform(
            self.transform, lay.window_px))}
        root = zarr.open_group(out, mode="w-", zarr_format=3)

        def records(col):
            return self._rec[col][:gy, :gx]

        def plain(parent, name, col, attrs):
            n = lay.groups[lay.kind_of[col]].chunk_windows
            rb = lay.columns[col].record_bytes
            a = parent.create_array(
                name, shape=(gy, gx, rb), dtype="uint8", chunks=(n, n, rb),
                compressors=None, filters=None, fill_value=0,
                dimension_names=["wy", "wx", f"{col}_byte"], attributes=attrs,
                config={"write_empty_chunks": True})
            a[...] = records(col)

        plain(root, "base", "base", win_attrs)
        for t in lay.tiers:
            if t.in_base:
                continue
            n = lay.groups[t.name].chunk_windows
            rb = lay.columns[t.columns[0]].record_bytes
            ny, nx = self.chunk_grids[t.name]
            a = root.create_array(
                t.name, shape=(gy, gx, t.levels, rb), dtype="uint8",
                chunks=(n, n, 1, rb), shards=(ny * n, nx * n, t.levels, rb),
                compressors=None, filters=None, fill_value=0,
                dimension_names=["wy", "wx", f"{t.name}_level", f"{t.name}_byte"],
                attributes=win_attrs, config={"write_empty_chunks": False})
            a[...] = np.stack([records(c) for c in t.columns], 2)
        if lay.quality is not None:
            q = quality_info(header)
            qg = root.create_group(QUALITY_GROUP,
                                   attributes={"decoders": [q["decoder_id"]]})
            plain(qg, q["decoder_id"], "quality", {**win_attrs, QUALITY_KEY: q})
        p = lay.groups[lay.mask_kind].chunk_px
        mask = root.create_array(
            "mask", shape=(H, Wd), dtype="bool", chunks=(p, p),
            compressors=ZstdCodec(level=MASK_ZSTD_LEVEL), filters=None,
            fill_value=True, dimension_names=["y", "x"],
            attributes={**conv, **_spatial(("y", "x"), (H, Wd), self.transform)},
            config={"write_empty_chunks": False})
        if self._valid is not None:
            mask[...] = self._valid
        root.update_attributes({
            HEADER_KEY: header, AUX_KEY: self.aux,
            "zarr_conventions": [PROJ_CONVENTION, SPATIAL_CONVENTION],
            **crs, **_spatial(("y", "x"), (H, Wd), self.transform)})
        if self.consolidated:
            # zarr-python warns on every write that consolidated metadata is
            # not part of the v3 specification
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Consolidated metadata")
                zarr.consolidate_metadata(out)

    def close(self) -> None:
        if self._closed:
            return
        if not self._filled.all():
            ys, xs = np.nonzero(~self._filled)
            raise ValueError(f"{len(ys)} windows were not added, first "
                             f"({ys[0]}, {xs[0]})")
        header = self._header()
        # written beside the target and moved over it when complete, so a
        # failed or killed write never leaves a partial store at self.path
        tag = f"{os.getpid()}.{threading.get_ident()}"
        tmp, old = f"{self.path}.{tag}.tmp", f"{self.path}.{tag}.old"
        try:
            self._write(tmp, header)
            if os.path.exists(self.path):
                os.replace(self.path, old)
            os.replace(tmp, self.path)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            if os.path.exists(old) and not os.path.exists(self.path):
                os.replace(old, self.path)
            raise
        shutil.rmtree(old, ignore_errors=True)
        self._rec = {}
        self._valid = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

def open_store(path_or_url, storage_options=None):
    """The zarr.Group of a store at a local path or a URL fsspec opens
    (https://, s3://, hf://, ...). `storage_options` go to fsspec, for
    example {"headers": {"Authorization": "Bearer <token>"}} over https."""
    import zarr
    kw = {} if storage_options is None else {"storage_options": storage_options}
    # without zarr_format, zarr-python also asks for v2 metadata, three
    # more requests that all fail
    return zarr.open_group(os.fspath(path_or_url), mode="r", zarr_format=3, **kw)


def _is_group(obj) -> bool:
    return not isinstance(obj, (str, bytes, os.PathLike)) and all(
        hasattr(obj, a) for a in ("attrs", "array_keys", "store"))


class RecordReader:
    """Reads records and pixel masks of window regions from a store, given
    as an open zarr.Group or as a path or URL for `open_store` (with
    `storage_options`). zarr-python does every read, so any store it
    opens works, including copies rechunked or recompressed with Zarr
    tools: only the shape and data type of each array are checked.

    A read lists the (column, chunk) values it needs, leaves out the
    cached ones, and makes one zarr-python read per array over the
    bounding box of the rest, every level down to the deepest asked for;
    the reads of different arrays run in parallel. No empty chunk is
    read, nor any chunk whose windows in the read all lie in empty
    chunks of any kind. `cache` defaults to a RecordCache of
    `cache_bytes`. `stats` counts the values read and their bytes, and
    the values served from the cache and their bytes. `trace`, a Trace,
    records the reads and the unpacking when given.

    Quality is served for the store's own decoder (the header's
    `codec.decoder_id`) until `select_quality` picks another."""

    def __init__(self, store, cache=None, cache_bytes=CACHE_BYTES, trace=None,
                 storage_options=None):
        if not _is_group(store):
            self.source = os.fspath(store)
            store = open_store(store, storage_options)
        else:
            if storage_options is not None:
                raise ValueError("storage_options apply to a path or URL, not "
                                 "an open zarr.Group")
            path = getattr(store, "path", "") or ""
            self.source = f"{store.store}" + (f"/{path}" if path else "")
        self.group = store
        self.trace = trace
        self._lock = threading.Lock()
        self.stats = {"values": 0, "bytes": 0, "cache_hits": 0,
                      "cache_bytes": 0}
        attrs = dict(store.attrs)
        if HEADER_KEY not in attrs:
            raise ValueError(f"{self.source}: no georvq header")
        self.header = attrs[HEADER_KEY]
        check_header(self.header, self.source)
        aux = attrs.get(AUX_KEY)
        self.aux = None if aux is None else json.dumps(aux)
        self._from_header()
        self.arrays = {"base": self._check("base", store["base"]),
                       "mask": self._check("mask", store["mask"])}
        for t in self.layout.tiers:
            if not t.in_base:
                self.arrays[t.name] = self._check(t.name, store[t.name])
        self._init_quality(store)
        bid = (self.header.get("codec") or {}).get("bitstream_id", "")
        ident = json.dumps([self.source, attrs.get(HEADER_KEY)], sort_keys=True)
        self.file_key = (str(bid), hashlib.sha1(ident.encode()).hexdigest()[:20])
        self.cache = cache if cache is not None else RecordCache(cache_bytes)

    def _from_header(self):
        """Layout, grids, empty chunks and masks from `self.header`."""
        lay = self.header["layout"]
        self.layout = Layout.from_json(lay)
        self.window_grid = tuple(int(v) for v in lay["window_grid"])
        self.chunk_grids, self.empty = {}, {}
        for g in lay["groups"]:
            self.chunk_grids[g["kind"]] = tuple(int(v) for v in g["chunk_grid"])
            self.empty[g["kind"]] = {tuple(c) for c in g.get("empty_chunks", [])}
        # windows of empty chunks have no valid pixel, whatever their kind
        self.dead = np.zeros(self.window_grid, bool)
        for kind, chunks in self.empty.items():
            n = self.layout.groups[kind].chunk_windows
            for i, j in chunks:
                self.dead[i * n:(i + 1) * n, j * n:(j + 1) * n] = True
        self.masked = {tuple(c) for c in lay["mask"].get("chunks", [])}
        self.height, self.width = (int(v) for v in lay["product_px"])

    def _check(self, name, arr, record=None):
        lay = self.layout
        gy, gx = self.window_grid
        if name == "mask":
            want, dtype = (self.height, self.width), "bool"
        elif name == "base":
            want, dtype = (gy, gx, lay.columns["base"].record_bytes), "uint8"
        elif name == "quality":
            want, dtype = (gy, gx, record), "uint8"
        else:
            t = next(t for t in lay.tiers if t.name == name)
            want = (gy, gx, t.levels, lay.columns[t.columns[0]].record_bytes)
            dtype = "uint8"
        if tuple(arr.shape) != want or str(arr.dtype) != dtype:
            raise ValueError(f"{self.source}: {name} is {arr.dtype} "
                             f"{tuple(arr.shape)}, the header needs {dtype} {want}")
        return arr

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        pass

    # ---- quality, per decoder ----

    def _init_quality(self, store) -> None:
        self._base_header = self.header
        codec = self.header.get("codec") or {}
        self.quality_decoders = {}
        if QUALITY_GROUP in store:
            qg = store[QUALITY_GROUP]
            for did in qg.attrs.get("decoders", []):
                arr = qg[did]
                q = dict(arr.attrs).get(QUALITY_KEY)
                if q is None or q.get("decoder_id") != did:
                    raise ValueError(f"{self.source}: quality/{did} does not name "
                                     f"the decoder it was measured with")
                if q.get("bitstream_id") != codec.get("bitstream_id"):
                    raise ValueError(f"{self.source}: quality/{did} holds quality "
                                     f"for bitstream {q.get('bitstream_id')}, the "
                                     f"store is {codec.get('bitstream_id')}")
                fields = [(f["name"], f["count"], f["bits"]) for f in q["fields"]]
                self._check("quality", arr, Column("quality", fields).record_bytes)
                self.quality_decoders[did] = {"array": arr, **q}
        own = codec.get("decoder_id") if self.layout.quality is not None else None
        if own is not None and own not in self.quality_decoders:
            raise ValueError(f"{self.source}: no quality array for its decoder "
                             f"{own}")
        self.quality_decoder = None
        self.select_quality(own)

    def select_quality(self, decoder_id) -> None:
        """Serve the quality values measured with `decoder_id` (None for
        no quality column). `header` then carries that decoder's rungs
        and quality definition, and `layout` its quality record."""
        if decoder_id is not None and decoder_id not in self.quality_decoders:
            raise ValueError(f"{self.source}: no quality for decoder "
                             f"{decoder_id}; the store has quality for "
                             f"{sorted(self.quality_decoders) or 'no decoder'}")
        h = copy.deepcopy(self._base_header)
        own = (h.get("codec") or {}).get("decoder_id")
        if decoder_id is None or decoder_id != own:
            lay = h["layout"]
            qgroup = [g for g in lay["groups"] if g["kind"] == "quality"]
            if decoder_id is not None and not qgroup:
                b = next(g for g in lay["groups"] if g["kind"] == "base")
                qgroup = [{**b, "kind": "quality", "columns": ["quality"],
                           "empty_chunks": b.get("empty_chunks", [])}]
            lay["groups"] = [g for g in lay["groups"] if g["kind"] != "quality"]
            lay["columns"] = [c for c in lay["columns"] if c["name"] != "quality"]
            h.pop("quality", None)
            if decoder_id is not None:
                q = self.quality_decoders[decoder_id]
                fields = [(f["name"], f["count"], f["bits"]) for f in q["fields"]]
                lay["groups"] += qgroup
                lay["columns"].append({
                    "name": "quality", "fields": q["fields"],
                    "record_bytes": Column("quality", fields).record_bytes})
                h["quality"] = {"column": "quality", **q["quality"]}
                h["rungs"] = q["rungs"]
        self.header = h
        self._from_header()
        self.quality_decoder = decoder_id
        if decoder_id is None:
            self.arrays.pop("quality", None)
        else:
            self.arrays["quality"] = self.quality_decoders[decoder_id]["array"]

    # ---- planning ----

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
        over the windows needs. Masks of chunks with every pixel valid,
        and chunks that `skipped` rules out, are left out."""
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
        if need[0] == "quality":
            return (self.file_key, "quality", self.quality_decoder) + tuple(need[1:])
        return (self.file_key,) + tuple(need)

    def _array_of(self, col) -> str:
        if col in ("base", "mask", "quality"):
            return col
        return self.layout.kind_of[col]

    def _regions(self, needs) -> dict:
        """{array: (slices, needs)} of one zarr-python read per array, over
        the bounding box of its needed chunks."""
        lay = self.layout
        by = {}
        for n in needs:
            by.setdefault(self._array_of(n[0]), []).append(n)
        out = {}
        for name, ns in by.items():
            i0, i1 = min(n[1] for n in ns), max(n[1] for n in ns) + 1
            j0, j1 = min(n[2] for n in ns), max(n[2] for n in ns) + 1
            arr = self.arrays[name]
            if name == "mask":
                p = lay.groups[lay.mask_kind].chunk_px
                sl = (slice(i0 * p, min(arr.shape[0], i1 * p)),
                      slice(j0 * p, min(arr.shape[1], j1 * p)))
            else:
                k = lay.groups[lay.kind_of[ns[0][0]]].chunk_windows
                sl = (slice(i0 * k, min(arr.shape[0], i1 * k)),
                      slice(j0 * k, min(arr.shape[1], j1 * k)))
                if name not in ("base", "quality"):
                    t = next(t for t in lay.tiers if t.name == name)
                    depth = max(t.columns.index(n[0]) for n in ns) + 1
                    sl += (slice(0, depth),)
            out[name] = (sl, ns)
        return out

    def plan(self, wy0, wy1, wx0, wx1, depths=None, base=True, mask=True,
             quality=False) -> list:
        """(array name, slices) of the zarr-python reads a read would
        make now."""
        cols = self.columns_for(depths, base, mask, quality)
        needs = [n for n in self.needs(wy0, wy1, wx0, wx1, cols)
                 if not self.cache.has(self._key(n))]
        return [(name, sl) for name, (sl, _) in self._regions(needs).items()]

    # ---- values ----

    def _decode_mask(self, v, shape) -> np.ndarray:
        """A mask value (the chunk's pixels, one byte each, padded to the
        full chunk) as bool `shape`."""
        p = self.layout.groups[self.layout.mask_kind].chunk_px
        if len(v) != p * p:
            raise ValueError(f"mask value of {len(v)} bytes, expected {p * p}")
        return np.frombuffer(v, np.uint8).reshape(p, p)[
            :shape[0], :shape[1]].astype(bool)

    def values(self, needs) -> dict:
        """{(column, chunk_row, chunk_col): value bytes} for `needs`, from
        the cache where held and otherwise from zarr-python reads. A code
        value holds the chunk's records, padded with zero records to the
        full chunk at the edges; a mask value holds the chunk's pixels,
        one byte each, padded to the full chunk."""
        out, missing, hits, hit_bytes = {}, [], 0, 0
        for n in needs:
            v = self.cache.get(self._key(n))
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
        lay = self.layout
        regions = self._regions(missing)

        def get(name):
            sl = regions[name][0]
            with span(self.trace, "get", name):
                return name, np.asarray(self.arrays[name][sl])
        if len(regions) > 1:
            with ThreadPoolExecutor(len(regions)) as ex:
                slabs = dict(ex.map(get, regions))
        else:
            slabs = dict(map(get, regions))
        got = 0
        with span(self.trace, "cpu", "slice values"):
            for name, (sl, ns) in regions.items():
                slab = slabs[name]
                r0, c0 = sl[0].start, sl[1].start
                for n in ns:
                    col, i, j = n
                    if name == "mask":
                        p = lay.groups[lay.mask_kind].chunk_px
                        blk = np.zeros((p, p), np.uint8)
                        sub = slab[i * p - r0:(i + 1) * p - r0,
                                   j * p - c0:(j + 1) * p - c0]
                    else:
                        k = lay.groups[lay.kind_of[col]].chunk_windows
                        rb = lay.columns[col].record_bytes
                        blk = np.zeros((k, k, rb), np.uint8)
                        sub = slab[i * k - r0:(i + 1) * k - r0,
                                   j * k - c0:(j + 1) * k - c0]
                        if name not in ("base", "quality"):
                            t = next(t for t in lay.tiers if t.name == name)
                            sub = sub[:, :, t.columns.index(col)]
                    blk[:sub.shape[0], :sub.shape[1]] = sub
                    v = blk.tobytes()
                    out[n] = v
                    got += len(v)
                    self.cache.put(self._key(n), v)
        with self._lock:
            self.stats["values"] += len(missing)
            self.stats["bytes"] += got
        return out

    def read(self, wy0, wy1, wx0, wx1, depths=None, base=True, mask=True,
             quality=False) -> dict:
        """Records of windows [wy0:wy1, wx0:wx1].

        Returns {"base": (h, w, n), tier name: (h, w, L, n) for each
        tier outside base read at L > 0, "quality": (h, w, n), "valid":
        (h * W, w * W) pixel validity, False beyond the product}, with
        only the parts asked for. Windows without a valid pixel that were
        not read have zero records."""
        if depths is None:
            depths = [t.levels for t in self.layout.tiers]
        cols = self.columns_for(depths, base, mask, quality)
        vals = self.values(self.needs(wy0, wy1, wx0, wx1, cols))
        with span(self.trace, "cpu", "unpack records"):
            return self._unpack(vals, wy0, wy1, wx0, wx1, depths, base,
                                mask, quality)

    def _unpack(self, vals, wy0, wy1, wx0, wx1, depths, base, mask, quality):
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
                    blk = self._decode_mask(vals[("mask", i, j)], (bh, bw))
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

    # ---- sizes ----

    def stored_bytes(self) -> dict:
        """Stored bytes per array (zarr-python's nbytes_stored, which lists
        the store, so it needs a store that can be listed, such as a
        local directory)."""
        out = {name: int(a.nbytes_stored()) for name, a in self.arrays.items()
               if name != "quality"}
        for did, q in self.quality_decoders.items():
            out[f"quality/{did}"] = int(q["array"].nbytes_stored())
        return out

    def column_bytes(self) -> dict:
        """Bytes of the records of each code column outside empty chunks,
        as a whole-product read takes them: chunks of `kind` times its
        chunk's records."""
        out = {}
        for name, col in self.layout.columns.items():
            kind = self.layout.kind_of[name]
            g = self.layout.groups[kind]
            ny, nx = self.chunk_grids[kind]
            live = ny * nx - len(self.empty[kind])
            out[name] = live * g.chunk_windows ** 2 * col.record_bytes
        return out
