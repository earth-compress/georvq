"""Reader and writer for the .gvq container (docs/GVQ_FORMAT.md).

A .gvq file is a 16-byte preamble, a JSON header padded to a region of
`H` bytes, fixed-size sections of per-window records, and variable
sections (mask, aux) at the end. Every section starts on a 4096-byte
boundary. The byte range of any window at any level follows from the
header by arithmetic, so a reader over HTTP needs one request for the
header and then one range per chunk per section.

This module is source-agnostic and depends only on numpy and the
standard library, so a reader can be installed without torch.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass, field

import numpy as np

from georvq.inference.bitpack import pack, record_bytes, unpack

MAGIC = b"GVQ\0"
MAJOR = 1
MINOR = 0
ALIGN = 4096
PREAMBLE = struct.Struct("<4sHHII")
FIRST_READ = 64 * 1024

MASK_CODEC = "packbits-zlib"
AUX_CODEC = "zip"
# Members with these suffixes are stored in the aux zip without deflate.
STORED_SUFFIXES = (".jp2", ".png", ".jpg", ".jpeg", ".zip", ".gz", ".bz2",
                   ".xz", ".zst", ".7z")
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_WRITER_KEYS = ("windows", "fixed_sections", "variable_sections")


def _align(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


@dataclass(frozen=True)
class SectionSpec:
    """Layout of one fixed section. `fields` is a list of
    (name, count, bits); a record is one window at one level."""
    name: str
    chunk_windows: tuple
    levels: int
    fields: list = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, "chunk_windows",
                           tuple(int(c) for c in self.chunk_windows))
        object.__setattr__(self, "fields",
                           [(str(n), int(c), int(b)) for n, c, b in self.fields])
        if len(self.chunk_windows) != 2 or min(self.chunk_windows) < 1:
            raise ValueError(f"{self.name}: chunk_windows must be two positive ints")
        if self.levels < 1:
            raise ValueError(f"{self.name}: levels must be >= 1")

    @property
    def widths(self) -> list:
        return [b for _, c, b in self.fields for _ in range(c)]

    @property
    def n_values(self) -> int:
        return sum(c for _, c, _ in self.fields)

    @property
    def record_bytes(self) -> int:
        return record_bytes(self.widths)

    @property
    def slab_bytes(self) -> int:
        cy, cx = self.chunk_windows
        return cy * cx * self.record_bytes

    @property
    def chunk_bytes(self) -> int:
        return self.levels * self.slab_bytes

    def chunk_grid(self, window_grid) -> tuple:
        cy, cx = self.chunk_windows
        return (-(-int(window_grid[0]) // cy), -(-int(window_grid[1]) // cx))

    def size(self, window_grid) -> int:
        ny, nx = self.chunk_grid(window_grid)
        return ny * nx * self.chunk_bytes

    def record_offsets(self, window_grid, level, wy, wx) -> np.ndarray:
        """Byte offsets, relative to the section start, of the records of
        windows (wy, wx) at `level`."""
        cy, cx = self.chunk_windows
        nx = self.chunk_grid(window_grid)[1]
        wy = np.asarray(wy, dtype=np.int64)
        wx = np.asarray(wx, dtype=np.int64)
        chunk = (wy // cy) * nx + (wx // cx)
        return (chunk * self.chunk_bytes + int(level) * self.slab_bytes
                + ((wy % cy) * cx + (wx % cx)) * self.record_bytes)

    def descriptor(self, offset: int) -> dict:
        return {"name": self.name, "offset": int(offset),
                "chunk_windows": list(self.chunk_windows),
                "levels": self.levels,
                "fields": [{"name": n, "count": c, "bits": b}
                           for n, c, b in self.fields],
                "record_bytes": self.record_bytes}

    @classmethod
    def from_descriptor(cls, d: dict) -> "SectionSpec":
        spec = cls(d["name"], d["chunk_windows"], int(d["levels"]),
                   [(f["name"], f["count"], f["bits"]) for f in d["fields"]])
        if spec.record_bytes != int(d["record_bytes"]):
            raise ValueError(f"section {spec.name}: record_bytes "
                             f"{d['record_bytes']} disagrees with its fields")
        return spec


def encode_mask(valid, block_px: int) -> bytes:
    """Mask section bytes for a (H, W) validity array (True = valid).
    Edge blocks are clipped to the array. All valid gives b""."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("mask must be 2-D")
    h, w = valid.shape
    b = int(block_px)
    blobs = []
    for y in range(0, h, b):
        for x in range(0, w, b):
            blk = valid[y:y + b, x:x + b]
            blobs.append(b"" if blk.all()
                         else zlib.compress(np.packbits(blk.ravel()).tobytes(), 9))
    if not any(blobs):
        return b""
    table = np.zeros((len(blobs), 2), dtype="<u8")
    pos = 4 + 16 * len(blobs)
    for i, blob in enumerate(blobs):
        if blob:
            table[i] = (pos, len(blob))
            pos += len(blob)
    return struct.pack("<I", len(blobs)) + table.tobytes() + b"".join(blobs)


def encode_aux(members) -> bytes:
    """Zip archive of {name: bytes}, with fixed timestamps so the output
    depends only on the members. Already-compressed members are stored."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
            info.compress_type = (zipfile.ZIP_STORED
                                  if name.lower().endswith(STORED_SUFFIXES)
                                  else zipfile.ZIP_DEFLATED)
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return buf.getvalue()


class GVQWriter:
    """Writes a .gvq file. Fixed sections are preallocated as zeros, so
    records can be written in any order and padding windows stay zero.
    `close` appends the variable sections and writes the header."""

    def __init__(self, path, header: dict, window_grid, sections,
                 header_reserve: int = FIRST_READ):
        clash = [k for k in _WRITER_KEYS if k in header]
        if clash:
            raise ValueError(f"header keys {clash} are set by the writer")
        if header_reserve % ALIGN or header_reserve < ALIGN:
            raise ValueError(f"header_reserve must be a multiple of {ALIGN}")
        names = [s.name for s in sections]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate section names in {names}")
        self.path = os.fspath(path)
        self.header = dict(header)
        self.window_grid = (int(window_grid[0]), int(window_grid[1]))
        self.header_reserve = int(header_reserve)
        self.sections = {s.name: s for s in sections}
        self.offsets = {}
        pos = self.header_reserve
        for s in sections:
            self.offsets[s.name] = pos
            pos = _align(pos + s.size(self.window_grid))
        self.fixed_end = pos
        self._variable = {}
        with open(self.path, "wb") as f:
            f.truncate(self.fixed_end)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r+",
                             shape=(self.fixed_end,))

    def write_records(self, section: str, level: int, window_rows, window_cols,
                      values) -> None:
        """values: (n, n_values) ints for windows (window_rows[i],
        window_cols[i]) at `level`."""
        spec = self.sections[section]
        wy = np.asarray(window_rows, dtype=np.int64).ravel()
        wx = np.asarray(window_cols, dtype=np.int64).ravel()
        values = np.asarray(values).reshape(len(wy), -1)
        if values.shape[1] != spec.n_values:
            raise ValueError(f"{section}: {values.shape[1]} values per record, "
                             f"layout has {spec.n_values}")
        if not 0 <= level < spec.levels:
            raise ValueError(f"{section}: level {level} not in 0..{spec.levels - 1}")
        rows, cols = self.window_grid
        if len(wy) and (wy.min() < 0 or wy.max() >= rows
                        or wx.min() < 0 or wx.max() >= cols):
            raise ValueError(f"{section}: window outside the {rows}x{cols} grid")
        data = pack(values, spec.widths)
        start = self.offsets[section] + spec.record_offsets(
            self.window_grid, level, wy, wx)
        idx = start[:, None] + np.arange(spec.record_bytes, dtype=np.int64)
        self._mm[idx] = data

    def set_mask(self, valid, block_px: int) -> None:
        valid = np.asarray(valid, dtype=bool)
        self._variable["mask"] = (encode_mask(valid, block_px), MASK_CODEC,
                                  {"block_px": int(block_px),
                                   "shape": list(valid.shape)})

    def set_aux(self, aux) -> None:
        """aux: zip archive bytes, or a dict of member name -> bytes."""
        data = bytes(aux) if isinstance(aux, (bytes, bytearray)) else encode_aux(aux)
        self._variable["aux"] = (data, AUX_CODEC, {})

    def close(self) -> None:
        if self._mm is None:
            return
        self._mm.flush()
        self._mm = None
        variable = []
        with open(self.path, "r+b") as f:
            pos = self.fixed_end
            for name, (data, codec, extra) in self._variable.items():
                f.seek(pos)
                f.write(data)
                variable.append({"name": name, "offset": pos, "length": len(data),
                                 "codec": codec,
                                 "sha256": hashlib.sha256(data).hexdigest(),
                                 **extra})
                pos = _align(pos + len(data))
            f.truncate(max(pos, self.fixed_end))
            header = dict(self.header)
            header["windows"] = list(self.window_grid)
            header["fixed_sections"] = [
                s.descriptor(self.offsets[n]) for n, s in self.sections.items()]
            header["variable_sections"] = variable
            text = json.dumps(header, separators=(",", ":")).encode("utf-8")
            room = self.header_reserve - PREAMBLE.size
            if len(text) > room:
                raise ValueError(f"header JSON is {len(text)} bytes, the reserved "
                                 f"region holds {room}; raise header_reserve")
            f.seek(0)
            f.write(PREAMBLE.pack(MAGIC, MAJOR, MINOR, len(text),
                                  self.header_reserve))
            f.write(text + b" " * (room - len(text)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:
            self._mm = None


class _RangeFile(io.RawIOBase):
    """Seekable read-only view of [offset, offset + length) of a reader's
    source, for zipfile. Reads are at least `readahead` bytes and the
    last fetched block is kept."""

    def __init__(self, reader, offset, length, readahead=16384):
        self._r, self._off, self._len, self._ra = reader, offset, length, readahead
        self._pos = 0
        self._buf_start, self._buf = 0, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, pos, whence=0):
        base = {0: 0, 1: self._pos, 2: self._len}[whence]
        self._pos = max(0, base + pos)
        return self._pos

    def read(self, n=-1):
        end = self._len if n is None or n < 0 else min(self._len, self._pos + n)
        if end <= self._pos:
            return b""
        bs, be = self._buf_start, self._buf_start + len(self._buf)
        if not (bs <= self._pos and end <= be):
            fetch = min(self._len, max(end, self._pos + self._ra))
            self._buf = self._r._read(self._off + self._pos, fetch - self._pos)
            self._buf_start = bs = self._pos
        out = self._buf[self._pos - bs:end - bs]
        self._pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


class GVQReader:
    """Reads a .gvq file from a local path or an http(s) URL (HTTP Range
    requests). `stats` counts requests and bytes fetched."""

    def __init__(self, path_or_url):
        self.source = os.fspath(path_or_url)
        self._url = self.source.startswith(("http://", "https://"))
        self._file = None if self._url else open(self.source, "rb")
        self.stats = {"requests": 0, "bytes": 0}
        head = self._read(0, FIRST_READ)
        if len(head) < PREAMBLE.size:
            raise ValueError(f"{self.source}: too short for a .gvq preamble")
        magic, major, minor, n, h = PREAMBLE.unpack(head[:PREAMBLE.size])
        if magic != MAGIC:
            raise ValueError(f"{self.source}: not a .gvq file (magic {magic!r})")
        if major != MAJOR:
            raise ValueError(f"{self.source}: .gvq major version {major}, this "
                             f"reader handles {MAJOR}")
        self.version = (major, minor)
        self.header_region = h
        end = PREAMBLE.size + n
        if end > len(head):
            head = head + self._read(len(head), end - len(head))
        self.header = json.loads(head[PREAMBLE.size:end].decode("utf-8"))
        self.window_grid = tuple(int(v) for v in self.header["windows"])
        self.sections = {}
        self.offsets = {}
        for d in self.header["fixed_sections"]:
            self.sections[d["name"]] = SectionSpec.from_descriptor(d)
            self.offsets[d["name"]] = int(d["offset"])
        self.variable = {d["name"]: d for d in self.header["variable_sections"]}
        self._mask_table = None

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

    def _chunks(self, spec, wy0, wy1, wx0, wx1):
        rows, cols = self.window_grid
        if not (0 <= wy0 < wy1 <= rows and 0 <= wx0 < wx1 <= cols):
            raise ValueError(f"windows [{wy0}:{wy1}, {wx0}:{wx1}] outside "
                             f"the {rows}x{cols} grid")
        cy, cx = spec.chunk_windows
        return [(i, j) for i in range(wy0 // cy, (wy1 - 1) // cy + 1)
                for j in range(wx0 // cx, (wx1 - 1) // cx + 1)]

    def byte_ranges(self, section: str, level_count: int, wy0: int, wy1: int,
                    wx0: int, wx1: int) -> list:
        """(offset, length) per chunk touched by windows [wy0:wy1, wx0:wx1],
        each covering levels 0..level_count-1 of that chunk."""
        spec = self.sections[section]
        if not 1 <= level_count <= spec.levels:
            raise ValueError(f"{section}: level_count {level_count} not in "
                             f"1..{spec.levels}")
        nx = spec.chunk_grid(self.window_grid)[1]
        base = self.offsets[section]
        return [(base + (i * nx + j) * spec.chunk_bytes,
                 level_count * spec.slab_bytes)
                for i, j in self._chunks(spec, wy0, wy1, wx0, wx1)]

    def read_records(self, section: str, level_count: int, wy0: int, wy1: int,
                     wx0: int, wx1: int) -> np.ndarray:
        """int64 array (wy1-wy0, wx1-wx0, level_count, n_values)."""
        spec = self.sections[section]
        cy, cx = spec.chunk_windows
        out = np.zeros((wy1 - wy0, wx1 - wx0, level_count, spec.n_values),
                       dtype=np.int64)
        ranges = self.byte_ranges(section, level_count, wy0, wy1, wx0, wx1)
        for (i, j), (off, length) in zip(self._chunks(spec, wy0, wy1, wx0, wx1),
                                         ranges):
            raw = np.frombuffer(self._read(off, length), dtype=np.uint8)
            if raw.size != length:
                raise OSError(f"{section}: short read at {off}")
            vals = unpack(raw.reshape(-1, spec.record_bytes), spec.widths)
            vals = vals.reshape(level_count, cy, cx, -1).transpose(1, 2, 0, 3)
            y0, x0 = i * cy, j * cx
            sy0, sy1 = max(wy0, y0), min(wy1, y0 + cy)
            sx0, sx1 = max(wx0, x0), min(wx1, x0 + cx)
            out[sy0 - wy0:sy1 - wy0, sx0 - wx0:sx1 - wx0] = \
                vals[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
        return out

    def read_variable(self, name: str, verify: bool = True) -> bytes:
        """A whole variable section, checked against its sha256."""
        d = self.variable[name]
        data = self._read(int(d["offset"]), int(d["length"]))
        if verify and hashlib.sha256(data).hexdigest() != d["sha256"]:
            raise ValueError(f"variable section {name}: sha256 mismatch")
        return data

    def read_mask(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Validity (True = valid) of pixels [y0:y1, x0:x1]. A file with
        no mask section is fully valid. Fully valid blocks are not fetched."""
        out = np.ones((y1 - y0, x1 - x0), dtype=bool)
        d = self.variable.get("mask")
        if d is None or int(d["length"]) == 0:
            return out
        h, w = d["shape"]
        b = int(d["block_px"])
        nbx = -(-w // b)
        if self._mask_table is None:
            base = int(d["offset"])
            (n,) = struct.unpack("<I", self._read(base, 4))
            self._mask_table = np.frombuffer(
                self._read(base + 4, 16 * n), dtype="<u8").reshape(n, 2)
        for by in range(max(y0, 0) // b, (min(y1, h) - 1) // b + 1):
            for bx in range(max(x0, 0) // b, (min(x1, w) - 1) // b + 1):
                off, length = (int(v) for v in self._mask_table[by * nbx + bx])
                if length == 0:
                    continue
                py0, px0 = by * b, bx * b
                bh, bw = min(b, h - py0), min(b, w - px0)
                bits = np.frombuffer(zlib.decompress(
                    self._read(int(d["offset"]) + off, length)), dtype=np.uint8)
                blk = np.unpackbits(bits, count=bh * bw).reshape(bh, bw).astype(bool)
                sy0, sy1 = max(y0, py0), min(y1, py0 + bh)
                sx0, sx1 = max(x0, px0), min(x1, px0 + bw)
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                    blk[sy0 - py0:sy1 - py0, sx0 - px0:sx1 - px0]
        return out

    def _aux_zip(self) -> zipfile.ZipFile:
        d = self.variable["aux"]
        return zipfile.ZipFile(_RangeFile(self, int(d["offset"]), int(d["length"])))

    def aux_names(self) -> list:
        if "aux" not in self.variable:
            return []
        with self._aux_zip() as zf:
            return zf.namelist()

    def read_aux(self, name: str) -> bytes:
        """One aux member, fetched through the zip's central directory."""
        with self._aux_zip() as zf:
            return zf.read(name)
