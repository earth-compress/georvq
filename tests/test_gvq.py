"""The .gvq container: record round trip and byte layout against the
offset formula in docs/GVQ_FORMAT.md, request counts for partial reads
from a file and over HTTP, and the mask and aux variable sections."""
import hashlib
import http.server
import io
import os
import struct
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

from georvq.inference.gvq import (GVQReader, GVQWriter, SectionSpec,
                                  encode_aux)

GRID = (10, 7)
BASE = SectionSpec("base", (4, 4), 1, [("mean", 3, 12), ("coarse", 2, 5)])
FINE = SectionSpec("fine", (4, 4), 3, [("codes", 4, 9)])
HEADER = {"codec": {"bitstream_id": "test"}, "product": {"kind": "synthetic"},
          "rungs": []}


def _values(spec, seed):
    rng = np.random.default_rng(seed)
    hi = np.array([1 << w for w in spec.widths])
    return rng.integers(0, hi, size=(GRID[0], GRID[1], spec.levels, spec.n_values))


def _write(path, order="sequential", mask=None, aux=None, **kw):
    vals = {s.name: _values(s, i) for i, s in enumerate([BASE, FINE])}
    ys, xs = np.meshgrid(np.arange(GRID[0]), np.arange(GRID[1]), indexing="ij")
    ys, xs = ys.ravel(), xs.ravel()
    jobs = [(s, l, blk) for s in (BASE, FINE) for l in range(s.levels)
            for blk in range(4)]
    if order == "shuffled":
        np.random.default_rng(7).shuffle(jobs)
    with GVQWriter(path, HEADER, GRID, [BASE, FINE], **kw) as w:
        for spec, level, blk in jobs:
            sel = np.arange(len(ys))[blk::4]
            if order == "shuffled":
                sel = sel[::-1]
            w.write_records(spec.name, level, ys[sel], xs[sel],
                            vals[spec.name][ys[sel], xs[sel], level])
        if mask is not None:
            w.set_mask(mask, 16)
        if aux is not None:
            w.set_aux(aux)
    return vals


def test_section_sizes():
    assert BASE.record_bytes == 6          # 46 bits
    assert FINE.record_bytes == 5          # 36 bits
    assert BASE.chunk_grid(GRID) == (3, 2)
    assert FINE.chunk_bytes == 3 * 16 * 5


def test_round_trip_and_partial_windows(tmp_path):
    vals = _write(tmp_path / "a.gvq")
    with GVQReader(tmp_path / "a.gvq") as r:
        assert r.header["windows"] == list(GRID)
        assert r.header["codec"] == HEADER["codec"]
        got = r.read_records("base", 1, 0, GRID[0], 0, GRID[1])
        assert (got == vals["base"]).all()
        got = r.read_records("fine", 3, 0, GRID[0], 0, GRID[1])
        assert (got == vals["fine"]).all()
        got = r.read_records("fine", 2, 3, 9, 2, 6)
        assert (got == vals["fine"][3:9, 2:6, :2]).all()


def test_shuffled_writes_give_identical_bytes(tmp_path):
    _write(tmp_path / "a.gvq", aux={"x.txt": b"x"})
    _write(tmp_path / "b.gvq", order="shuffled", aux={"x.txt": b"x"})
    assert (tmp_path / "a.gvq").read_bytes() == (tmp_path / "b.gvq").read_bytes()


def test_layout_matches_spec_formula(tmp_path):
    from georvq.inference.bitpack import pack
    vals = _write(tmp_path / "a.gvq")
    raw = (tmp_path / "a.gvq").read_bytes()
    magic, major, minor, n, h = struct.unpack("<4sHHII", raw[:16])
    assert magic == b"GVQ\0" and h == 65536 and h % 4096 == 0
    with GVQReader(tmp_path / "a.gvq") as r:
        descs = {d["name"]: d for d in r.header["fixed_sections"]}
    for spec in (BASE, FINE):
        d = descs[spec.name]
        assert d["offset"] % 4096 == 0 and d["offset"] >= h
        cy, cx = d["chunk_windows"]
        rb = d["record_bytes"]
        chunks_x = -(-GRID[1] // cx)
        slab = cy * cx * rb
        chunk = d["levels"] * slab
        for wy, wx, l in [(0, 0, 0), (9, 6, spec.levels - 1), (5, 3, 0), (4, 4, 1 % spec.levels)]:
            i, j, y, x = wy // cy, wx // cx, wy % cy, wx % cx
            off = d["offset"] + (i * chunks_x + j) * chunk + l * slab + (y * cx + x) * rb
            want = pack(vals[spec.name][wy, wx, l][None], spec.widths)[0].tobytes()
            assert raw[off:off + rb] == want
        # padding windows of the edge chunk (2, 1) are zero
        i, j = 2, 1
        start = d["offset"] + (i * chunks_x + j) * chunk
        y, x = 3, 3                          # window (11, 7): beyond the grid
        off = start + (y * cx + x) * rb
        assert raw[off:off + rb] == b"\0" * rb


def test_level_prefix_costs_one_range_per_chunk(tmp_path):
    _write(tmp_path / "a.gvq")
    with GVQReader(tmp_path / "a.gvq") as r:
        assert r.stats["requests"] == 1
        ranges = r.byte_ranges("fine", 2, 3, 6, 2, 6)   # chunks (0,0),(0,1),(1,0),(1,1)
        assert len(ranges) == 4
        assert all(length == 2 * FINE.slab_bytes for _, length in ranges)
        before = dict(r.stats)
        r.read_records("fine", 2, 3, 6, 2, 6)
        assert r.stats["requests"] - before["requests"] == 4
        assert r.stats["bytes"] - before["bytes"] == sum(l for _, l in ranges)
        assert len(r.byte_ranges("base", 1, 0, 1, 0, 1)) == 1


def test_mask_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    shape = (37, 50)
    mostly_invalid = rng.random(shape) < 0.1
    mostly_invalid[16:32, 16:32] = True     # one fully valid block
    _write(tmp_path / "m.gvq", mask=mostly_invalid)
    with GVQReader(tmp_path / "m.gvq") as r:
        assert (r.read_mask(0, 37, 0, 50) == mostly_invalid).all()
        assert (r.read_mask(5, 30, 11, 49) == mostly_invalid[5:30, 11:49]).all()
        data = r.read_variable("mask")
        n = struct.unpack("<I", data[:4])[0]
        assert n == 3 * 4
        table = np.frombuffer(data[4:4 + 16 * n], dtype="<u8").reshape(n, 2)
        assert table[1 * 4 + 1, 1] == 0
        before = r.stats["requests"]
        assert r.read_mask(16, 32, 16, 32).all()
        assert r.stats["requests"] == before

    _write(tmp_path / "v.gvq", mask=np.ones(shape, bool))
    with GVQReader(tmp_path / "v.gvq") as r:
        d = r.variable["mask"]
        assert d["length"] == 0 and d["sha256"] == hashlib.sha256(b"").hexdigest()
        before = r.stats["requests"]
        assert r.read_mask(0, 37, 0, 50).all()
        assert r.stats["requests"] == before


def test_aux_members(tmp_path):
    members = {"meta/tags.json": b'{"a": 1}' * 200, "preview.png": os.urandom(3000)}
    _write(tmp_path / "a.gvq", aux=members)
    with GVQReader(tmp_path / "a.gvq") as r:
        d = r.variable["aux"]
        assert d["offset"] % 4096 == 0
        assert r.aux_names() == list(members)
        for name, data in members.items():
            assert r.read_aux(name) == data
        zf = zipfile.ZipFile(io.BytesIO(r.read_variable("aux")))
        kinds = {i.filename: i.compress_type for i in zf.infolist()}
        assert kinds == {"meta/tags.json": zipfile.ZIP_DEFLATED,
                         "preview.png": zipfile.ZIP_STORED}
    assert encode_aux(members) == encode_aux(members)


def test_sha256_checked(tmp_path):
    _write(tmp_path / "a.gvq", aux={"x.txt": b"hello"})
    with GVQReader(tmp_path / "a.gvq") as r:
        off = r.variable["aux"]["offset"]
    with open(tmp_path / "a.gvq", "r+b") as f:
        f.seek(off + 40)
        f.write(b"Z")
    with GVQReader(tmp_path / "a.gvq") as r:
        with pytest.raises(ValueError, match="sha256"):
            r.read_variable("aux")


def test_refuses_bad_magic_and_major(tmp_path):
    _write(tmp_path / "a.gvq")
    raw = bytearray((tmp_path / "a.gvq").read_bytes())
    bad = raw.copy()
    bad[:4] = b"XYZ\0"
    (tmp_path / "b.gvq").write_bytes(bad)
    with pytest.raises(ValueError, match="magic"):
        GVQReader(tmp_path / "b.gvq")
    bad = raw.copy()
    bad[4:6] = struct.pack("<H", 99)
    (tmp_path / "c.gvq").write_bytes(bad)
    with pytest.raises(ValueError, match="major version 99"):
        GVQReader(tmp_path / "c.gvq")


def test_header_too_large(tmp_path):
    header = dict(HEADER, product={"blob": "x" * 5000})
    w = GVQWriter(tmp_path / "a.gvq", header, GRID, [BASE], header_reserve=4096)
    with pytest.raises(ValueError, match="header_reserve"):
        w.close()


def test_writer_rejects_reserved_keys_and_bad_windows(tmp_path):
    with pytest.raises(ValueError):
        GVQWriter(tmp_path / "a.gvq", dict(HEADER, windows=[1, 1]), GRID, [BASE])
    w = GVQWriter(tmp_path / "a.gvq", HEADER, GRID, [BASE])
    with pytest.raises(ValueError, match="outside"):
        w.write_records("base", 0, [10], [0], np.zeros((1, BASE.n_values), int))
    with pytest.raises(ValueError, match="level"):
        w.write_records("base", 1, [0], [0], np.zeros((1, BASE.n_values), int))
    w.close()


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    path_on_disk = None
    requests = []

    def do_GET(self):
        data = Path(self.path_on_disk).read_bytes()
        rng = self.headers.get("Range")
        type(self).requests.append(rng)
        if rng is None:
            self.send_response(200)
            body = data
        else:
            a, b = rng.removeprefix("bytes=").split("-")
            a, b = int(a), min(int(b), len(data) - 1)
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_http_range_reads(tmp_path):
    rng = np.random.default_rng(5)
    mask = rng.random((37, 50)) < 0.5
    members = {"tags.json": b"{}" * 100, "q.jp2": b"\x00\x01" * 50}
    vals = _write(tmp_path / "a.gvq", mask=mask, aux=members)

    handler = type("H", (_RangeHandler,), {"path_on_disk": str(tmp_path / "a.gvq"),
                                           "requests": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/a.gvq"
        r = GVQReader(url)
        assert handler.requests == ["bytes=0-65535"]
        got = r.read_records("fine", 2, 3, 6, 2, 6)
        assert (got == vals["fine"][3:6, 2:6, :2]).all()
        assert len(handler.requests) == 1 + 4
        assert r.stats["requests"] == len(handler.requests)
        assert (r.read_mask(0, 37, 0, 50) == mask).all()
        assert r.read_aux("q.jp2") == members["q.jp2"]
        assert r.read_aux("tags.json") == members["tags.json"]
        assert all(h.startswith("bytes=") for h in handler.requests)
    finally:
        server.shutdown()
        server.server_close()


def test_gvq_does_not_import_torch():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, georvq.inference.gvq; print('torch' in sys.modules)"],
        cwd=root, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
