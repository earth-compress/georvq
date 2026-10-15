"""Prototype Zarr store of a georvq GeoParquet file: the conversion keeps
every record, zarr-python reads the store, the levels of a tier chunk
are one byte range, and ZarrReader reads what RVQParquetReader reads,
locally and over HTTP. tests/test_ae_v2_zarr.py checks AE v2 decodes."""
import http.server
import json
import os
import re
import subprocess
import sys
import threading

import numpy as np
import pytest

from georvq.inference import georvq_parquet as gp
from georvq.inference import georvq_zarr as gz
from tests.test_georvq_parquet import FULL, H, W, _layout, _write

zarr = pytest.importorskip("zarr")


def _masks():
    rng = np.random.default_rng(3)
    scattered = rng.random((H, W)) > 0.1
    east = np.ones((H, W), bool)
    east[:, 24:] = False           # fine chunk columns 3..6 and mid 2..3 empty
    return {"full": None, "scattered": scattered, "east": east}


@pytest.fixture(params=["full", "scattered", "east"])
def product(request, tmp_path):
    valid = _masks()[request.param]
    src = tmp_path / "a.georvq.parquet"
    vals = _write(src, valid=valid)
    out = gz.parquet_to_zarr(src, tmp_path / "a.georvq.zarr")
    attrs = gz.parquet_to_zarr(src, tmp_path / "attrs" / "a.georvq.zarr",
                               offsets_in_attrs=True)
    return {"src": src, "zarr": out, "attrs": attrs, "vals": vals,
            "valid": valid, "name": request.param}


REGIONS = [((0, 10, 0, 13), None, True), ((0, 10, 0, 13), (3, 1, 2), False),
           ((2, 5, 7, 13), FULL, True), ((3, 4, 5, 6), (3, 2, 1), False),
           ((1, 9, 2, 11), (3, 0, 0), True), ((9, 10, 12, 13), FULL, True)]


def _same_reads(a, b):
    for (wy0, wy1, wx0, wx1), depths, quality in REGIONS:
        ra = a.read(wy0, wy1, wx0, wx1, depths=depths, quality=quality)
        rb = b.read(wy0, wy1, wx0, wx1, depths=depths, quality=quality)
        assert ra.keys() == rb.keys()
        for k in ra:
            assert ra[k].dtype == rb[k].dtype and np.array_equal(ra[k], rb[k]), k
    assert np.array_equal(a.read_mask(0, H, 0, W), b.read_mask(0, H, 0, W))
    assert np.array_equal(a.read_mask(5, 30, 17, 41), b.read_mask(5, 30, 17, 41))


@pytest.mark.parametrize("which", ["zarr", "attrs"])
def test_reader_reads_what_the_parquet_reader_reads(product, which):
    with gp.RVQParquetReader(product["src"]) as p, \
            gz.ZarrReader(product[which]) as z:
        assert z.header == p.header and json.loads(z.aux or "null") == (
            json.loads(p.aux) if p.aux else None)
        assert z.empty == p.empty and z.masked == p.masked
        _same_reads(p, z)


def test_values_are_the_parquet_values(product):
    with gp.RVQParquetReader(product["src"]) as p, \
            gz.ZarrReader(product["zarr"]) as z:
        needs = [n for n in p.needs(0, 10, 0, 13, p.columns_for(quality=True))
                 if n[0] != "mask"]
        assert p.values(needs) == z.values(needs)


def test_zarr_python_reads_the_store(product):
    lay = _layout()
    g = zarr.open_group(str(product["zarr"]), mode="r")
    assert set(g.array_keys()) == {"base", "mid", "fine", "quality", "mask"}
    with gp.RVQParquetReader(product["src"]) as p:
        assert g.attrs["georvq"] == p.header
        got = p.read(0, 10, 0, 13, quality=True)
        n_chunks = {k: len(p.empty[k]) for k in p.empty}
    for name, col in [("base", "base"), ("quality", "quality")]:
        rec = g[name][...]
        assert rec.shape == (10, 13, lay.columns[col].record_bytes)
    for t in ("mid", "fine"):
        a = g[t]
        assert all(s >= n for s, n in zip(a.shards, a.shape))   # one shard
        assert a.shape[:3] == (10, 13, lay.tiers[["coarse", "mid", "fine"].index(
            t)].levels)
    # unpack the stored bytes and compare with the parquet reader's records
    for t in ("mid", "fine"):
        raw = g[t][...]
        widths = lay.columns[f"{t}_1"].widths
        for lvl in range(raw.shape[2]):
            u = gp.unpack(raw[:, :, lvl].reshape(-1, raw.shape[3]), widths)
            assert np.array_equal(u.reshape(10, 13, -1), got[t][:, :, lvl]), t
    valid = product["valid"]
    m = g["mask"][...]
    assert m.dtype == bool and np.array_equal(m, np.ones((H, W), bool)
                                              if valid is None else valid)
    if product["name"] == "east":
        assert n_chunks["fine"] == 20 and n_chunks["mid"] == 6
        assert not g["fine"][:, 6:].any()


def test_levels_are_one_range_per_chunk(product):
    z = gz.ZarrReader(product["zarr"])
    z._load_indices(["mid", "fine"])
    for kind in ("mid", "fine"):
        a, idx = z.arrays[kind], z._index[kind]
        ny, nx, L, _ = a["chunks_per_shard"]
        at = 0
        for i in range(ny):
            for j in range(nx):
                o = idx[i, j, :, 0, 0]
                if (i, j) in z.empty[kind]:
                    assert (o == gz._MISSING).all()
                    continue
                # C order: levels 1..L of a chunk follow one another
                assert list(o.astype(np.int64)) == [at + l * a["chunk_bytes"]
                                                    for l in range(L)]
                at += L * a["chunk_bytes"]
        # the planned read of one chunk to any depth is one range
        n = z.layout.groups[kind].chunk_windows
        live = [(i, j) for i in range(ny) for j in range(nx)
                if (i, j) not in z.empty[kind]]
        i, j = live[-1]
        for L_ in range(1, L + 1):
            d = {"mid": (3, L_, 0), "fine": (3, 0, L_)}[kind]
            plan = z.plan(i * n, i * n + 1, j * n, j * n + 1, depths=d,
                          base=False, mask=False)
            assert len(plan) == 1 and plan[0][2] == L_ * a["chunk_bytes"]


def test_offsets_in_attrs_match_the_shard_index(product):
    z = gz.ZarrReader(product["zarr"], shard_index="shard")
    z._load_indices(["mid", "fine"])
    a = gz.ZarrReader(product["attrs"], shard_index="attrs")
    for kind in ("mid", "fine"):
        assert np.array_equal(z._index[kind], a._index[kind])
    with pytest.raises(ValueError, match="no shard offsets"):
        gz.ZarrReader(product["zarr"], shard_index="attrs")


def test_empty_chunks_are_not_stored_or_fetched(tmp_path):
    src = tmp_path / "a.georvq.parquet"
    _write(src, valid=_masks()["east"])
    out = gz.parquet_to_zarr(src, tmp_path / "a.georvq.zarr")
    z = gz.ZarrReader(out)
    assert z.empty["fine"] == {(i, j) for i in range(5) for j in range(3, 7)}
    before = dict(z.stats)
    got = z.read(2, 5, 7, 13, depths=FULL, quality=True)
    assert z.stats == before and not got["valid"].any()
    assert not got["fine"].any() and not got["base"].any()
    shard = os.path.getsize(os.path.join(out, "fine", "c", "0", "0", "0", "0"))
    rb = _layout().columns["fine_1"].record_bytes
    assert shard == 15 * 3 * 4 * rb + gz.index_bytes(5 * 7 * 3)


def test_index_crc_is_checked():
    idx = np.arange(8, dtype=np.uint64).reshape(2, 2, 2)
    buf = gz.encode_index(idx)
    assert np.array_equal(gz.decode_index(buf, (2, 2)), idx)
    bad = bytearray(buf)
    bad[3] ^= 1
    with pytest.raises(ValueError, match="crc32c"):
        gz.decode_index(bytes(bad), (2, 2))
    # the pure-python crc32c agrees with the check value of the standard
    gz._CRC_TABLE = gz._crc32c_table()
    c, t = 0xFFFFFFFF, gz._CRC_TABLE
    for b in b"123456789":
        c = (c >> 8) ^ t[(c ^ b) & 0xFF]
    assert c ^ 0xFFFFFFFF == 0xE3069283 == gz.crc32c(b"123456789")


def test_names_and_dispatch(product):
    assert gz.zarr_name("x/a.georvq.parquet") == "a.georvq.zarr"
    assert gz.zarr_name("x/a.tiff") == "a.georvq.zarr"
    assert isinstance(gz.open_reader(product["zarr"]), gz.ZarrReader)
    assert isinstance(gz.open_reader(product["src"]), gp.RVQParquetReader)


def test_module_does_not_import_torch_or_zarr():
    code = ("import sys, georvq.inference.georvq_zarr; "
            "print('torch' in sys.modules, 'zarr' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == ["False", "False"]


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class _DirHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    root = None
    log = None

    def do_GET(self):
        path = os.path.join(self.root, self.path.lstrip("/"))
        if not os.path.isfile(path):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        self.log.append((self.path, rng))
        m = _RANGE.match(rng or "")
        if m.group(1) == "":
            a, b = max(0, len(data) - int(m.group(2))), len(data) - 1
        else:
            a, b = int(m.group(1)), min(int(m.group(2)), len(data) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        self.send_header("ETag", f'"{len(data)}"')
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self.wfile.write(data[a:b + 1])

    def log_message(self, *a):
        pass


@pytest.fixture
def served_dir(tmp_path):
    src = tmp_path / "a.georvq.parquet"
    _write(src, valid=_masks()["east"])
    gz.parquet_to_zarr(src, tmp_path / "a.georvq.zarr")
    gz.parquet_to_zarr(src, tmp_path / "attrs" / "a.georvq.zarr",
                       offsets_in_attrs=True)
    log = []
    h = type("H", (_DirHandler,), {"root": str(tmp_path), "log": log})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path, log
    finally:
        srv.shutdown()
        srv.server_close()


def test_http_reads_and_round_trips(served_dir):
    base, tmp, log = served_dir
    with gp.RVQParquetReader(tmp / "a.georvq.parquet") as p:
        for sub, extra in (("a.georvq.zarr", 2), ("attrs/a.georvq.zarr", 0)):
            log.clear()
            z = gz.ZarrReader(f"{base}/{sub}/")
            assert z.stats["requests"] == 1 and log[0][0].endswith("zarr.json")
            # a point at full depth: (shard indices of mid and fine), then
            # base with the mask, one mid range and one fine range
            z.read(0, 1, 0, 1, depths=FULL)
            assert z.stats["requests"] == 1 + extra + 4, log
            assert sum(1 for path, _ in log if "fine" in path) == 1 + extra // 2
            _same_reads(p, z)
            assert z._disk and z.version_tag is not None


def test_http_object_change_is_refused(served_dir):
    base, tmp, _ = served_dir
    z = gz.ZarrReader(f"{base}/a.georvq.zarr")
    p = tmp / "a.georvq.zarr" / "base" / "c" / "0" / "0" / "0"
    with open(p, "ab") as f:
        f.write(b"x")
    with pytest.raises(OSError, match="changed"):
        z.read(0, 1, 0, 1, depths=FULL)
