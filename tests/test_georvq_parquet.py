"""georvq GeoParquet files: record and mask round trip, independence
from the order blocks are added in, request counts for block, point and
whole-product reads from a file and over HTTP, and readability by plain
pyarrow."""
import http.server
import json
import os
import re
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from georvq.inference import georvq_parquet as gp
from georvq.inference.bitpack import pack

H, W = 37, 50                     # pixels
WIN, BLOCK = 4, 16                # 4 x 4 windows per block, 3 x 4 blocks
TRANSFORM = (10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
HEADER = {"codec": {"bitstream_id": "test"},
          "product": {"kind": "synthetic", "height": H, "width": W},
          "rungs": []}
CRS = {"type": "ProjectedCRS", "name": "WGS 84 / UTM zone 33N",
       "id": {"authority": "EPSG", "code": 32633}}


def _layout(quality=True):
    return gp.Layout(WIN, BLOCK, [("mean", 3, 12), ("coarse", 6, 5)],
                     [("coarse", 3, None), ("mid", 2, [("mid", 4, 9)]),
                      ("fine", 3, [("fine", 8, 11)])],
                     [("mean", 4, 8), ("p99", 4, 8)] if quality else None)


def _values(lay, seed=0):
    rng = np.random.default_rng(seed)
    gh, gw = lay.window_grid(H, W)
    return {c.name: rng.integers(0, 1 << np.array(c.widths),
                                 size=(gh, gw, c.n_values))
            for c in lay.columns.values()}


def _write(path, order="sequential", valid=None, by_tier=False, aux=None):
    lay = _layout()
    vals = _values(lay)
    n = lay.block_windows
    w = gp.RVQParquetWriter(path, HEADER, lay, "synthetic-1", TRANSFORM, H, W,
                            crs=CRS, aux=aux)
    jobs = [(i, j) for i in range(w.block_grid[0]) for j in range(w.block_grid[1])]
    if order == "shuffled":
        np.random.default_rng(7).shuffle(jobs)
    for i, j in jobs:
        sl = np.s_[i * n:(i + 1) * n, j * n:(j + 1) * n]
        rec = {k: v[sl] for k, v in vals.items()}
        if by_tier:
            for t in ("mid", "fine"):
                cols = [c for c in list(rec) if c.startswith(t + "_")]
                rec[t] = np.stack([rec.pop(c) for c in cols], 2)
        m = None if valid is None else valid[i * BLOCK:(i + 1) * BLOCK,
                                              j * BLOCK:(j + 1) * BLOCK]
        w.add_block(i, j, rec, m)
    w.close()
    return vals


@pytest.fixture
def mask():
    rng = np.random.default_rng(3)
    valid = rng.random((H, W)) > 0.1
    valid[16:32, 16:32] = True                # block (1, 1) fully valid
    return valid


def test_round_trip_and_partial_windows(tmp_path, mask):
    vals = _write(tmp_path / "a.georvq.parquet", valid=mask)
    with gp.RVQParquetReader(tmp_path / "a.georvq.parquet") as r:
        assert r.header["codec"] == HEADER["codec"]
        assert r.header["format"] == "georvq-parquet"
        assert r.header["status"] == "placeholder"
        assert r.block_grid == (3, 4) and r.window_grid == (10, 13)
        got = r.read(0, 10, 0, 13, quality=True)
        assert (got["base"] == vals["base"]).all()
        assert (got["quality"] == vals["quality"]).all()
        for t, L in (("mid", 2), ("fine", 3)):
            for l in range(L):
                assert (got[t][:, :, l] == vals[f"{t}_{l + 1}"]).all()
        assert (got["valid"][:H, :W] == mask).all()
        assert not got["valid"][H:].any() and not got["valid"][:, W:].any()
        part = r.read(3, 9, 2, 11, depths=(3, 1, 2))
        assert set(part) == {"base", "mid", "fine", "valid"}
        assert part["mid"].shape == (6, 9, 1, 4)
        assert (part["fine"][:, :, 1] == vals["fine_2"][3:9, 2:11]).all()
        assert (r.read_mask(5, 30, 11, 49) == mask[5:30, 11:49]).all()


def test_tier_arrays_and_columns_write_the_same_file(tmp_path):
    _write(tmp_path / "a.parquet")
    _write(tmp_path / "b.parquet", by_tier=True)
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()


def test_block_order_does_not_change_the_file(tmp_path, mask):
    _write(tmp_path / "a.parquet", valid=mask, aux='{"x": 1}')
    _write(tmp_path / "b.parquet", order="shuffled", valid=mask, aux='{"x": 1}')
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()


def test_column_order_and_codecs(tmp_path):
    _write(tmp_path / "a.parquet")
    md = pq.read_metadata(tmp_path / "a.parquet")
    names = [md.schema.column(k).path for k in range(md.num_columns)]
    assert names == (["product_id", "block_row", "block_col", "geometry"]
                     + [f"bbox.{k}" for k in ("xmin", "ymin", "xmax", "ymax")]
                     + ["base", "mask", "mid_1", "mid_2", "fine_1", "fine_2",
                        "fine_3", "quality"])
    assert md.num_row_groups == 12
    rg = md.row_group(5)
    assert rg.num_rows == 1
    for k, name in enumerate(names):
        cc = rg.column(k)
        assert cc.compression == "UNCOMPRESSED"
        assert not cc.has_dictionary_page
        assert cc.is_stats_set == (name in ("product_id", "block_row", "block_col")
                                   or name.startswith("bbox."))
    # column chunks of a row group are contiguous, base first
    ends = [rg.column(k).data_page_offset + rg.column(k).total_compressed_size
            for k in range(8, md.num_columns - 1)]
    starts = [rg.column(k).data_page_offset for k in range(9, md.num_columns)]
    assert ends == starts


def test_block_read_is_one_range_at_every_rung(tmp_path, mask):
    _write(tmp_path / "a.parquet", valid=mask)
    ladder = [(3, 0, 0), (3, 1, 0), (3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3)]
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=4096,
                             gap=0) as r:
        for depths in ladder:
            # quality sits after the last level: contiguous only at full depth
            for q in (False, True):
                ranges = r.byte_ranges([(1, 2)], r.columns_for(depths, quality=q))
                assert len(ranges) == (1 if not q or depths == (3, 2, 3) else 2)
            before = r.stats["requests"]
            r.read(4, 8, 8, 12, depths=depths)
            assert r.stats["requests"] - before == 1
        # a point: one window, still one range of its block
        before = r.stats["requests"]
        r.read(5, 6, 9, 10, depths=(3, 2, 3))
        assert r.stats["requests"] - before == 1


def test_whole_read_merges_and_gap_zero_does_not(tmp_path):
    _write(tmp_path / "a.parquet")
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=4096) as r:
        assert len(r.byte_ranges([(i, j) for i in range(3) for j in range(4)],
                                 r.columns_for(quality=True))) == 1
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=4096, gap=0) as r:
        assert len(r.byte_ranges([(i, j) for i in range(3) for j in range(4)],
                                 r.columns_for(quality=True))) == 12


def test_footer_in_one_read_when_small(tmp_path):
    _write(tmp_path / "a.parquet")
    size = os.path.getsize(tmp_path / "a.parquet")
    with gp.RVQParquetReader(tmp_path / "a.parquet") as r:
        assert r.stats == {"requests": 1, "bytes": size}
        r.read(0, 10, 0, 13, quality=True)
        assert r.stats["requests"] == 1          # the suffix read held it all
    footer = pq.read_metadata(tmp_path / "a.parquet").serialized_size + 8
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=1024) as r:
        assert r.stats == {"requests": 2, "bytes": max(1024, footer)}


def test_edge_blocks_are_padded_with_zero_records(tmp_path):
    vals = _write(tmp_path / "a.parquet")
    lay = _layout()
    t = pq.read_table(tmp_path / "a.parquet")
    rows = t.column("block_row").to_pylist()
    cols = t.column("block_col").to_pylist()
    k = [i for i, (a, b) in enumerate(zip(rows, cols)) if (a, b) == (2, 3)][0]
    col = lay.columns["fine_1"]
    raw = np.frombuffer(t.column("fine_1")[k].as_py(), np.uint8).reshape(16, -1)
    # block (2, 3) holds windows rows 8..9, cols 12 of the 10 x 13 grid
    want = pack(vals["fine_1"][8:10, 12:13].reshape(-1, col.n_values), col.widths)
    assert (raw.reshape(4, 4, -1)[:2, :1].reshape(2, -1) == want).all()
    assert not raw.reshape(4, 4, -1)[2:].any() and not raw.reshape(4, 4, -1)[:, 1:].any()
    assert t.column("mask")[k].as_py() == b""


def test_plain_pyarrow_reads_the_file(tmp_path, mask):
    vals = _write(tmp_path / "a.parquet", valid=mask, aux={"tags": {"a": "b"}})
    t = pq.read_table(tmp_path / "a.parquet")
    assert t.num_rows == 12
    assert set(t.column("product_id").to_pylist()) == {"synthetic-1"}
    lay = _layout()
    col = lay.columns["base"]
    k = 4 * 1 + 2                                 # block (1, 2)
    assert (t.column("block_row")[k].as_py(), t.column("block_col")[k].as_py()) == (1, 2)
    want = pack(vals["base"][4:8, 8:12].reshape(-1, col.n_values), col.widths)
    assert t.column("base")[k].as_py() == want.tobytes()
    # geometry is WKB of the block footprint, and agrees with bbox
    g = t.column("geometry")[k].as_py()
    order, kind, nring, npts = struct.unpack("<BIII", g[:13])
    assert (order, kind, nring, npts) == (1, 3, 1, 5)
    xy = np.frombuffer(g[13:], "<f8").reshape(5, 2)
    bb = t.column("bbox")[k].as_py()
    assert (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()) == \
        (bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"])
    assert bb == {"xmin": 500000.0 + 32 * 10, "xmax": 500000.0 + 48 * 10,
                  "ymax": 4000000.0 - 16 * 10, "ymin": 4000000.0 - 32 * 10}
    # edge blocks are clipped to the product
    last = t.column("bbox")[11].as_py()
    assert last["xmax"] == 500000.0 + W * 10 and last["ymin"] == 4000000.0 - H * 10
    md = pq.read_metadata(tmp_path / "a.parquet").metadata
    assert json.loads(md[b"georvq_aux"]) == {"tags": {"a": "b"}}


def test_geo_metadata(tmp_path):
    _write(tmp_path / "a.parquet")
    md = pq.read_metadata(tmp_path / "a.parquet").metadata
    geo = json.loads(md[b"geo"])
    assert geo["version"] == "1.1.0" and geo["primary_column"] == "geometry"
    g = geo["columns"]["geometry"]
    assert g["encoding"] == "WKB" and g["geometry_types"] == ["Polygon"]
    assert g["crs"] == CRS
    assert g["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
    assert g["bbox"] == [500000.0, 4000000.0 - H * 10, 500000.0 + W * 10, 4000000.0]
    hdr = json.loads(md[b"georvq"])
    assert hdr["version"] == "1.0"
    lay = gp.Layout.from_json(hdr["layout"])
    assert lay.to_json() == _layout().to_json()
    assert hdr["layout"]["block_grid"] == [3, 4]


def test_writer_checks(tmp_path):
    lay = _layout()
    with pytest.raises(ValueError, match="set by the writer"):
        gp.RVQParquetWriter(tmp_path / "a", dict(HEADER, layout={}), lay, "p",
                            TRANSFORM, H, W)
    w = gp.RVQParquetWriter(tmp_path / "a", HEADER, lay, "p", TRANSFORM, H, W)
    with pytest.raises(ValueError, match="outside"):
        w.add_block(3, 0, {})
    with pytest.raises(ValueError, match="no column"):
        w.add_block(0, 0, {"nope": np.zeros((4, 4, 1))})
    with pytest.raises(ValueError, match="mask"):
        w.add_block(2, 3, {}, np.ones((16, 16), bool))
    w.add_block(2, 3, {}, np.ones((5, 2), bool))
    with pytest.raises(ValueError, match="not added"):
        w.close()


def test_refuses_other_major_version(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "MAJOR", 2)
    _write(tmp_path / "a.parquet")
    monkeypatch.setattr(gp, "MAJOR", 1)
    with pytest.raises(ValueError, match="major version 2"):
        gp.RVQParquetReader(tmp_path / "a.parquet")


def test_tier_names():
    assert gp.tier_names(3) == ["coarse", "mid", "fine"]
    assert gp.tier_names(2) == ["tier0", "tier1"]
    assert gp.output_name("/a/b/tile_1.tif") == "tile_1.georvq.parquet"


_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Single Range requests, including suffix ranges (bytes=-n)."""
    protocol_version = "HTTP/1.1"
    path_on_disk = None
    requests = None

    def do_GET(self):
        data = Path(self.path_on_disk).read_bytes()
        rng = self.headers.get("Range")
        type(self).requests.append(rng)
        if rng is None:
            a, b, status = 0, len(data) - 1, 200
        else:
            m = _RANGE.match(rng)
            if m.group(1) == "":
                a, b = max(0, len(data) - int(m.group(2))), len(data) - 1
            else:
                a, b = int(m.group(1)), min(int(m.group(2)), len(data) - 1)
            status = 206
        body = data[a:b + 1]
        self.send_response(status)
        if status == 206:
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def served(tmp_path, mask):
    vals = _write(tmp_path / "a.georvq.parquet", valid=mask)
    handler = type("H", (_RangeHandler,), {
        "path_on_disk": str(tmp_path / "a.georvq.parquet"), "requests": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (f"http://127.0.0.1:{server.server_address[1]}/a.georvq.parquet",
               handler, vals)
    finally:
        server.shutdown()
        server.server_close()


def test_http_reads(served, mask):
    url, handler, vals = served
    r = gp.RVQParquetReader(url, first_read=4096)
    footer = r.metadata.serialized_size + 8
    assert handler.requests[0] == "bytes=-4096"
    assert len(handler.requests) == (1 if footer <= 4096 else 2)
    n0 = len(handler.requests)
    got = r.read(4, 8, 8, 12, depths=(3, 2, 1))           # block (1, 2)
    assert len(handler.requests) == n0 + 1
    assert (got["fine"][:, :, 0] == vals["fine_1"][4:8, 8:12]).all()
    assert (got["valid"] == mask[16:32, 32:48]).all()
    got = r.read(0, 10, 0, 13, quality=True)              # whole product
    assert len(handler.requests) == n0 + 2
    assert (got["fine"][:, :, 2] == vals["fine_3"]).all()
    assert (got["valid"][:H, :W] == mask).all()
    assert r.stats["requests"] == len(handler.requests)
    assert all(h.startswith("bytes=") for h in handler.requests)

    r = gp.RVQParquetReader(url, first_read=4096, gap=0)
    n0 = len(handler.requests)
    r.read(0, 10, 0, 13, depths=(3, 0, 0))
    # base and mask of each block are one range; blocks are separated by
    # the levels not read
    assert len(handler.requests) - n0 == 12

    n0 = len(handler.requests)
    r = gp.RVQParquetReader(url)
    assert handler.requests[n0:] == [f"bytes=-{gp.FIRST_READ}"]


def test_module_does_not_import_torch():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, georvq.inference.georvq_parquet; print('torch' in sys.modules)"],
        cwd=root, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
