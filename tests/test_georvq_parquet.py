"""georvq GeoParquet files: record and mask round trip with per-tier
chunk sizes, independence from the order windows are added in, the row
group and column layout, readability by plain pyarrow (and DuckDB when
installed), range planning and merging, parallel fetch over HTTP, and
the record cache."""
import http.server
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from georvq.inference import georvq_parquet as gp
from georvq.inference.bitpack import pack

H, W = 37, 50                     # pixels
WIN = 4                           # 10 x 13 windows
CHUNK = {"base": 64, "mid": 16, "fine": 8}
# base and quality: 1 x 1 chunks; mid: 3 x 4; fine: 5 x 7, all with
# partial edge chunks
TRANSFORM = (10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
HEADER = {"codec": {"bitstream_id": "test"},
          "product": {"kind": "synthetic", "height": H, "width": W},
          "rungs": []}
CRS = {"type": "ProjectedCRS", "name": "WGS 84 / UTM zone 33N",
       "id": {"authority": "EPSG", "code": 32633}}
FULL = (3, 2, 3)


def _layout(quality=True, window=WIN, chunk=CHUNK):
    return gp.Layout(window, chunk, [("mean", 3, 12), ("coarse", 6, 5)],
                     [("coarse", 3, None), ("mid", 2, [("mid", 4, 9)]),
                      ("fine", 3, [("fine", 8, 11)])],
                     [("mean", 4, 8), ("p99", 4, 8)] if quality else None)


def _values(lay, h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    gh, gw = lay.window_grid(h, w)
    return {c.name: rng.integers(0, 1 << np.array(c.widths),
                                 size=(gh, gw, c.n_values))
            for c in lay.columns.values()}


def _write(path, order="sequential", valid=None, by_tier=False, aux=None,
           lay=None, h=H, w=W, step=3):
    """Writes the product in window strips of `step` rows by `step + 2`
    columns, in the given order."""
    lay = lay or _layout()
    vals = _values(lay, h, w)
    header = dict(HEADER, product=dict(HEADER["product"], height=h, width=w))
    wr = gp.RVQParquetWriter(path, header, lay, "synthetic-1", TRANSFORM, h, w,
                             crs=CRS, aux=aux)
    gh, gw = wr.window_grid
    jobs = [(i, j) for i in range(0, gh, step) for j in range(0, gw, step + 2)]
    if order == "shuffled":
        np.random.default_rng(7).shuffle(jobs)
    P = lay.window_px
    for i, j in jobs:
        sl = np.s_[i:i + step, j:j + step + 2]
        rec = {k: v[sl] for k, v in vals.items()}
        if by_tier:
            for t in ("mid", "fine"):
                cols = [c for c in list(rec) if c.startswith(t + "_")]
                rec[t] = np.stack([rec.pop(c) for c in cols], 2)
        m = None if valid is None else valid[i * P:(i + step) * P,
                                              j * P:(j + step + 2) * P]
        wr.add_windows(i, j, rec, m)
    wr.close()
    return vals


@pytest.fixture
def mask():
    rng = np.random.default_rng(3)
    valid = rng.random((H, W)) > 0.1
    valid[8:24, 8:24] = True
    return valid


def test_round_trip_and_partial_windows(tmp_path, mask):
    vals = _write(tmp_path / "a.georvq.parquet", valid=mask)
    with gp.RVQParquetReader(tmp_path / "a.georvq.parquet") as r:
        assert r.header["codec"] == HEADER["codec"]
        assert r.header["format"] == "georvq-parquet"
        assert r.header["status"] == "placeholder"
        assert r.window_grid == (10, 13)
        assert r.chunk_grids == {"base": (1, 1), "mid": (3, 4), "fine": (5, 7),
                                 "quality": (1, 1)}
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
        assert (part["base"] == vals["base"][3:9, 2:11]).all()
        assert (r.read_mask(5, 30, 11, 49) == mask[5:30, 11:49]).all()
        assert r.masked == {(0, 0)}                # the one base chunk


def test_window_size_not_a_power_of_two(tmp_path):
    # 96 px windows, 960 px base and mid chunks, 480 px fine chunks
    h, w = 2000, 1500
    lay = _layout(window=96, chunk={"base": 1920, "mid": 960, "fine": 480})
    valid = np.ones((h, w), bool)
    valid[1990:, 1450:] = False
    vals = _write(tmp_path / "a.parquet", lay=lay, h=h, w=w, valid=valid, step=4)
    with gp.RVQParquetReader(tmp_path / "a.parquet") as r:
        assert r.window_grid == (21, 16)
        assert r.chunk_grids == {"base": (2, 1), "mid": (3, 2), "fine": (5, 4),
                                 "quality": (2, 1)}
        assert r.masked == {(1, 0)}
        got = r.read(0, 21, 0, 16, quality=True)
        for k, v in vals.items():
            t, _, l = k.partition("_")
            g = got[t][:, :, int(l) - 1] if l else got[k]
            assert (g == v).all(), k
        assert (got["valid"][:h, :w] == valid).all()
        pt = r.read(20, 21, 15, 16)
        assert (pt["fine"][0, 0, 2] == vals["fine_3"][20, 15]).all()
    t = pq.read_table(tmp_path / "a.parquet", filters=[("kind", "=", "mid")])
    n = 10                                     # windows per mid chunk side
    k = [i for i, (a, b) in enumerate(zip(t.column("chunk_row").to_pylist(),
                                            t.column("chunk_col").to_pylist()))
         if (a, b) == (2, 1)][0]
    col = lay.columns["mid_1"]
    raw = np.frombuffer(t.column("mid_1")[k].as_py(), np.uint8).reshape(n, n, -1)
    # mid chunk (2, 1) holds window rows 20, cols 10..15 of the 21 x 16 grid
    want = pack(vals["mid_1"][20:21, 10:16].reshape(-1, col.n_values), col.widths)
    assert (raw[:1, :6].reshape(6, -1) == want).all()
    assert not raw[1:].any() and not raw[:, 6:].any()


def test_mask_in_a_tier_kind(tmp_path, mask):
    lay = gp.Layout(WIN, CHUNK, [("mean", 3, 12), ("coarse", 6, 5)],
                    [("coarse", 3, None), ("mid", 2, [("mid", 4, 9)]),
                     ("fine", 3, [("fine", 8, 11)])], mask_kind="fine")
    assert lay.column_order == ["base", "mid_1", "mid_2", "mask", "fine_1",
                                "fine_2", "fine_3"]
    _write(tmp_path / "a.parquet", valid=mask, lay=lay)
    with gp.RVQParquetReader(tmp_path / "a.parquet") as r:
        assert r.layout.mask_kind == "fine" and (1, 1) not in r.masked
        assert (r.read_mask(0, H, 0, W) == mask).all()
        cols = r.columns_for((3, 2, 0))
        assert ("mask", 2, 4) in r.needs(5, 6, 9, 10, cols)
        # a fully valid fine chunk has no mask to fetch
        assert not any(n[0] == "mask" for n in r.needs(2, 3, 2, 3, cols))
    with pytest.raises(ValueError, match="mask_kind"):
        gp.Layout(WIN, 8, [("a", 1, 1)], [("t", 1, None)], mask_kind="mid")


def test_tier_arrays_and_columns_write_the_same_file(tmp_path):
    _write(tmp_path / "a.parquet")
    _write(tmp_path / "b.parquet", by_tier=True)
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()


def test_add_order_does_not_change_the_file(tmp_path, mask):
    _write(tmp_path / "a.parquet", valid=mask, aux='{"x": 1}')
    _write(tmp_path / "b.parquet", order="shuffled", valid=mask, aux='{"x": 1}',
           step=2)
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()


def test_row_groups_columns_and_codecs(tmp_path, mask):
    _write(tmp_path / "a.parquet", valid=mask)
    md = pq.read_metadata(tmp_path / "a.parquet")
    names = [md.schema.column(k).path for k in range(md.num_columns)]
    data = ["base", "mask", "mid_1", "mid_2", "fine_1", "fine_2", "fine_3",
            "quality"]
    assert names == (["kind", "product_id", "chunk_row", "chunk_col", "geometry"]
                     + [f"bbox.{k}" for k in ("xmin", "ymin", "xmax", "ymax")]
                     + data)
    assert md.num_row_groups == 1 + 12 + 35 + 1
    t = pq.read_table(tmp_path / "a.parquet")
    kinds = t.column("kind").to_pylist()
    assert kinds == ["base"] + ["mid"] * 12 + ["fine"] * 35 + ["quality"]
    pos = list(zip(t.column("chunk_row").to_pylist(), t.column("chunk_col").to_pylist()))
    assert pos[1:13] == [(i, j) for i in range(3) for j in range(4)]
    assert pos[13:48] == [(i, j) for i in range(5) for j in range(7)]
    own = {"base": ["base", "mask"], "mid": ["mid_1", "mid_2"],
           "fine": ["fine_1", "fine_2", "fine_3"], "quality": ["quality"]}
    for c in data:
        nulls = [v is None for v in t.column(c).to_pylist()]
        assert nulls == [c not in own[k] for k in kinds], c
    hdr = json.loads(md.metadata[b"georvq"])
    first = {g["kind"]: g["row_groups"] for g in hdr["layout"]["groups"]}
    assert first == {"base": [0, 1], "mid": [1, 12], "fine": [13, 35],
                     "quality": [48, 1]}
    for i in (0, 5, 20, 48):
        rg = md.row_group(i)
        assert rg.num_rows == 1
        for k, name in enumerate(names):
            cc = rg.column(k)
            assert cc.compression == "UNCOMPRESSED"
            assert not cc.has_dictionary_page
            assert cc.is_stats_set == (name in ("kind", "product_id", "chunk_row",
                                                "chunk_col")
                                       or name.startswith("bbox.")), name
    # the mask follows base directly, and a fine chunk's levels are adjacent
    for i, k in ((0, names.index("base")), (20, names.index("fine_1"))):
        rg = md.row_group(i)
        ends = [rg.column(j).data_page_offset + rg.column(j).total_compressed_size
                for j in range(k, k + 2)]
        assert ends == [rg.column(j).data_page_offset for j in range(k + 1, k + 3)]


def test_plain_pyarrow_reads_the_file(tmp_path, mask):
    vals = _write(tmp_path / "a.parquet", valid=mask, aux={"tags": {"a": "b"}})
    fine = pq.read_table(tmp_path / "a.parquet", filters=[("kind", "=", "fine")],
                         columns=["chunk_row", "chunk_col", "mask", "fine_1"])
    assert set(fine.column("mask").to_pylist()) == {None}
    assert fine.num_rows == 35
    lay = _layout()
    col = lay.columns["fine_1"]
    k = 7 * 1 + 2                               # fine chunk (1, 2)
    assert (fine.column("chunk_row")[k].as_py(), fine.column("chunk_col")[k].as_py()) == (1, 2)
    want = pack(vals["fine_1"][2:4, 4:6].reshape(-1, col.n_values), col.widths)
    assert fine.column("fine_1")[k].as_py() == want.tobytes()
    t = pq.read_table(tmp_path / "a.parquet")
    assert set(t.column("product_id").to_pylist()) == {"synthetic-1"}
    base = t.column("base")[0].as_py()
    bc = lay.columns["base"]
    raw = np.frombuffer(base, np.uint8).reshape(16, 16, -1)
    assert (raw[:10, :13].reshape(130, -1)
            == pack(vals["base"].reshape(130, -1), bc.widths)).all()
    assert not raw[10:].any() and not raw[:, 13:].any()
    assert (gp.decode_mask(t.column("mask")[0].as_py(), (H, W)) == mask).all()
    # geometry is WKB of the chunk footprint, and agrees with bbox
    k = 1 + 4 * 1 + 2                           # mid chunk (1, 2)
    g = t.column("geometry")[k].as_py()
    order, kind, nring, npts = struct.unpack("<BIII", g[:13])
    assert (order, kind, nring, npts) == (1, 3, 1, 5)
    xy = np.frombuffer(g[13:], "<f8").reshape(5, 2)
    bb = t.column("bbox")[k].as_py()
    assert (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()) == \
        (bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"])
    assert bb == {"xmin": 500000.0 + 32 * 10, "xmax": 500000.0 + 48 * 10,
                  "ymax": 4000000.0 - 16 * 10, "ymin": 4000000.0 - 32 * 10}
    # edge chunks, and the whole-product base chunk, are clipped
    for k in (0, 12):
        bb = t.column("bbox")[k].as_py()
        assert bb["xmax"] == 500000.0 + W * 10 and bb["ymin"] == 4000000.0 - H * 10
    md = pq.read_metadata(tmp_path / "a.parquet").metadata
    assert json.loads(md[b"georvq_aux"]) == {"tags": {"a": "b"}}


def test_duckdb_reads_the_file(tmp_path, mask):
    duckdb = pytest.importorskip("duckdb")
    vals = _write(tmp_path / "a.parquet", valid=mask)
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT kind, count(*) FROM '{tmp_path / 'a.parquet'}' GROUP BY kind "
        "ORDER BY kind").fetchall()
    assert rows == [("base", 1), ("fine", 35), ("mid", 12), ("quality", 1)]
    v = con.execute(
        f"SELECT fine_1 FROM '{tmp_path / 'a.parquet'}' WHERE kind = 'fine' "
        "AND chunk_row = 1 AND chunk_col = 2").fetchone()[0]
    col = _layout().columns["fine_1"]
    assert bytes(v) == pack(vals["fine_1"][2:4, 4:6].reshape(-1, col.n_values),
                            col.widths).tobytes()


def test_geo_metadata_and_header(tmp_path, mask):
    _write(tmp_path / "a.parquet", valid=mask)
    md = pq.read_metadata(tmp_path / "a.parquet").metadata
    geo = json.loads(md[b"geo"])
    assert geo["version"] == "1.1.0" and geo["primary_column"] == "geometry"
    g = geo["columns"]["geometry"]
    assert g["encoding"] == "WKB" and g["geometry_types"] == ["Polygon"]
    assert g["crs"] == CRS
    assert g["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
    assert g["bbox"] == [500000.0, 4000000.0 - H * 10, 500000.0 + W * 10, 4000000.0]
    hdr = json.loads(md[b"georvq"])
    assert hdr["version"] == "1.1"
    lay = gp.Layout.from_json(hdr["layout"])
    assert lay.to_json() == _layout().to_json()
    groups = {x["kind"]: x for x in hdr["layout"]["groups"]}
    assert [x["kind"] for x in hdr["layout"]["groups"]] == ["base", "mid", "fine",
                                                            "quality"]
    assert groups["mid"]["chunk_px"] == 16 and groups["mid"]["chunk_windows"] == 4
    assert groups["base"]["columns"] == ["base", "mask"]
    assert groups["fine"]["columns"] == ["fine_1", "fine_2", "fine_3"]
    assert groups["fine"]["chunk_grid"] == [5, 7]
    assert hdr["layout"]["window_grid"] == [10, 13]
    m = hdr["layout"]["mask"]
    assert m["kind"] == "base" and m["codec"] == "packbits-zlib"
    assert m["chunks"] == [[0, 0]]


# --------------------------------------------------------------------------
# empty chunks
# --------------------------------------------------------------------------

@pytest.fixture
def east_nodata():
    """Valid west of column 24 only: window columns 6..12 have no valid
    pixel, so fine chunk columns 3..6 and mid chunk columns 2..3 are
    empty, and mid chunk column 1 is partial."""
    valid = np.ones((H, W), bool)
    valid[:, 24:] = False
    return valid


def test_empty_chunks_are_null(tmp_path, east_nodata):
    vals = _write(tmp_path / "a.parquet", valid=east_nodata)
    _write(tmp_path / "dense.parquet")
    size = os.path.getsize(tmp_path / "a.parquet")
    dense = os.path.getsize(tmp_path / "dense.parquet")
    fine_bytes = sum(_layout().columns[f"fine_{l}"].record_bytes
                     for l in (1, 2, 3)) * 4
    mid_bytes = sum(_layout().columns[f"mid_{l}"].record_bytes
                    for l in (1, 2)) * 16
    assert dense - size >= 20 * fine_bytes + 6 * mid_bytes
    md = pq.read_metadata(tmp_path / "a.parquet")
    hdr = json.loads(md.metadata[b"georvq"])
    empty = {g["kind"]: g["empty_chunks"] for g in hdr["layout"]["groups"]}
    assert empty == {"base": [], "quality": [],
                     "mid": [[i, j] for i in range(3) for j in (2, 3)],
                     "fine": [[i, j] for i in range(5) for j in range(3, 7)]}
    t = pq.read_table(tmp_path / "a.parquet", filters=[("kind", "=", "fine")])
    for row in t.to_pylist():
        null = row["chunk_col"] >= 3
        assert all((row[f"fine_{l}"] is None) == null for l in (1, 2, 3))
    t = pq.read_table(tmp_path / "a.parquet", filters=[("kind", "=", "mid")])
    assert [r["mid_1"] is None for r in t.to_pylist()] == [
        j >= 2 for i in range(3) for j in range(4)]
    with gp.RVQParquetReader(tmp_path / "a.parquet") as r:
        assert r.empty["fine"] == {(i, j) for i in range(5) for j in range(3, 7)}
        assert r.dead[:, 6:].all() and not r.dead[:, :6].any()
        got = r.read(0, 10, 0, 13, quality=True)
        live = np.zeros((10, 13), bool)
        live[:, :6] = True
        for k, v in vals.items():
            t, _, l = k.partition("_")
            g = got[t][:, :, int(l) - 1] if l else got[k]
            assert (g[live] == v[live]).all(), k
        assert not got["fine"][:, 6:].any() and not got["mid"][:, 8:].any()
        # windows 6 and 7 are dead in the partial mid chunk column 1, and
        # keep the records they were given
        assert (got["mid"][:, 6:8, 0] == vals["mid_1"][:, 6:8]).all()
        assert (got["valid"][:H, :W] == east_nodata).all()


def test_reads_over_empty_chunks_fetch_nothing(tmp_path, east_nodata):
    _write(tmp_path / "a.parquet", valid=east_nodata)
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=1024) as r:
        # windows (2..5, 7..12) lie in empty fine chunks; base, mid and the
        # mask are not fetched for them either
        assert r.plan(2, 5, 7, 13, depths=FULL, quality=True) == []
        before = dict(r.stats)
        got = r.read(2, 5, 7, 13, depths=FULL, quality=True)
        assert r.stats == before
        assert not got["valid"].any()
        assert not got["base"].any() and not got["fine"].any()
        # a region touching one live window fetches its chunks only
        needs = r.needs(2, 3, 5, 7, r.columns_for(FULL))
        assert {(n[0], n[1], n[2]) for n in needs if n[0].startswith("fine")} \
            == {(f"fine_{l}", 1, 2) for l in (1, 2, 3)}
        # a whole-product read is still one range
        assert len(r.plan(0, 10, 0, 13, depths=FULL, quality=True)) == 1


def test_wholly_invalid_product(tmp_path):
    _write(tmp_path / "a.parquet", valid=np.zeros((H, W), bool))
    t = pq.read_table(tmp_path / "a.parquet")
    for c in _layout().column_order:
        assert set(t.column(c).to_pylist()) == {None}, c
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=1024) as r:
        assert r.dead.all() and r.plan(0, 10, 0, 13, quality=True) == []
        got = r.read(0, 10, 0, 13, quality=True)
        assert not got["valid"].any() and not got["base"].any()
        assert not r.read_mask(0, H, 0, W).any()


# --------------------------------------------------------------------------
# range planning
# --------------------------------------------------------------------------

def test_point_read_is_one_range_per_tier(tmp_path, mask):
    _write(tmp_path / "a.parquet", valid=mask)
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=4096,
                             gap=0) as r:
        # (5, 9) lies in mid chunk (1, 2) and fine chunk (2, 4)
        assert len(r.plan(5, 6, 9, 10, depths=FULL)) == 3
        assert len(r.plan(5, 6, 9, 10, depths=(3, 2, 0))) == 2   # mask with base
        assert len(r.plan(5, 6, 9, 10, depths=(3, 0, 0))) == 1
        for depths in [(3, 0, 1), (3, 1, 2), FULL]:
            before = r.stats["requests"]
            r.read(5, 6, 9, 10, depths=depths)
            assert r.stats["requests"] - before <= 3


def test_whole_reads_merge_across_row_groups(tmp_path, mask):
    _write(tmp_path / "a.parquet", valid=mask)
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=4096) as r:
        assert len(r.plan(0, 10, 0, 13, depths=FULL)) == 1
        assert len(r.plan(0, 10, 0, 13, depths=FULL, quality=True)) == 1
        # base followed by every mid chunk at full mid depth
        assert len(r.plan(0, 10, 0, 13, depths=(3, 2, 0), mask=False)) == 1
        # every fine chunk at full depth
        assert len(r.plan(0, 10, 0, 13, depths=FULL, base=False)) == 1
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=4096, gap=0) as r:
        # partial mid depth: the base chunk, then one range per mid chunk
        assert len(r.plan(0, 10, 0, 13, depths=(3, 1, 0), mask=False)) == 1 + 12
        # row groups are separated by their position and footprint columns
        assert len(r.plan(0, 10, 0, 13, depths=(3, 0, 3), base=False,
                          mask=False)) == 35
        before = r.stats["requests"]
        r.read(0, 10, 0, 13, depths=(3, 1, 0), mask=False)
        assert r.stats["requests"] - before == 13
    assert gp.merge_ranges([(0, 10), (10, 5), (40, 5), (20, 2)], gap=5) == [
        (0, 22), (40, 5)]


def test_footer_in_one_read_when_small(tmp_path):
    _write(tmp_path / "a.parquet")
    size = os.path.getsize(tmp_path / "a.parquet")
    with gp.RVQParquetReader(tmp_path / "a.parquet") as r:
        assert r.stats == {"requests": 1, "bytes": size, "retries": 0,
                           "cache_hits": 0, "cache_bytes": 0}
        r.read(0, 10, 0, 13, quality=True)
        assert r.stats["requests"] == 1          # the suffix read held it all
    footer = pq.read_metadata(tmp_path / "a.parquet").serialized_size + 8
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=1024) as r:
        assert (r.stats["requests"], r.stats["bytes"]) == (2, max(1024, footer))


def test_writer_checks(tmp_path):
    lay = _layout()
    with pytest.raises(ValueError, match="set by the writer"):
        gp.RVQParquetWriter(tmp_path / "a", dict(HEADER, layout={}), lay, "p",
                            TRANSFORM, H, W)
    with pytest.raises(ValueError, match="multiple of window_px"):
        _layout(chunk={"base": 64, "mid": 18, "fine": 8})
    with pytest.raises(ValueError, match="chunk_px keys"):
        _layout(chunk={"base": 64, "mid": 16})
    w = gp.RVQParquetWriter(tmp_path / "a", HEADER, lay, "p", TRANSFORM, H, W)
    z = np.zeros((2, 2, 4), int)
    with pytest.raises(ValueError, match="outside"):
        w.add_windows(9, 0, {"mid_1": z})
    with pytest.raises(ValueError, match="no column"):
        w.add_windows(0, 0, {"nope": z})
    with pytest.raises(ValueError, match="mask"):
        w.add_windows(8, 12, {"mid_1": z[:, :1]}, np.ones((8, 4), bool))
    w.add_windows(8, 12, {"mid_1": z[:, :1]}, np.ones((5, 2), bool))
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


# --------------------------------------------------------------------------
# http, parallel fetch and the cache
# --------------------------------------------------------------------------

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Single Range requests, including suffix ranges (bytes=-n), with an
    ETag and a per-request delay; records the requests and the most
    requests in flight at once. Faults, set in `state`: `fail`, statuses
    to answer the next requests with; `stall`, seconds to wait before
    answering the next request; `ignore_range`, answer 200 with the whole
    file; `shift`, move the answered range by that many bytes (range
    requests only, or every request when `shift_suffix`); `range_etag`
    False, no ETag on range answers."""
    protocol_version = "HTTP/1.1"
    path_on_disk = None
    state = None

    def do_GET(self):
        st = type(self).state
        with st["lock"]:
            st["active"] += 1
            st["max_active"] = max(st["max_active"], st["active"])
        try:
            time.sleep(st["delay"])
            with st["lock"]:
                stall, st["stall"] = st.get("stall", 0), 0
                fail = st["fail"].pop(0) if st.get("fail") else None
            time.sleep(stall)
            data = Path(self.path_on_disk).read_bytes()
            rng = self.headers.get("Range")
            st["requests"].append(rng)
            if fail is not None:
                self.send_response(fail)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            suffix = False
            if rng is None or st.get("ignore_range"):
                a, b, status = 0, len(data) - 1, 200
            else:
                m = _RANGE.match(rng)
                suffix = m.group(1) == ""
                if suffix:
                    a, b = max(0, len(data) - int(m.group(2))), len(data) - 1
                else:
                    a, b = int(m.group(1)), min(int(m.group(2)), len(data) - 1)
                if st.get("shift") and (not suffix or st.get("shift_suffix")):
                    a, b = a - st["shift"], b - st["shift"]
                status = 206
            body = data[a:b + 1]
            self.send_response(status)
            if status == 206:
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            if st["etag"] and (suffix or st.get("range_etag", True)):
                self.send_header("ETag", st["etag"])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with st["lock"]:
                st["active"] -= 1

    def log_message(self, *args):
        pass


@pytest.fixture
def served(tmp_path, mask):
    vals = _write(tmp_path / "a.georvq.parquet", valid=mask)
    state = {"lock": threading.Lock(), "active": 0, "max_active": 0,
             "delay": 0.0, "etag": '"v1"', "requests": [], "fail": []}
    handler = type("H", (_RangeHandler,), {
        "path_on_disk": str(tmp_path / "a.georvq.parquet"), "state": state})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (f"http://127.0.0.1:{server.server_address[1]}/a.georvq.parquet",
               state, vals)
    finally:
        server.shutdown()
        server.server_close()


def test_http_reads(served, mask):
    url, state, vals = served
    r = gp.RVQParquetReader(url, first_read=4096)
    footer = r.metadata.serialized_size + 8
    assert state["requests"][0] == "bytes=-4096"
    assert len(state["requests"]) == (1 if footer <= 4096 else 2)
    assert r.version_tag == '"v1"'
    plan = r.plan(4, 8, 8, 12, depths=(3, 2, 1))
    n0 = len(state["requests"])
    got = r.read(4, 8, 8, 12, depths=(3, 2, 1))
    assert len(state["requests"]) - n0 == len(plan) > 0
    assert (got["fine"][:, :, 0] == vals["fine_1"][4:8, 8:12]).all()
    assert (got["valid"] == mask[16:32, 32:48]).all()
    got = r.read(0, 10, 0, 13, quality=True)
    assert (got["fine"][:, :, 2] == vals["fine_3"]).all()
    assert (got["valid"][:H, :W] == mask).all()
    assert r.stats["requests"] == len(state["requests"])
    assert all(h.startswith("bytes=") for h in state["requests"])
    n0 = len(state["requests"])
    gp.RVQParquetReader(url)
    assert state["requests"][n0:] == [f"bytes=-{gp.FIRST_READ}"]


def test_parallel_fetch_matches_sequential(served):
    url, state, vals = served
    state["delay"] = 0.05
    out = {}
    for conc in (1, 16):
        state["max_active"] = 0
        r = gp.RVQParquetReader(url, first_read=4096, gap=0, concurrency=conc)
        ranges = r.plan(0, 10, 0, 13, depths=(3, 1, 0))
        n0 = len(state["requests"])
        out[conc] = r.read(0, 10, 0, 13, depths=(3, 1, 0))
        assert len(state["requests"]) - n0 == len(ranges) > 10
        out[conc]["max_active"] = state["max_active"]
    assert out[1].pop("max_active") == 1
    assert out[16].pop("max_active") > 1
    for k in out[1]:
        assert np.array_equal(out[1][k], out[16][k])
    assert (out[16]["mid"][:, :, 0] == vals["mid_1"]).all()


def test_trace_records_requests_and_their_overlap(served):
    url, state, vals = served
    state["delay"] = 0.05
    plain = gp.RVQParquetReader(url, first_read=4096, gap=0)
    tr = gp.Trace()
    r = gp.RVQParquetReader(url, first_read=4096, gap=0, trace=tr)
    assert plain.trace is None
    n_open = len(tr.events)
    got = r.read(0, 10, 0, 13, depths=(3, 1, 0))
    ref = plain.read(0, 10, 0, 13, depths=(3, 1, 0))
    for k in ref:
        assert np.array_equal(got[k], ref[k])
    gets = [e for e in tr.events if e["kind"] == "get"]
    assert len(gets) == r.stats["requests"]
    assert sum(e["bytes"] for e in gets) == r.stats["bytes"]
    assert gets[0]["label"] == "suffix"
    assert {"parse footer", "plan ranges", "parse pages", "unpack records"} <= {
        e["label"] for e in tr.events if e["kind"] == "cpu"}
    assert all(e["t1"] >= e["t0"] for e in tr.events)
    # the parallel requests of one read overlap in time
    reads = sorted((e for e in gets[n_open:] if e["label"] != "suffix"),
                   key=lambda e: e["t0"])
    assert len(reads) > 1 and reads[1]["t0"] < reads[0]["t1"]


def test_cache_serves_later_reads(served):
    url, state, vals = served
    r = gp.RVQParquetReader(url, first_read=4096)
    base = r.chunk_range("base", 0, 0)
    # whole-product base, then a window: no base bytes are fetched again
    r.read(0, 10, 0, 13, depths=(3, 0, 0), mask=False)
    plan = r.plan(5, 7, 9, 11, depths=FULL)
    assert plan and not any(o < base[0] + base[1] and base[0] < o + n
                            for o, n in plan)
    before = dict(r.stats)
    got = r.read(5, 7, 9, 11, depths=FULL)
    assert r.stats["bytes"] - before["bytes"] == sum(n for _, n in plan)
    assert r.stats["cache_hits"] - before["cache_hits"] == 1
    assert (got["base"] == vals["base"][5:7, 9:11]).all()
    # the same read again fetches nothing
    before = dict(r.stats)
    again = r.read(5, 7, 9, 11, depths=FULL)
    assert r.stats["requests"] == before["requests"]
    assert r.stats["cache_hits"] > before["cache_hits"]
    for k in got:
        assert np.array_equal(got[k], again[k])
    # a reader sharing the cache fetches nothing for the same file
    r2 = gp.RVQParquetReader(url, first_read=4096, cache=r.cache)
    assert r2.file_key == r.file_key and r2.plan(5, 7, 9, 11, depths=FULL) == []
    # a new ETag is a new file: its values are not taken from the cache
    state["etag"] = '"v2"'
    r3 = gp.RVQParquetReader(url, first_read=4096, cache=r.cache)
    assert r3.file_key != r.file_key
    assert len(r3.plan(5, 7, 9, 11, depths=FULL)) > 0
    # and a reader opened before the change refuses to mix versions
    with pytest.raises(OSError, match="changed"):
        r2.read(0, 1, 0, 1, depths=(3, 0, 0), mask=False, quality=True)


def test_disk_cache_and_local_identity(tmp_path, mask):
    vals = _write(tmp_path / "a.parquet", valid=mask)
    cdir = tmp_path / "cache"
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=4096,
                             cache_dir=cdir) as r:
        r.read(0, 10, 0, 13, depths=FULL)
        key = r.file_key
    files = list(cdir.rglob("*.bin"))
    assert files and all(f.parent == cdir / "test" / key[1] for f in files)
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=4096,
                             cache_dir=cdir) as r:
        assert r.file_key == key
        opened = r.stats["requests"]
        got = r.read(2, 9, 1, 12, depths=FULL)
        assert r.stats["requests"] == opened and r.stats["cache_hits"] > 0
        assert (got["fine"][:, :, 2] == vals["fine_3"][2:9, 1:12]).all()
    # rewriting the file changes its identity
    time.sleep(0.01)
    _write(tmp_path / "a.parquet", valid=mask)
    os.utime(tmp_path / "a.parquet", ns=(1, 1))
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=4096,
                             cache_dir=cdir) as r:
        assert r.file_key != key
        assert len(r.plan(2, 9, 1, 12, depths=FULL)) > 0


def test_memory_cache_budget():
    c = gp.RecordCache(max_bytes=10)
    k = lambda i: (("b", "f"), "base", i, 0)       # noqa: E731
    c.put(k(0), b"12345")
    c.put(k(1), b"12345")
    assert c.get(k(0)) == b"12345"                 # 0 is now most recent
    c.put(k(2), b"123")
    assert c.get(k(1)) is None and c.get(k(0)) and c.get(k(2))
    c.put(k(3), b"x" * 11)                         # larger than the budget
    assert c.get(k(3)) is None and c.nbytes <= 10


def test_module_does_not_import_torch():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, georvq.inference.georvq_parquet; print('torch' in sys.modules)"],
        cwd=root, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------
# threads, faults and hostile inputs
# --------------------------------------------------------------------------

def _region_reads(r, vals, n_threads=8, per_thread=25):
    """Reads random regions from `n_threads` threads through one reader;
    returns the errors and the number of reads that disagreed with
    `vals`."""
    errors, wrong = [], []

    def work(seed):
        rng = np.random.default_rng(seed)
        for _ in range(per_thread):
            wy0, wx0 = int(rng.integers(0, 9)), int(rng.integers(0, 12))
            wy1 = int(rng.integers(wy0 + 1, 11))
            wx1 = int(rng.integers(wx0 + 1, 14))
            try:
                got = r.read(wy0, wy1, wx0, wx1, depths=FULL, quality=True)
            except Exception as e:                 # noqa: BLE001
                errors.append(e)
                continue
            sl = np.s_[wy0:wy1, wx0:wx1]
            ok = (got["base"] == vals["base"][sl]).all() and all(
                (got[t][:, :, l] == vals[f"{t}_{l + 1}"][sl]).all()
                for t, L in (("mid", 2), ("fine", 3)) for l in range(L))
            if not ok:
                wrong.append((wy0, wy1, wx0, wx1))

    threads = [threading.Thread(target=work, args=(s,)) for s in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors, wrong


def test_one_local_reader_shared_by_threads(tmp_path, mask):
    vals = _write(tmp_path / "a.parquet", valid=mask)
    # no cache, so every read fetches and parses
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=1024,
                             cache_bytes=0) as r:
        errors, wrong = _region_reads(r, vals)
        assert errors == [] and wrong == []
        assert r.stats["requests"] > 8 * 25


def test_one_http_reader_shared_by_threads(served):
    url, state, vals = served
    state["delay"] = 0.002
    r = gp.RVQParquetReader(url, first_read=1024, cache_bytes=0, gap=0)
    errors, wrong = _region_reads(r, vals, per_thread=10)
    assert errors == [] and wrong == []
    assert r.stats["requests"] == len(state["requests"])


@pytest.mark.parametrize("status", [503, 429, 500])
def test_transient_failures_are_retried(served, status):
    url, state, vals = served
    state["fail"] = [status]                       # the suffix request
    r = gp.RVQParquetReader(url, first_read=4096, backoff=0.01)
    assert r.stats["retries"] == 1
    state["fail"] = [status, status]               # the next range request
    got = r.read(0, 10, 0, 13, depths=(3, 1, 0), mask=False)
    assert (got["mid"][:, :, 0] == vals["mid_1"]).all()
    assert r.stats["retries"] == 3


def test_persistent_failures_give_up(served):
    import urllib.error
    url, state, _ = served
    state["fail"] = [503] * 10
    with pytest.raises(urllib.error.HTTPError) as e:
        gp.RVQParquetReader(url, retries=2, backoff=0.01)
    assert e.value.code == 503 and len(state["requests"]) == 3
    # a 404 is final: one request, no retry
    state["fail"], state["requests"] = [404] * 10, []
    with pytest.raises(urllib.error.HTTPError) as e:
        gp.RVQParquetReader(url, retries=5, backoff=0.01)
    assert e.value.code == 404 and len(state["requests"]) == 1
    state["fail"], state["requests"] = [], []
    r_ok = gp.RVQParquetReader(url, first_read=4096, retries=5, backoff=0.01)
    state["fail"] = [416] * 10
    with pytest.raises(urllib.error.HTTPError):
        r_ok.read(0, 1, 0, 1, depths=(3, 1, 0))
    assert r_ok.stats["retries"] == 0


def test_timeout_is_retried(served):
    url, state, _ = served
    state["stall"] = 2.0
    t0 = time.perf_counter()
    r = gp.RVQParquetReader(url, first_read=4096, timeout=0.3, backoff=0.01)
    assert time.perf_counter() - t0 < 1.5
    assert r.stats["retries"] == 1
    state["stall"] = 2.0
    with pytest.raises((TimeoutError, OSError)):
        gp.RVQParquetReader(url, first_read=4096, timeout=0.3, retries=0)


def test_server_that_ignores_range(served, tmp_path):
    url, state, vals = served
    state["ignore_range"] = True
    with pytest.raises(OSError, match="ignored the Range header"):
        gp.RVQParquetReader(url, first_read=4096)
    # a file no longer than the suffix arrives whole, and is read from it
    size = os.path.getsize(tmp_path / "a.georvq.parquet")
    r = gp.RVQParquetReader(url, first_read=size + 10)
    got = r.read(0, 10, 0, 13, depths=FULL)
    assert (got["fine"][:, :, 2] == vals["fine_3"]).all()
    assert r.stats["requests"] == 1 and r.size == size


def test_content_range_must_match_the_request(served):
    url, state, _ = served
    r = gp.RVQParquetReader(url, first_read=4096)
    state["shift"] = 1
    with pytest.raises(OSError, match="were requested and bytes"):
        r.read(0, 10, 0, 13, depths=FULL)
    state["shift_suffix"] = True
    with pytest.raises(OSError, match="was answered with bytes"):
        gp.RVQParquetReader(url, first_read=4096)


def test_server_without_version_tags(served, tmp_path):
    url, state, vals = served
    state["range_etag"] = False                    # tag on the suffix only
    r = gp.RVQParquetReader(url, first_read=4096)
    assert r.version_tag == '"v1"'
    got = r.read(0, 10, 0, 13, depths=FULL)
    assert (got["fine"][:, :, 0] == vals["fine_1"]).all()
    assert r.identity == [url, '"v1"', r.size]
    # no tag at all: values are cached in memory only
    state["etag"] = None
    cdir = tmp_path / "cache"
    with pytest.warns(UserWarning, match="in memory only"):
        r = gp.RVQParquetReader(url, first_read=4096, cache_dir=cdir)
    assert r.identity == [url, None, r.size]
    got = r.read(0, 10, 0, 13, depths=FULL)
    assert (got["fine"][:, :, 0] == vals["fine_1"]).all()
    assert not list(cdir.rglob("*.bin"))
    before = r.stats["requests"]
    r.read(0, 10, 0, 13, depths=FULL)
    assert r.stats["requests"] == before           # memory cache still serves


def test_cache_paths_stay_inside_cache_dir(tmp_path):
    c = gp.RecordCache(cache_dir=tmp_path / "c")
    for key in [(("..", ".."), "../../evil", 1, 0),
                (("a/../../b", "f"), "x/../../y", 0, 0),
                ((".", "f"), "..", 0, 0)]:
        p = Path(c._path(key)).resolve()
        assert p.is_relative_to((tmp_path / "c").resolve()), (key, p)
        assert len(p.relative_to((tmp_path / "c").resolve()).parts) == 3
    # legitimate keys keep their names
    assert c._path((("test", "abc123"), "fine_2", 3, 4)) == str(
        tmp_path / "c" / "test" / "abc123" / "fine_2-3-4.bin")
    for bad in ("../evil", "a-b", "fine.1", "", "mid/2"):
        with pytest.raises(ValueError, match="tier name"):
            gp.Layout(WIN, 8, [("mean", 3, 12)], [(bad, 2, [("x", 4, 9)])])


def test_corrupt_disk_cache_values_are_misses(tmp_path, mask):
    vals = _write(tmp_path / "a.parquet", valid=mask)
    cdir = tmp_path / "cache"
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=1024,
                             cache_dir=cdir) as r:
        r.read(0, 10, 0, 13, depths=FULL)
    files = sorted(cdir.rglob("*.bin"))
    fine = next(f for f in files if f.name.startswith("fine_1-"))
    good = fine.read_bytes()
    fine.write_bytes(good[:-5])                    # truncated
    maskf = next(f for f in files if f.name.startswith("mask-"))
    maskf.write_bytes(b"not zlib")
    with gp.RVQParquetReader(tmp_path / "a.parquet", first_read=1024,
                             cache_dir=cdir) as r:
        got = r.read(0, 10, 0, 13, depths=FULL)
        assert (got["fine"][:, :, 0] == vals["fine_1"]).all()
        assert (got["valid"][:H, :W] == mask).all()
        assert r.stats["requests"] > 1
    assert fine.read_bytes() == good               # the fetch replaced it


def test_failed_open_releases_the_file(tmp_path):
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"x" * 100)
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(20):
        with pytest.raises(ValueError):
            gp.RVQParquetReader(bad)
    assert len(os.listdir("/proc/self/fd")) <= before


def test_failed_write_leaves_no_file(tmp_path, monkeypatch):
    path = tmp_path / "a.parquet"
    path.write_bytes(b"previous")
    calls = []
    real = gp.pq.ParquetWriter.write_table

    def failing(self, table, *a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("killed")
        return real(self, table, *a, **k)
    monkeypatch.setattr(gp.pq.ParquetWriter, "write_table", failing)
    with pytest.raises(RuntimeError, match="killed"):
        _write(path)
    assert path.read_bytes() == b"previous"
    assert os.listdir(tmp_path) == ["a.parquet"]


def test_mask_decoding_is_bounded():
    valid = np.random.default_rng(0).random((30, 21)) > 0.5
    data = gp.encode_mask(valid)
    assert (gp.decode_mask(data, valid.shape) == valid).all()
    import zlib
    for bad in (zlib.compress(b"\xff" * 10_000_000),   # inflates far past
                zlib.compress(b"\xff" * 10),           # too short
                data + b"junk"):                       # input after the stream
        with pytest.raises(ValueError, match="mask value"):
            gp.decode_mask(bad, valid.shape)
