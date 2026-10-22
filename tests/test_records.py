"""georvq Zarr stores (georvq.inference.records): record and mask round
trip with per-tier chunk sizes, independence from the order windows are
added in, empty chunks, reads planned as one zarr-python read per array,
the record cache, quality kept per decoder, and zarr-python and xarray
reading the store, locally, as a rechunked copy and over HTTP."""
import http.server
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from georvq.inference import records as R  # noqa: E402
from georvq.inference.bitpack import pack, unpack  # noqa: E402

H, W = 37, 50                     # pixels
WIN = 4                           # 10 x 13 windows
CHUNK = {"base": 64, "mid": 16, "fine": 8}
# base and quality: 1 x 1 chunks; mid: 3 x 4; fine: 5 x 7, all with
# partial edge chunks
TRANSFORM = (10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
DID = "0123456789abcdef"
HEADER = {"codec": {"bitstream_id": "test", "decoder_id": DID},
          "product": {"kind": "synthetic", "height": H, "width": W,
                      "epsg": 32633},
          "rungs": []}
FULL = (3, 2, 3)


def _layout(quality=True, window=WIN, chunk=CHUNK):
    return R.Layout(window, chunk, [("mean", 3, 12), ("coarse", 6, 5)],
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
           lay=None, h=H, w=W, step=3, consolidated=True):
    """Writes the product in window strips of `step` rows by `step + 2`
    columns, in the given order."""
    lay = lay or _layout()
    vals = _values(lay, h, w)
    header = dict(HEADER, product=dict(HEADER["product"], height=h, width=w))
    wr = R.RecordWriter(path, header, lay, "synthetic-1", TRANSFORM, h, w,
                        aux=aux, consolidated=consolidated)
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


def _files(root):
    """{relative path: bytes} of every file of a store."""
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = Path(p).read_bytes()
    return out


@pytest.fixture
def mask():
    rng = np.random.default_rng(3)
    valid = rng.random((H, W)) > 0.1
    valid[8:24, 8:24] = True
    return valid


@pytest.fixture
def east_nodata():
    """Valid west of column 24 only: window columns 6..12 have no valid
    pixel, so fine chunk columns 3..6 and mid chunk columns 2..3 are
    empty, and mid chunk column 1 is partial."""
    valid = np.ones((H, W), bool)
    valid[:, 24:] = False
    return valid


def test_round_trip_and_partial_windows(tmp_path, mask):
    vals = _write(tmp_path / "a.georvq.zarr", valid=mask)
    with R.RecordReader(tmp_path / "a.georvq.zarr") as r:
        assert r.header["codec"] == HEADER["codec"]
        assert r.header["format"] == "georvq-zarr"
        assert r.header["status"] == "prototype"
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
    vals = _write(tmp_path / "a.zarr", lay=lay, h=h, w=w, valid=valid, step=4)
    with R.RecordReader(tmp_path / "a.zarr") as r:
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
    # the stored bytes are the packed records, over the window grid
    g = zarr.open_group(str(tmp_path / "a.zarr"), mode="r")
    col = lay.columns["mid_1"]
    want = pack(vals["mid_1"][20:21, 10:16].reshape(-1, col.n_values), col.widths)
    assert (g["mid"][20:21, 10:16, 0].reshape(6, -1) == want).all()


def test_mask_in_a_tier_kind(tmp_path, mask):
    lay = R.Layout(WIN, CHUNK, [("mean", 3, 12), ("coarse", 6, 5)],
                   [("coarse", 3, None), ("mid", 2, [("mid", 4, 9)]),
                    ("fine", 3, [("fine", 8, 11)])], mask_kind="fine")
    assert lay.column_order == ["base", "mid_1", "mid_2", "mask", "fine_1",
                                "fine_2", "fine_3"]
    _write(tmp_path / "a.zarr", valid=mask, lay=lay)
    with R.RecordReader(tmp_path / "a.zarr") as r:
        assert r.layout.mask_kind == "fine" and (1, 1) not in r.masked
        assert (r.read_mask(0, H, 0, W) == mask).all()
        cols = r.columns_for((3, 2, 0))
        assert ("mask", 2, 4) in r.needs(5, 6, 9, 10, cols)
        # a fully valid fine chunk has no mask to read
        assert not any(n[0] == "mask" for n in r.needs(2, 3, 2, 3, cols))
    with pytest.raises(ValueError, match="mask_kind"):
        R.Layout(WIN, 8, [("a", 1, 1)], [("t", 1, None)], mask_kind="mid")


def test_tier_arrays_and_columns_write_the_same_store(tmp_path):
    _write(tmp_path / "a.zarr")
    _write(tmp_path / "b.zarr", by_tier=True)
    assert _files(tmp_path / "a.zarr") == _files(tmp_path / "b.zarr")


def test_add_order_does_not_change_the_store(tmp_path, mask):
    _write(tmp_path / "a.zarr", valid=mask, aux='{"x": 1}')
    _write(tmp_path / "b.zarr", order="shuffled", valid=mask, aux='{"x": 1}',
           step=2)
    assert _files(tmp_path / "a.zarr") == _files(tmp_path / "b.zarr")


def test_header_and_attributes(tmp_path, mask):
    _write(tmp_path / "a.zarr", valid=mask, aux={"tags": {"a": 1}})
    g = zarr.open_group(str(tmp_path / "a.zarr"), mode="r")
    hdr = g.attrs["georvq"]
    assert hdr["version"] == f"{R.MAJOR}.{R.MINOR}"
    assert hdr["layout"]["product_px"] == [H, W]
    assert hdr["layout"]["mask"] == {"column": "mask", "kind": "base",
                                     "chunks": [[0, 0]]}
    assert g.attrs["georvq_aux"] == {"tags": {"a": 1}}
    assert g.attrs["proj:code"] == "EPSG:32633"
    assert "proj:projjson" not in g.attrs
    assert g.attrs["spatial:transform"] == list(TRANSFORM)
    assert g["base"].attrs["spatial:transform"] == [40.0, 0.0, 500000.0, 0.0,
                                                    -40.0, 4000000.0]
    assert list(g["quality"].attrs["decoders"]) == [DID]


# --------------------------------------------------------------------------
# empty chunks
# --------------------------------------------------------------------------

def test_empty_chunks_are_not_stored(tmp_path, east_nodata):
    vals = _write(tmp_path / "a.zarr", valid=east_nodata)
    g = zarr.open_group(str(tmp_path / "a.zarr"), mode="r")
    hdr = g.attrs["georvq"]
    empty = {k["kind"]: k["empty_chunks"] for k in hdr["layout"]["groups"]}
    assert empty == {"base": [], "quality": [],
                     "mid": [[i, j] for i in range(3) for j in (2, 3)],
                     "fine": [[i, j] for i in range(5) for j in range(3, 7)]}
    # the fine shard holds the 15 live chunks at 3 levels, and its index
    rb = _layout().columns["fine_1"].record_bytes
    shard = os.path.getsize(tmp_path / "a.zarr" / "fine" / "c" / "0" / "0" / "0" / "0")
    assert shard == 15 * 3 * 4 * rb + 16 * 5 * 7 * 3 + 4
    with R.RecordReader(tmp_path / "a.zarr") as r:
        assert r.empty["fine"] == {(i, j) for i in range(5) for j in range(3, 7)}
        assert r.dead[:, 6:].all() and not r.dead[:, :6].any()
        got = r.read(0, 10, 0, 13, quality=True)
        live = np.zeros((10, 13), bool)
        live[:, :6] = True
        for k, v in vals.items():
            t, _, l = k.partition("_")
            gk = got[t][:, :, int(l) - 1] if l else got[k]
            assert (gk[live] == v[live]).all(), k
        assert not got["fine"][:, 6:].any() and not got["mid"][:, 8:].any()
        # windows 6 and 7 are dead in the partial mid chunk column 1, and
        # keep the records they were given
        assert (got["mid"][:, 6:8, 0] == vals["mid_1"][:, 6:8]).all()
        assert (got["valid"][:H, :W] == east_nodata).all()


def test_reads_over_empty_chunks_read_nothing(tmp_path, east_nodata):
    _write(tmp_path / "a.zarr", valid=east_nodata)
    with R.RecordReader(tmp_path / "a.zarr") as r:
        # windows (2..5, 7..12) lie in empty fine chunks; base, mid and the
        # mask are not read for them either
        assert r.plan(2, 5, 7, 13, depths=FULL, quality=True) == []
        before = dict(r.stats)
        got = r.read(2, 5, 7, 13, depths=FULL, quality=True)
        assert r.stats == before
        assert not got["valid"].any()
        assert not got["base"].any() and not got["fine"].any()
        # a region touching one live window reads its chunks only
        needs = r.needs(2, 3, 5, 7, r.columns_for(FULL))
        assert {(n[0], n[1], n[2]) for n in needs if n[0].startswith("fine")} \
            == {(f"fine_{l}", 1, 2) for l in (1, 2, 3)}


def test_wholly_invalid_product(tmp_path):
    _write(tmp_path / "a.zarr", valid=np.zeros((H, W), bool))
    g = zarr.open_group(str(tmp_path / "a.zarr"), mode="r")
    assert not g["mid"][...].any() and not g["fine"][...].any()
    with R.RecordReader(tmp_path / "a.zarr") as r:
        assert r.dead.all() and r.plan(0, 10, 0, 13, quality=True) == []
        got = r.read(0, 10, 0, 13, quality=True)
        assert not got["valid"].any() and not got["base"].any()
        assert not r.read_mask(0, H, 0, W).any()


# --------------------------------------------------------------------------
# reads, cache and writer checks
# --------------------------------------------------------------------------

def test_one_zarr_read_per_array(tmp_path, mask):
    _write(tmp_path / "a.zarr", valid=mask)
    with R.RecordReader(tmp_path / "a.zarr") as r:
        plan = dict(r.plan(0, 10, 0, 13, quality=True))
        assert set(plan) == {"base", "mask", "mid", "fine", "quality"}
        assert plan["fine"] == (slice(0, 10), slice(0, 13), slice(0, 3))
        # (5, 9) lies in mid chunk (1, 2) and fine chunk (2, 4)
        plan = dict(r.plan(5, 6, 9, 10, depths=(3, 1, 2), mask=False))
        assert plan == {"base": (slice(0, 10), slice(0, 13)),
                        "mid": (slice(4, 8), slice(8, 12), slice(0, 1)),
                        "fine": (slice(4, 6), slice(8, 10), slice(0, 2))}


def test_cache_serves_later_reads(tmp_path, mask):
    vals = _write(tmp_path / "a.zarr", valid=mask)
    with R.RecordReader(tmp_path / "a.zarr") as r:
        r.read(2, 9, 1, 12, depths=FULL)
        n = r.stats["values"]
        assert n > 0 and r.stats["cache_hits"] == 0
        got = r.read(2, 9, 1, 12, depths=FULL)
        assert r.stats["values"] == n and r.stats["cache_hits"] == n
        assert r.plan(2, 9, 1, 12, depths=FULL) == []
        assert (got["fine"][:, :, 2] == vals["fine_3"][2:9, 1:12]).all()
    with R.RecordReader(tmp_path / "a.zarr", cache_bytes=0) as r:
        r.read(2, 9, 1, 12, depths=FULL)
        r.read(2, 9, 1, 12, depths=FULL)
        assert r.stats["cache_hits"] == 0


def test_memory_cache_budget():
    c = R.RecordCache(max_bytes=10)
    k = lambda i: (("b", "f"), "base", i, 0)       # noqa: E731
    c.put(k(0), b"12345")
    c.put(k(1), b"12345")
    assert c.get(k(0)) == b"12345"                 # 0 is now most recent
    c.put(k(2), b"123")
    assert c.get(k(1)) is None and c.get(k(0)) and c.get(k(2))
    c.put(k(3), b"x" * 11)                         # larger than the budget
    assert c.get(k(3)) is None and c.nbytes <= 10


def test_one_reader_shared_by_threads(tmp_path, mask):
    vals = _write(tmp_path / "a.zarr", valid=mask)
    errors, wrong = [], []
    r = R.RecordReader(tmp_path / "a.zarr", cache_bytes=0)

    def work(seed):
        rng = np.random.default_rng(seed)
        for _ in range(15):
            wy0, wx0 = int(rng.integers(0, 9)), int(rng.integers(0, 12))
            wy1 = int(rng.integers(wy0 + 1, 11))
            wx1 = int(rng.integers(wx0 + 1, 14))
            try:
                got = r.read(wy0, wy1, wx0, wx1, depths=FULL, quality=True)
            except Exception as e:                 # noqa: BLE001
                errors.append(e)
                continue
            sl = np.s_[wy0:wy1, wx0:wx1]
            if not ((got["base"] == vals["base"][sl]).all() and all(
                    (got[t][:, :, l] == vals[f"{t}_{l + 1}"][sl]).all()
                    for t, L in (("mid", 2), ("fine", 3)) for l in range(L))):
                wrong.append((wy0, wy1, wx0, wx1))

    threads = [threading.Thread(target=work, args=(s,)) for s in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and wrong == []


def test_writer_checks(tmp_path):
    lay = _layout()
    with pytest.raises(ValueError, match="set by the writer"):
        R.RecordWriter(tmp_path / "a", dict(HEADER, layout={}), lay, "p",
                       TRANSFORM, H, W)
    with pytest.raises(ValueError, match="multiple of window_px"):
        _layout(chunk={"base": 64, "mid": 18, "fine": 8})
    with pytest.raises(ValueError, match="chunk_px keys"):
        _layout(chunk={"base": 64, "mid": 16})
    w = R.RecordWriter(tmp_path / "a", HEADER, lay, "p", TRANSFORM, H, W)
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
    # quality needs the decoder it was measured with
    hdr = dict(HEADER, codec={"bitstream_id": "test"})
    with pytest.raises(ValueError, match="decoder_id"):
        _write_with_header(tmp_path / "b", hdr)


def _write_with_header(path, header):
    lay = _layout()
    w = R.RecordWriter(path, header, lay, "p", TRANSFORM, H, W)
    w.add_windows(0, 0, _values(lay))
    w.close()


def test_failed_write_keeps_the_previous_store(tmp_path, monkeypatch):
    _write(tmp_path / "a.zarr")
    before = _files(tmp_path / "a.zarr")

    def failing(self, out, header):
        os.makedirs(out)
        raise RuntimeError("killed")
    monkeypatch.setattr(R.RecordWriter, "_write", failing)
    with pytest.raises(RuntimeError, match="killed"):
        _write(tmp_path / "a.zarr", valid=np.zeros((H, W), bool))
    assert _files(tmp_path / "a.zarr") == before
    assert os.listdir(tmp_path) == ["a.zarr"]


def test_a_new_store_replaces_the_old(tmp_path, mask):
    _write(tmp_path / "a.zarr")
    _write(tmp_path / "a.zarr", valid=mask)
    assert R.RecordReader(tmp_path / "a.zarr").masked == {(0, 0)}
    assert os.listdir(tmp_path) == ["a.zarr"]


def test_refuses_other_major_version(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MAJOR", 7)
    _write(tmp_path / "a.zarr")
    monkeypatch.setattr(R, "MAJOR", 0)
    with pytest.raises(ValueError, match="major version 7"):
        R.RecordReader(tmp_path / "a.zarr")


def test_tier_names():
    assert R.tier_names(3) == ["coarse", "mid", "fine"]
    assert R.tier_names(2) == ["tier0", "tier1"]
    assert R.output_name("/a/b/tile_1.tif") == "tile_1.georvq.zarr"


def test_module_does_not_import_torch_or_zarr():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, georvq.inference.records; "
         "print('torch' in sys.modules, 'zarr' in sys.modules)"],
        cwd=root, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


# --------------------------------------------------------------------------
# zarr-python and xarray
# --------------------------------------------------------------------------

@pytest.mark.parametrize("consolidated", [True, False])
def test_zarr_python_reads_the_store(tmp_path, mask, consolidated):
    lay = _layout()
    vals = _write(tmp_path / "a.zarr", valid=mask, consolidated=consolidated)
    g = zarr.open_group(str(tmp_path / "a.zarr"), mode="r")
    assert set(g.array_keys()) == {"base", "mid", "fine", "mask"}
    assert list(g["quality"].array_keys()) == [DID]
    for name, arr in (("base", g["base"]), ("quality", g["quality"][DID])):
        col = lay.columns[name]
        u = unpack(arr[...].reshape(-1, col.record_bytes), col.widths)
        assert (u.reshape(10, 13, -1) == vals[name]).all()
    for t, L in (("mid", 2), ("fine", 3)):
        col = lay.columns[f"{t}_1"]
        for l in range(L):
            u = unpack(g[t][:, :, l].reshape(-1, col.record_bytes), col.widths)
            assert (u.reshape(10, 13, -1) == vals[f"{t}_{l + 1}"]).all()
    assert (g["mask"][...] == mask).all()
    with R.RecordReader(g) as r:
        assert (r.read(0, 10, 0, 13, quality=True)["fine"][:, :, 2]
                == vals["fine_3"]).all()


def test_xarray_opens_the_store(tmp_path, mask):
    xr = pytest.importorskip("xarray")
    _write(tmp_path / "a.zarr", valid=mask)
    ds = xr.open_zarr(str(tmp_path / "a.zarr"), chunks=None)
    assert set(ds.data_vars) == {"base", "mid", "fine", "mask"}
    assert ds["fine"].dtype == np.uint8 and ds["mask"].dtype == bool
    q = xr.open_zarr(str(tmp_path / "a.zarr"), group="quality", chunks=None)
    assert set(q.data_vars) == {DID}


def _copy(src, out, recompress=True):
    """A copy of a store with other chunks, no sharding and zstd on every
    array, as Zarr tools might write it."""
    from zarr.codecs import ZstdCodec
    g = zarr.open_group(str(src), mode="r")
    dst = zarr.open_group(str(out), mode="w-", zarr_format=3)
    names = list(g.array_keys()) + [f"quality/{d}" for d in g["quality"].attrs["decoders"]]
    for name in names:
        a = g[name]
        parent, _, leaf = name.rpartition("/")
        if parent and parent not in dst:
            dst.create_group(parent, attributes=dict(g[parent].attrs))
        chunks = tuple(max(1, s // 3) for s in a.shape[:2]) + tuple(a.shape[2:])
        b = (dst[parent] if parent else dst).create_array(
            leaf, shape=a.shape, dtype=a.dtype, chunks=chunks,
            compressors=ZstdCodec(level=3) if recompress else None,
            fill_value=a.fill_value, attributes=dict(a.attrs))
        b[...] = a[...]
    dst.update_attributes(dict(g.attrs))
    return zarr.open_group(str(out), mode="r")


def test_a_rechunked_copy_reads_the_same(tmp_path, mask):
    _write(tmp_path / "a.zarr", valid=mask)
    a = R.RecordReader(tmp_path / "a.zarr")
    b = R.RecordReader(_copy(tmp_path / "a.zarr", tmp_path / "b.zarr"))
    for depths, quality in ((None, True), ((3, 1, 2), False)):
        ra = a.read(0, 10, 0, 13, depths=depths, quality=quality)
        rb = b.read(0, 10, 0, 13, depths=depths, quality=quality)
        assert ra.keys() == rb.keys()
        assert all((ra[k] == rb[k]).all() for k in ra)


def test_a_wrong_shape_is_refused(tmp_path):
    _write(tmp_path / "a.zarr")
    g = zarr.open_group(str(tmp_path / "a.zarr"), mode="r+")
    a = g["fine"]
    del g["fine"]
    g.create_array("fine", shape=(9,) + a.shape[1:], dtype="uint8",
                   chunks=(9,) + a.shape[1:])
    zarr.consolidate_metadata(str(tmp_path / "a.zarr"))
    with pytest.raises(ValueError, match="fine is uint8"):
        R.RecordReader(tmp_path / "a.zarr")


# --------------------------------------------------------------------------
# quality per decoder
# --------------------------------------------------------------------------

OTHER = "fedcba9876543210"


def add_decoder(store, did=OTHER, bitstream=None, rungs=3):
    """Add the quality array of another decoder of the same codes, with
    one stat over `rungs` rungs (so a record of another size), as a later
    release would; returns its (wy, wx, rungs) record bytes."""
    g = zarr.open_group(str(store), mode="r+")
    own = g["quality"].attrs["decoders"][0]
    q0 = g["quality"][own]
    info = dict(q0.attrs["georvq_quality"])
    info.update(decoder_id=did,
                fields=[{"name": "p99", "count": rungs, "bits": 8}],
                rungs=[{"tuple": [3, k, 0]} for k in range(rungs)])
    if bitstream is not None:
        info["bitstream_id"] = bitstream
    rec = np.random.default_rng(5).integers(0, 256, q0.shape[:2] + (rungs,),
                                            dtype=np.uint8)
    a = g["quality"].create_array(did, shape=rec.shape, dtype="uint8",
                                  chunks=q0.chunks[:2] + (rungs,),
                                  compressors=None, fill_value=0,
                                  attributes={**dict(q0.attrs),
                                              "georvq_quality": info})
    a[...] = rec
    g["quality"].update_attributes(
        {"decoders": list(g["quality"].attrs["decoders"]) + [did]})
    zarr.consolidate_metadata(str(store))
    return rec


@pytest.mark.parametrize("via", ["path", "group"])
def test_quality_is_kept_per_decoder(tmp_path, mask, via):
    vals = _write(tmp_path / "a.zarr", valid=mask)
    rec = add_decoder(tmp_path / "a.zarr")
    src = (tmp_path / "a.zarr" if via == "path"
           else zarr.open_group(str(tmp_path / "a.zarr"), mode="r"))
    z = R.RecordReader(src)
    assert sorted(z.quality_decoders) == [DID, OTHER]
    assert z.quality_decoder == DID
    want = vals["quality"]
    assert (z.read(0, 10, 0, 13, quality=True)["quality"] == want).all()
    z.select_quality(OTHER)
    assert [r["tuple"] for r in z.header["rungs"]] == [[3, 0, 0], [3, 1, 0],
                                                      [3, 2, 0]]
    assert z.layout.columns["quality"].n_values == 3
    got = z.read(0, 10, 0, 13, quality=True)
    assert (got["quality"] == rec.astype(np.int64)).all()
    assert (got["fine"][:, :, 0] == vals["fine_1"]).all()   # codes unchanged
    z.select_quality(DID)                              # cached values do not mix
    assert (z.read(0, 10, 0, 13, quality=True)["quality"] == want).all()
    z.select_quality(None)
    assert z.layout.quality is None and "quality" not in z.header
    with pytest.raises(ValueError, match="no quality column"):
        z.read(0, 1, 0, 1, quality=True)
    with pytest.raises(ValueError, match="no quality for decoder 1111"):
        z.select_quality("1111")


def test_quality_of_another_bitstream_is_refused(tmp_path):
    _write(tmp_path / "a.zarr")
    add_decoder(tmp_path / "a.zarr", bitstream="other")
    with pytest.raises(ValueError, match="bitstream other"):
        R.RecordReader(tmp_path / "a.zarr")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class _DirHandler(http.server.BaseHTTPRequestHandler):
    """Range server of a directory, logging (path, Authorization)."""
    protocol_version = "HTTP/1.1"
    root = None
    log = None

    def _send(self, body):
        self.log.append((self.path, self.headers.get("Authorization")))
        path = os.path.join(self.root, self.path.lstrip("/"))
        if not os.path.isfile(path):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = open(path, "rb").read()
        m = _RANGE.match(self.headers.get("Range") or "")
        if m is None:
            a, b, code = 0, len(data) - 1, 200
        elif m.group(1) == "":
            a, b, code = max(0, len(data) - int(m.group(2))), len(data) - 1, 206
        else:
            a, b, code = int(m.group(1)), min(int(m.group(2)), len(data) - 1), 206
        self.send_response(code)
        if code == 206:
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        self.send_header("Content-Length", str(b - a + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if body:
            self.wfile.write(data[a:b + 1])

    def do_GET(self):
        self._send(True)

    def do_HEAD(self):
        self._send(False)

    def log_message(self, *a):
        pass


@pytest.fixture
def served(tmp_path, mask):
    pytest.importorskip("aiohttp")
    vals = _write(tmp_path / "a.zarr", valid=mask)
    log = []
    h = type("H", (_DirHandler,), {"root": str(tmp_path), "log": log})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path, log, vals
    finally:
        srv.shutdown()
        srv.server_close()


def test_http_reads_through_zarr_python(served):
    base, tmp, log, vals = served
    r = R.RecordReader(f"{base}/a.zarr",
                       storage_options={"headers": {"Authorization": "Bearer t"}})
    local = R.RecordReader(tmp / "a.zarr")
    for depths, q in ((None, True), ((3, 1, 2), False), ((3, 0, 0), False)):
        a = r.read(1, 9, 2, 11, depths=depths, quality=q)
        b = local.read(1, 9, 2, 11, depths=depths, quality=q)
        assert a.keys() == b.keys() and all((a[k] == b[k]).all() for k in a)
    assert log and {auth for _, auth in log} == {"Bearer t"}
    with pytest.raises(ValueError, match="storage_options"):
        R.RecordReader(zarr.open_group(str(tmp / "a.zarr"), mode="r"),
                       storage_options={})
