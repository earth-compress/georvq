"""The AE v2 tile index: synthetic tiles coded by a tiny untrained
stats_cond model, indexed against a synthetic source index with the
AEF index's schema."""
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

rasterio = pytest.importorskip("rasterio")

from georvq.ae import v2  # noqa: E402
from georvq.ae.index import build_index, tile_record  # noqa: E402
from georvq.config import load_config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference.georvq_parquet import RVQParquetReader  # noqa: E402
from georvq.inference.mean_codec import RVQPCAMeanChannel  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = 16
TIERS = "model.tiers=['1:16:32:3','2:8:32:2','4:8:16:2']"
STATS = ("model.stats_cond={subtract_mean: true, cond_encoder: true, "
         "cond_decoder: true, bits_per_band: 6}")
CHUNK_PX = {"base": 32, "mid": 32, "fine": 16}
SRC_PREFIX = "s3://bucket/tge-labs/aef/v1/annual"
TILES = [("2024", "33N", "xaaaa-0000000000-0000000000", 40, 48),
         ("2024", "33N", "xaaaa-0000000000-0000008192", 32, 40),
         ("2023", "34S", "xbbbb-0000000000-0000000000", 48, 32)]
SMOKE = os.environ.get(
    "GEORVQ_TEST_AE_V2_TILE",
    "/home/ali/projects/georvq/scratch/ae-v2/parquet_smoke2/serve/"
    "x21yfwngg902u0lsh-0000008192-0000000000.georvq.parquet")
AEF_INDEX = os.environ.get("GEORVQ_TEST_AEF_INDEX",
                           "/data/AE-samples/aef_index.parquet")


def tiny_codec():
    """An untrained AE v2 codec: a smoke-sized model and a mean channel
    fitted to random means."""
    torch.manual_seed(0)
    cfg = load_config(os.path.join(REPO, "configs/smoke.yaml"), [
        "model.bands=64", f"model.window={WIN}", TIERS, STATS])
    model = build_model(cfg, "cpu").eval()
    rng = np.random.default_rng(0)
    r = rng.normal(size=(128, 64)) * 0.05
    books = []
    for _ in range(2):
        C = r[rng.choice(len(r), 16, replace=False)].copy()
        a = ((r[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        r = r - C[a]
        books.append(C)
    channel = RVQPCAMeanChannel.fit_residual(np.stack(books), r, total_bits=136)
    return v2.AEV2Codec(model, channel)


def write_tile(path, codec, h, w, seed=0, stats=True):
    """Encode a random bottom-up int8 tile to `path`; returns the source."""
    from rasterio.transform import Affine
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, size=(64, h, w), dtype=np.int8)
    a[:, :5, :7] = v2.NODATA
    src = os.path.splitext(os.path.splitext(path)[0])[0] + ".tif"
    t = Affine(10.0, 0.0, 500000.0, 0.0, 10.0, 4000000.0)
    with rasterio.open(src, "w", driver="GTiff", width=w, height=h, count=64,
                       dtype="int8", nodata=v2.NODATA, crs="EPSG:32633",
                       transform=t) as ds:
        ds.write(a)
    v2.encode_tile(src, path, codec, stats=stats, chunk_px=CHUNK_PX, strip_px=WIN)
    os.remove(src)
    return src


def source_index(path, tiles, extra=2):
    """A source index with the AEF schema and geo metadata: one row per
    tile plus `extra` rows for tiles that were not coded."""
    rows = [(y, z, pid) for y, z, pid, *_ in tiles] + [
        ("2022", "10N", f"xzzzz-{i:010d}-0000000000") for i in range(extra)]
    rows = rows[::-1]                       # source order differs from file order
    n = len(rows)
    ring = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float)
    geoms, bbs = [], []
    for i in range(n):
        pts = ring * 0.5 + [10.0 + i, 40.0 - i]
        geoms.append(b"\x01\x03\x00\x00\x00\x01\x00\x00\x00\x05\x00\x00\x00"
                     + pts.astype("<f8").tobytes())
        bbs.append({"xmax": pts[:, 0].max(), "xmin": pts[:, 0].min(),
                    "ymax": pts[:, 1].max(), "ymin": pts[:, 1].min()})
    t = pa.table({
        "fid": pa.array(range(100, 100 + n), pa.int64()),
        "crs": [f"EPSG:326{z[:2]}" for _, z, _ in rows],
        "path": [f"{SRC_PREFIX}/{y}/{z}/{p}.tiff" for y, z, p in rows],
        "year": pa.array([int(y) for y, _, _ in rows], pa.int64()),
        "utm_zone": [z for _, z, _ in rows],
        **{f"utm_{k}": pa.array([float(i) for i in range(n)])
           for k in ("west", "south", "east", "north")},
        **{f"wgs84_{k}": pa.array([b[f"{a}"] for b in bbs])
           for k, a in (("west", "xmin"), ("south", "ymin"), ("east", "xmax"),
                        ("north", "ymax"))},
        "geom": pa.array(geoms, pa.binary()),
        "geom_bbox": pa.array(bbs, pa.struct([(k, pa.float64()) for k in
                                              ("xmax", "xmin", "ymax", "ymin")])),
        "location": [f"VRT://vsis3/bucket/{y}/{z}/{p}.tiff" for y, z, p in rows]})
    geo = {"primary_column": "geom", "version": "1.0.0",
           "columns": {"geom": {"encoding": "WKB", "crs": None,
                                "geometry_types": ["Polygon"],
                                "bbox": [-180.0, -90.0, 180.0, 90.0]}}}
    t = t.replace_schema_metadata({"geo": json.dumps(geo), "pandas": "{}"})
    pq.write_table(t, path)
    return t


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("aeindex")
    codec = tiny_codec()
    root = tmp / "v2"
    files = []
    for i, (y, z, pid, h, w) in enumerate(TILES):
        d = root / y / z
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{pid}.georvq.parquet"
        write_tile(str(f), codec, h, w, seed=i, stats=i != 1)
        files.append(str(f))
    src = tmp / "aef_index.parquet"
    source_index(src, TILES)
    return {"codec": codec, "root": root, "files": files, "src": src, "tmp": tmp}


def test_rows_match_the_source_index(tree):
    out = tree["tmp"] / "idx.parquet"
    t = build_index(tree["root"], tree["src"], out_path=out,
                    url_prefix="https://host/aef/v2/")
    src = pq.read_table(tree["src"])
    assert t.num_rows == len(TILES)
    src_cols = src.schema.names
    assert t.schema.names[:len(src_cols) + 1] == (
        src_cols[:3] + ["source_path"] + src_cols[3:])
    by_src = {r["path"]: r for r in src.to_pylist()}
    want = {f"{SRC_PREFIX}/{y}/{z}/{p}.tiff": f"https://host/aef/v2/{y}/{z}/"
            f"{p}.georvq.parquet" for y, z, p, *_ in TILES}
    rows = t.to_pylist()
    for r in rows:
        assert r["path"] == want[r["source_path"]]
        s = by_src[r["source_path"]]
        for c in src_cols:
            if c != "path":
                assert r[c] == s[c], c
        assert r["bitstream_id"] == tree["codec"].bitstream_id
        assert r["decoder_id"] == tree["codec"].decoder_id
    # source order
    src_order = [s["path"] for s in src.to_pylist()]
    idx = [src_order.index(r["source_path"]) for r in rows]
    assert idx == sorted(idx)
    assert pq.read_table(out).equals(t)


def test_rung_bytes_account_for_the_file(tree):
    for f in tree["files"]:
        rec = tile_record(f)
        with RVQParquetReader(f) as rd:
            col = rd.column_bytes()
            rungs = rd.header["rungs"]
            assert [r["rung"] for r in rec["rungs"]] == [
                "-".join(map(str, r["tuple"])) for r in rungs]
            for r, h in zip(rec["rungs"], rungs):
                assert r["bits_px"] == h["bits_px"]
                assert r["bytes"] == sum(col[c] for c in rd.columns_for(h["tuple"]))
            full = sum(col[c] for c in rd.columns_for(None))
        b = [r["bytes"] for r in rec["rungs"]]
        assert b == sorted(b) and b[0] > 0
        assert b[-1] == full                # the last rung is full depth
        meta = sum(col[c] for c in ("kind", "product_id", "chunk_row",
                                    "chunk_col", "geometry", "bbox"))
        assert rec["quality_bytes"] == col.get("quality", 0)
        assert rec["file_bytes"] == os.path.getsize(f)
        assert rec["file_bytes"] == (4 + meta + full + rec["quality_bytes"]
                                     + rec["footer_bytes"])
    assert tile_record(tree["files"][1])["quality_bytes"] == 0


def test_only_footers_are_read(tree):
    """Zeroing everything but the footer leaves the record unchanged."""
    f = tree["files"][0]
    rec = tile_record(f)
    g = tree["tmp"] / "blank.georvq.parquet"
    shutil.copy(f, g)
    with open(g, "r+b") as fh:
        fh.seek(4)
        fh.write(b"\0" * (rec["file_bytes"] - rec["footer_bytes"] - 4))
    assert tile_record(g, first_read=rec["footer_bytes"]) == rec


def test_geoparquet_metadata(tree):
    out = tree["tmp"] / "idx_geo.parquet"
    t = build_index(tree["files"], tree["src"], out_path=out)
    assert sorted(t.column("path").to_pylist()) == sorted(tree["files"])
    geo = json.loads(pq.read_schema(out).metadata[b"geo"])
    assert set(pq.read_schema(out).metadata) == {b"geo"}
    gc = geo["columns"][geo["primary_column"]]
    assert geo["primary_column"] == "geom" and geo["version"] == "1.1.0"
    assert gc["encoding"] == "WKB"
    assert gc["covering"]["bbox"] == {k: ["geom_bbox", k]
                                      for k in ("xmin", "ymin", "xmax", "ymax")}
    bb = t.column("geom_bbox").to_pylist()
    assert gc["bbox"] == [min(b["xmin"] for b in bb), min(b["ymin"] for b in bb),
                          max(b["xmax"] for b in bb), max(b["ymax"] for b in bb)]


def test_unknown_or_repeated_products_raise(tree):
    other = tree["tmp"] / "other_index.parquet"
    source_index(other, TILES[:1])
    with pytest.raises(ValueError, match="not in"):
        build_index(tree["root"], other)
    with pytest.raises(ValueError, match="both code"):
        build_index([tree["files"][0]] * 2, tree["src"])
    with pytest.raises(ValueError, match="needs root"):
        build_index(tree["files"], tree["src"], url_prefix="https://host/")


@pytest.mark.skipif(not (os.path.exists(SMOKE) and os.path.exists(AEF_INDEX)),
                    reason="real AE v2 smoke file or AEF index absent")
def test_real_file_against_the_aef_index():
    t = build_index([SMOKE], AEF_INDEX)
    (r,) = t.to_pylist()
    assert r["source_path"].endswith(
        "/annual/2021/44N/x21yfwngg902u0lsh-0000008192-0000000000.tiff")
    assert r["path"] == SMOKE and r["year"] == 2021
    assert r["rungs"][-1]["rung"] == "8-8-8"
    assert (r["rungs"][-1]["bytes"] + r["quality_bytes"] + r["footer_bytes"]
            < r["file_bytes"])
