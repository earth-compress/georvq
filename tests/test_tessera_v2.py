"""Tessera v2 release layer: the prep and validity hooks agree with
georvq.tessera.io, the geotessera representation round trips, tiles are
georeferenced by their landmask or the rule, and a tile padded to whole
windows reads back through georvq.open with no arguments, against a tiny
Tessera-shaped model."""
import json
import os
import shutil

import numpy as np
import pytest
import torch

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")
import rasterio  # noqa: E402

import georvq  # noqa: E402
from georvq.config import load_config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference.api import Codec  # noqa: E402
from georvq.inference.mean_codec import RVQPCAMeanChannel  # noqa: E402
from georvq.inference.records import RecordReader  # noqa: E402
from georvq.tessera import v2  # noqa: E402
from georvq.tessera.io import prep  # noqa: E402

H, WD, WIN = 70, 45, 16
YEAR, LON, LAT = 2024, 17.25, 46.75
LANDMASKS = os.path.expanduser(
    "~/.cache/georvq_lab/geotessera/global_0.1_degree_tiff_all")


def _tile(seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:WD] / 11.0
    dirs = rng.normal(size=(4, 128))
    f = np.einsum("khw,kc->hwc", np.stack([np.sin(yy + k) * np.cos(xx * (k + 1))
                                           for k in range(4)]), dirs)
    f += 0.2 * rng.normal(size=f.shape)
    m = np.abs(f).max(-1)
    q = np.clip(np.rint(f / m[..., None] * 127), -127, 127).astype(np.int8)
    s = (m * 0.05 / 127 + 0.01).astype(np.float32)
    q[:6, :10] = 0                  # all-zero vectors: nodata
    s[50:52, 20:40] = np.nan        # water
    s[60, 3] = np.inf
    s[30:35, 30:34] = 0.0
    return q, s                     # (H, W, 128), (H, W), as distributed


def test_validity_and_prep_agree_with_io():
    q, s = _tile()
    qc = np.ascontiguousarray(q.transpose(2, 0, 1))
    x, v = prep(qc, np.where(np.isfinite(s), s, np.nan), channels_last=False)
    assert np.array_equal(v[0].numpy(), v2.raw_valid(qc, s))
    x2, v2_ = v2.TESSERA_V2.prep(None, (qc[None], s[None]), "cpu")
    assert torch.equal(x, x2) and torch.equal(v, v2_)


def test_geotessera_representation_round_trips():
    q, s = _tile()
    qc = q.transpose(2, 0, 1).astype(np.float32)
    valid = v2.raw_valid(q.transpose(2, 0, 1), s)
    x = qc * np.where(valid, s, 0)[None]
    q2, s2 = v2.to_geotessera(x, valid)
    assert q2.shape == q.shape and q2.dtype == np.int8 and s2.dtype == np.float32
    assert (np.abs(q2[valid].astype(int)).max(-1) == 127).all()
    assert (q2[~valid] == 0).all() and (s2[~valid] == 0).all()
    err = np.abs(q2.transpose(2, 0, 1) * s2[None] - x)
    assert (err <= 0.5 * s2[None] + 1e-7).all()


@pytest.mark.skipif(not os.path.isdir(LANDMASKS), reason="no cached landmasks")
def test_rule_matches_cached_landmasks():
    for f in sorted(os.listdir(LANDMASKS)):
        lon, lat = (float(v) for v in f[len("grid_"):-len(".tiff")].split("_"))
        with rasterio.open(os.path.join(LANDMASKS, f)) as ds:
            lm = (ds.crs.to_epsg(), tuple(list(ds.transform)[:6]),
                  (ds.height, ds.width))
        assert v2._same_georef(lm, v2.rule_georef(lon, lat)), f


def _root(tmp, q, s, landmask=True):
    """A local copy of the bucket's layout holding one tile and, with
    `landmask`, a landmask GeoTIFF of the tile's (small) size."""
    name = v2.tile_name(LON, LAT)
    d = tmp / v2.EMBEDDINGS_DIR / str(YEAR) / name
    d.mkdir(parents=True)
    np.save(d / f"{name}.npy", q)
    np.save(d / f"{name}_scales.npy", s)
    if landmask:
        lm = tmp / v2.LANDMASK_DIR
        lm.mkdir()
        epsg, t, _ = v2.rule_georef(LON, LAT)
        with rasterio.open(lm / f"{name}.tiff", "w", driver="GTiff", height=H,
                           width=WD, count=1, dtype="uint8",
                           crs=f"EPSG:{epsg}",
                           transform=rasterio.Affine(*t)) as ds:
            ds.write(np.ones((1, H, WD), np.uint8))
    return tmp


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tesv2")
    torch.manual_seed(0)
    cfg = load_config("configs/smoke.yaml", [
        "data.source=tessera", "model.bands=128", f"model.window={WIN}",
        "model.tiers=['1:16:32:3','2:8:32:2','4:8:16:2']",
        "model.min_depths=[2,0,0]", "model.vq.book_bits=8",
        "model.stats_cond={subtract_mean: true, cond_encoder: true, "
        "cond_decoder: true, bits_per_band: 6}"])
    model = build_model(cfg, "cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    for rvq in model.rvqs:
        rvq.project_books()
    model.eval()
    q, s = _tile()
    root = _root(tmp / "bucket", q, s)
    tile = v2.GeotesseraTile(YEAR, LON, LAT, root=str(root), check_rule=False)
    qc = tile.q[:, :64, :32]
    x, v = prep(qc.reshape(128, 4, WIN, 2, WIN).transpose(1, 3, 0, 2, 4)
                .reshape(-1, 128, WIN, WIN).copy(),
                tile.s[:64, :32].reshape(4, WIN, 2, WIN).transpose(0, 2, 1, 3)
                .reshape(-1, WIN, WIN).copy(), channels_last=False)
    means = model.patch_stats(x, v).numpy()
    means = means[v.reshape(len(v), -1).any(1).numpy()]
    books = means[:4][None].copy()
    a = ((means[:, None] - books[0][None]) ** 2).sum(-1).argmin(1)
    channel = RVQPCAMeanChannel.fit_residual(books, means - books[0][a],
                                             total_bits=200)
    codec = v2.TesseraV2Codec(model, channel)
    out = v2.store_path(tmp / "stores", YEAR, LON, LAT)
    os.makedirs(os.path.dirname(out))
    v2.encode_tile(tile, out, codec, chunk_px={"base": 64, "mid": 64,
                                               "fine": 32})
    ckpt = tmp / "tiny.pt"
    from georvq.repro import save_checkpoint
    save_checkpoint(ckpt, model, cfg, 3)
    npz = tmp / "mean.npz"
    channel.save(npz)
    bundle = tmp / "bundle"
    v2.build_bundle(ckpt, npz, bundle)
    return {"tmp": tmp, "codec": codec, "tile": tile, "store": out,
            "bundle": bundle}


def _direct(tiny, depths):
    codec, t = tiny["codec"], tiny["tile"]
    gh, gw = -(-H // WIN), -(-WD // WIN)
    qp = np.zeros((128, gh * WIN, gw * WIN), np.int8)
    sp = np.zeros((gh * WIN, gw * WIN), np.float32)
    qp[:, :H, :WD], sp[:H, :WD] = t.q, t.s
    qw = qp.reshape(128, gh, WIN, gw, WIN).transpose(1, 3, 0, 2, 4)
    sw = sp.reshape(gh, WIN, gw, WIN).transpose(0, 2, 1, 3)
    x, v = prep(qw.reshape(-1, 128, WIN, WIN).copy(),
                sw.reshape(-1, WIN, WIN).copy(), channels_last=False)
    mu = codec.model.patch_stats(x, v).numpy()
    m = torch.from_numpy(codec.channel.round_trip(mu))
    c = Codec(codec.model)
    y = c.decode(c.encode(x, valid=v, mean=m), depths=list(depths), valid=v)
    y = (y * v2.XNORM * v[:, None]).numpy()
    y = y.reshape(gh, gw, 128, WIN, WIN).transpose(2, 0, 3, 1, 4)
    return y.reshape(128, gh * WIN, gw * WIN)[:, :H, :WD]


def test_tile_is_georeferenced_by_its_landmask(tiny, tmp_path):
    t = tiny["tile"]
    assert t.georef == "landmask" and (t.height, t.width) == (H, WD)
    epsg, tr, _ = v2.rule_georef(LON, LAT)
    assert t.crs.to_epsg() == epsg == 32633 and t.transform == tr
    q, s = _tile()
    root = _root(tmp_path / "b", q, s, landmask=False)
    with pytest.raises(ValueError, match="its rule gives"):
        v2.GeotesseraTile(YEAR, LON, LAT, root=str(root))
    rel = f"{v2.EMBEDDINGS_DIR}/{YEAR}/grid_17.25_46.75/grid_17.25_46.75"
    assert t.source == str(tiny["tmp"] / "bucket") + "/" + rel + ".npy"
    u = v2.GeotesseraTile(YEAR, LON, LAT, root=str(tiny["tmp"] / "bucket"),
                          check_rule=False, source_root=v2.TESSERA_BUCKET)
    assert u.source == v2.TESSERA_BUCKET + rel + ".npy"
    assert u.product_meta["scales"] == v2.TESSERA_BUCKET + rel + "_scales.npy"
    assert np.array_equal(u.q, t.q) and np.array_equal(u.s, t.s)


def test_store_header(tiny):
    with RecordReader(tiny["store"]) as rd:
        h = rd.header
    assert h["codec"]["format"] == v2.FORMAT
    assert h["product"]["kind"] == v2.KIND and h["product"]["year"] == YEAR
    assert h["product"]["id"] == f"{YEAR}/grid_17.25_46.75"
    assert h["output"] == {"dtype": "float32", "nodata": 0.0, "count": 128,
                           "band_names": None}
    assert h["licence"]["spdx"] == "CC-BY-SA-4.0"
    assert "relative L2" in h["quality"]["error"]
    assert [tuple(r["tuple"]) for r in h["rungs"]][0] == (2, 0, 0)
    assert h["layout"]["product_px"] == [H, WD]


def test_georvq_open_reads_like_a_direct_decode(tiny, tmp_path, monkeypatch):
    root = tmp_path / "cache"
    entry = root / "bundles" / tiny["codec"].bitstream_id
    entry.parent.mkdir(parents=True)
    shutil.copytree(tiny["bundle"], entry)
    monkeypatch.setenv("GEORVQ_CACHE", str(root))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    t = tiny["tile"]
    invalid = ~v2.raw_valid(t.q, t.s)
    with georvq.open(tiny["store"]) as ds:
        assert ds.dtypes == ("float32",) * 128 and ds.nodata == 0.0
        assert ds.crs.to_epsg() == 32633 and (ds.height, ds.width) == (H, WD)
        for r in [(3, 0, 0), (3, 2, 0), (3, 2, 2)]:
            got = ds.read(quality=r)
            np.testing.assert_allclose(got, _direct(tiny, r), rtol=0, atol=1e-4)
            assert (got[:, invalid] == 0).all()
        model_units = None
    with georvq.open(tiny["store"], units="model", quality="3-2-2") as ds:
        model_units = ds.read()
    np.testing.assert_allclose(model_units * v2.XNORM, got, rtol=1e-6)
    meta = json.load(open(tiny["bundle"] / "bundle.json"))
    assert meta["licence_stores"]["spdx"] == "CC-BY-SA-4.0"
    assert meta["licence_weights"] == "Apache-2.0"
    assert meta["variant"] == "full" and meta["release"] == "tessera-v2"
    assert "licensed Apache-2.0" in open(tiny["bundle"] / "README.md").read()


def test_index_joins_the_manifest(tiny, tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    man = tmp_path / "manifest.parquet"
    pq.write_table(pa.table({
        "year": pa.array([YEAR, YEAR], pa.int16()),
        "lon": [LON, 17.35], "lat": [LAT, LAT],
        "grid_path": ["a", "b"],
        "lon_i": pa.array([1725, 1735], pa.int32()),
        "lat_i": pa.array([4675, 4675], pa.int32())}), man)
    t = v2.build_index(os.path.dirname(os.path.dirname(tiny["store"])), man)
    assert t.num_rows == 1 and t.column("grid_path").to_pylist() == ["a"]
    assert t.column("rungs").to_pylist()[0][0]["rung"] == "2-0-0"
    from georvq.inference.store_index import CORE
    assert t.schema.names[:len(CORE)] == CORE.names
    (r,) = t.to_pylist()
    assert r["crs"] == "EPSG:32633" and r["release"] == "tessera-v2"
    assert r["product_id"] == "2024/grid_17.25_46.75"
    root = os.path.dirname(os.path.dirname(tiny["store"]))
    for prefix, want in (("", "2024/grid_17.25_46.75.georvq.zarr"),
                         ("https://h/x/", "https://h/x/2024/grid_17.25_46.75"
                                          ".georvq.zarr")):
        t = v2.build_index(root, man, url_prefix=prefix)
        assert t.column("path").to_pylist() == [want]
