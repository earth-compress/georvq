"""Tessera v1.1 release layer: the prep and validity hooks agree with
georvq.tessera.io, the int8 representation round trips, tiles are read
from a v1.1 Zarr store with the window and georeferencing of the chip
store builder, a tile padded to whole windows reads back through
georvq.open with no arguments against a tiny Tessera-shaped model, and
stores and bundles name the embeddings they code."""
import json
import os
import shutil

import numpy as np
import pytest
import torch

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")
zarr = pytest.importorskip("zarr")

import georvq  # noqa: E402
from georvq.config import load_config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference.api import Codec  # noqa: E402
from georvq.inference.bundle_resolve import BundleError  # noqa: E402
from georvq.inference.mean_codec import RotatedChannel  # noqa: E402
from georvq.inference.records import RecordReader  # noqa: E402
from georvq.tessera import tiles, v1_1  # noqa: E402
from georvq.tessera.io import prep  # noqa: E402

H, WD, WIN = 70, 45, 16
YEAR, LON, LAT = 2024, 17.25, 46.75
ORIGIN = (500000.0, 5180000.0)       # top-left of the tiny tile, EPSG:32633


def _tile_arrays(seed=0):
    """(128, H, W) int8 codes and an (H, W) scale plane, with nodata as
    all-zero vectors and as NaN, infinite and zero scales."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:WD] / 11.0
    dirs = rng.normal(size=(4, 128))
    f = np.einsum("khw,kc->chw", np.stack([np.sin(yy + k) * np.cos(xx * (k + 1))
                                           for k in range(4)]), dirs)
    f += 0.2 * rng.normal(size=f.shape)
    m = np.abs(f).max(0)
    q = np.clip(np.rint(f / m * 127), -127, 127).astype(np.int8)
    s = (m * 0.05 / 127 + 0.01).astype(np.float32)
    q[:, :6, :10] = 0
    s[50:52, 20:40] = np.nan
    s[60, 3] = np.inf
    s[30:35, 30:34] = 0.0
    return q, s


def _tiny_tile():
    q, s = _tile_arrays()
    t = (10.0, 0.0, ORIGIN[0], 0.0, -10.0, ORIGIN[1])
    return v1_1.TesseraTile.from_arrays(YEAR, LON, LAT, q, s, 32633, t,
                                        "test/utm33/embeddings[0,:,0:70,0:45]",
                                        (0, 0))


def test_validity_and_prep_agree_with_io():
    q, s = _tile_arrays()
    x, v = prep(q, s, channels_last=False)
    x2, v2_ = v1_1.TESSERA_V1_1.prep(None, (q[None], s[None]), "cpu")
    assert torch.equal(x, x2) and torch.equal(v, v2_)
    assert np.array_equal(v1_1.raw_valid(q, s), v[0].numpy())


def test_int8_representation_round_trips():
    q, s = _tile_arrays()
    valid = v1_1.raw_valid(q, s)
    x = q.astype(np.float32) * np.where(valid, s, 0)[None]
    q2, s2 = v1_1.to_geotessera(x, valid)
    assert q2.shape == (H, WD, 128)
    assert np.array_equal(q2.transpose(2, 0, 1)[:, valid], q[:, valid])
    np.testing.assert_allclose(s2[valid], s[valid], rtol=1e-6)
    assert (s2[~valid] == 0).all() and (q2[~valid] == 0).all()


def _v11_store(path, model=tiles.V11_MODEL):
    """A local v1.1 store with one zone (33), one year and fill values
    everywhere (nodata) except a known patch inside the tile's window.
    Returns (path, (r0, c0, h, w), zone origin, patch origin)."""
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    lo, la = np.meshgrid(LON + np.array([-.05, 0, .05]),
                         LAT + np.array([-.05, 0, .05]))
    e, n = fwd.transform(lo.ravel(), la.ravel())
    west, top = np.floor(min(e) / 10) * 10 - 70, np.ceil(max(n) / 10) * 10 + 50
    win = tiles.tile_window(LON, LAT, west, top, 10.0, fwd.transform)
    r0, c0, h, w = win
    hz, wz = r0 + h + 20, c0 + w + 20
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update({"geoemb:model": model})
    g = root.create_group("utm33")
    g.attrs.update({"spatial:transform": [10, 0, west, 0, -10, top],
                    "proj:code": "EPSG:32633"})
    g.create_array("time", data=np.array([2023, YEAR], np.int32))
    emb = g.create_array("embeddings", shape=(2, 128, hz, wz), dtype="int8",
                         chunks=(1, 128, 64, 64), fill_value=0)
    sc = g.create_array("scales", shape=(2, hz, wz), dtype="float32",
                        chunks=(1, 64, 64), fill_value=float("nan"))
    q, s = _tile_arrays()
    py, px = r0 + 100, c0 + 200
    emb[1, :, py:py + H, px:px + WD] = q
    sc[1, py:py + H, px:px + WD] = s
    return str(path), win, (west, top), (py - r0, px - c0)


def test_tile_is_read_with_the_builders_window(tmp_path):
    path, (r0, c0, h, w), (west, top), (py, px) = _v11_store(tmp_path / "v11")
    t = v1_1.TesseraTile.read(YEAR, LON, LAT, store=path)
    assert (t.height, t.width) == (h, w) and t.crs.to_epsg() == 32633
    assert t.transform == (10.0, 0.0, west + 10 * c0, 0.0, -10.0, top - 10 * r0)
    assert t.source == f"{path}/utm33/embeddings[1,:,{r0}:{r0 + h},{c0}:{c0 + w}]"
    q, s = _tile_arrays()
    tiles.clean_nodata(q, s)
    assert np.array_equal(t.q[:, py:py + H, px:px + WD], q)
    assert np.array_equal(t.s[py:py + H, px:px + WD], s)
    # the fill (NaN scales) is nodata in the v1 convention
    assert (t.s[:py] == 0).all() and (t.q[:, :py] == 0).all()
    p = t.product_meta
    assert p["embeddings"] == tiles.EMBEDDINGS and p["utm_zone"] == 33
    assert p["zone_window"] == {"row": r0, "col": c0, "height": h, "width": w}
    assert t.product_id == "2024/grid_17.25_46.75"
    with pytest.raises(ValueError, match="has no 2022"):
        v1_1.TesseraTile.read(2022, LON, LAT, store=path)
    other, *_ = _v11_store(tmp_path / "v1", model="https://geotessera.org/model/1.0")
    with pytest.raises(ValueError, match="not https://geotessera.org/model/1.1"):
        v1_1.TesseraTile.read(YEAR, LON, LAT, store=other)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tesv11")
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
    tile = _tiny_tile()
    W = WIN
    qc, sc = tile.q[:, :64, :32], tile.s[:64, :32]
    x, v = prep(qc.reshape(128, 4, W, 2, W).transpose(1, 3, 0, 2, 4)
                .reshape(-1, 128, W, W).copy(),
                sc.reshape(4, W, 2, W).transpose(0, 2, 1, 3)
                .reshape(-1, W, W).copy(), channels_last=False)
    means = model.patch_stats(x, v).numpy()
    means = means[v.reshape(len(v), -1).any(1).numpy()]
    # a few windows do not span 128 dimensions: fit on jittered copies
    rng = np.random.default_rng(0)
    fit = (means[rng.integers(len(means), size=2000)]
           + 0.01 * rng.normal(size=(2000, 128)))
    channel = RotatedChannel.fit_pca(fit, fit, 128 * 6, box=v1_1.BOX)
    codec = v1_1.TesseraV11Codec(model, channel)
    out = v1_1.store_path(tmp / "stores", YEAR, LON, LAT)
    os.makedirs(os.path.dirname(out))
    v1_1.encode_tile(tile, out, codec, chunk_px={"base": 64, "mid": 64,
                                                 "fine": 32})
    ckpt = tmp / "tiny.pt"
    from georvq.repro import save_checkpoint
    save_checkpoint(ckpt, model, cfg, 3)
    npz = tmp / "mean.npz"
    channel.save(npz)
    bundle = tmp / "bundle"
    v1_1.build_bundle(ckpt, npz, bundle)
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
    y = (y * v1_1.XNORM * v[:, None]).numpy()
    y = y.reshape(gh, gw, 128, WIN, WIN).transpose(2, 0, 3, 1, 4)
    return y.reshape(128, gh * WIN, gw * WIN)[:, :H, :WD]


def test_store_header_names_the_embeddings(tiny):
    with RecordReader(tiny["store"]) as rd:
        h = rd.header
    assert h["codec"]["format"] == v1_1.FORMAT == "georvq-tessera/v1.1"
    assert h["codec"]["mean_channel"] == {"kind": "rotated", "bits": 768}
    assert h["product"]["kind"] == v1_1.KIND and h["product"]["year"] == YEAR
    assert h["product"]["id"] == f"{YEAR}/grid_17.25_46.75"
    assert h["product"]["embeddings"]["version"] == "1.1"
    assert h["product"]["embeddings"]["run"] == "dclimate"
    assert h["product"]["source"] == tiny["tile"].source
    assert h["output"] == {"dtype": "float32", "nodata": 0.0, "count": 128,
                           "band_names": None}
    assert h["licence"]["spdx"] == "CC-BY-SA-4.0"
    assert "dClimate" in h["licence"]["attribution"]
    assert "relative L2" in h["quality"]["error"]
    assert [tuple(r["tuple"]) for r in h["rungs"]][0] == (2, 0, 0)
    assert h["layout"]["product_px"] == [H, WD]
    assert h["codec"]["release"]["name"] == "tessera-v1.1"
    assert h["codec"]["release"]["uri"] == "hf://earth-compress/georvq-tessera-v1.1"


def test_encode_refuses_other_embeddings(tiny, tmp_path):
    t = _tiny_tile()
    t.product_meta["embeddings"] = dict(t.product_meta["embeddings"],
                                        version="1.0")
    with pytest.raises(ValueError, match="embeddings"):
        v1_1.encode_tile(t, str(tmp_path / "x.georvq.zarr"), tiny["codec"])


def test_channel_kind_is_checked(tiny):
    from georvq.inference.mean_codec import MeanChannel
    ch = MeanChannel(np.zeros(128), np.ones(128), 4)
    with pytest.raises(ValueError, match="rotated"):
        v1_1.TesseraV11Codec(tiny["codec"].model, ch)


def test_georvq_open_reads_like_a_direct_decode(tiny, tmp_path, monkeypatch):
    root = tmp_path / "cache"
    entry = root / "bundles" / tiny["codec"].bitstream_id
    entry.parent.mkdir(parents=True)
    shutil.copytree(tiny["bundle"], entry)
    monkeypatch.setenv("GEORVQ_CACHE", str(root))
    monkeypatch.delenv(v1_1.ENV_BUNDLE, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    t = tiny["tile"]
    invalid = ~v1_1.raw_valid(t.q, t.s)
    with georvq.open(tiny["store"]) as ds:
        assert ds.dtypes == ("float32",) * 128 and ds.nodata == 0.0
        assert ds.crs.to_epsg() == 32633 and (ds.height, ds.width) == (H, WD)
        for r in [(3, 0, 0), (3, 2, 0), (3, 2, 2)]:
            got = ds.read(quality=r)
            np.testing.assert_allclose(got, _direct(tiny, r), rtol=0, atol=1e-4)
            assert (got[:, invalid] == 0).all()
    with georvq.open(tiny["store"], units="model", quality="3-2-2") as ds:
        model_units = ds.read()
    np.testing.assert_allclose(model_units * v1_1.XNORM, got, rtol=1e-6)
    meta = json.load(open(tiny["bundle"] / "bundle.json"))
    assert meta["licence_stores"]["spdx"] == "CC-BY-SA-4.0"
    assert meta["licence_weights"] == "Apache-2.0"
    assert meta["variant"] == "full" and meta["release"] == "tessera-v1.1"
    assert meta["embeddings"] == tiles.EMBEDDINGS
    dec = json.load(open(tiny["bundle"] / "decoder" / "bundle.json"))
    assert dec["embeddings"] == tiles.EMBEDDINGS and dec["variant"] == "decoder"
    assert "licensed Apache-2.0" in open(tiny["bundle"] / "README.md").read()


def test_bundles_for_other_embeddings_are_refused(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # a bundle of this format whose bundle.json names other embeddings
    other = tmp_path / "other"
    shutil.copytree(tiny["bundle"], other)
    meta = json.load(open(other / "bundle.json"))
    meta["embeddings"] = dict(meta["embeddings"], version="1.0")
    json.dump(meta, open(other / "bundle.json", "w"))
    with pytest.raises(ValueError, match="embeddings"):
        v1_1.load_bundle(other)
    # a bundle of another release's format, as a Tessera v1 bundle has
    v1 = tmp_path / "v1"
    shutil.copytree(tiny["bundle"], v1)
    meta = json.load(open(v1 / "bundle.json"))
    meta["format"] = "georvq-tessera/v2-bundle"
    json.dump(meta, open(v1 / "bundle.json", "w"))
    with pytest.raises(BundleError, match="not a tessera-v1.1 bundle"):
        with georvq.open(tiny["store"], bundle=str(v1)) as ds:
            ds.read(quality="2-0-0")


def test_index_of_stores(tiny):
    pytest.importorskip("pyarrow")
    from georvq.inference.store_index import CORE
    from georvq.tessera.index import build_index
    root = os.path.dirname(os.path.dirname(tiny["store"]))
    t = build_index(root)
    assert t.schema.names[:len(CORE)] == CORE.names
    (r,) = t.to_pylist()
    assert r["crs"] == "EPSG:32633" and r["release"] == "tessera-v1.1"
    assert r["product_id"] == "2024/grid_17.25_46.75"
    assert (r["year"], r["tile_lon"], r["tile_lat"], r["utm_zone"]) == \
        (YEAR, LON, LAT, 33)
    assert r["embeddings_version"] == "1.1"
    t0 = tiny["tile"]
    assert r["valid_fraction"] == pytest.approx(
        v1_1.raw_valid(t0.q, t0.s).mean())
    assert r["rungs"][0]["rung"] == "2-0-0"
    assert r["source_url"] == t0.source
    for prefix, want in (("", "2024/grid_17.25_46.75.georvq.zarr"),
                         ("https://h/x/", "https://h/x/2024/grid_17.25_46.75"
                                          ".georvq.zarr")):
        t = build_index(root, url_prefix=prefix)
        assert t.column("path").to_pylist() == [want]
