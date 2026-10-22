"""S2 v1 release layer with a tiny S2 model: stores on three output grids,
the SAFE metadata zip, manifest.safe checksums, and the SAFE -> store ->
SAFE round trip. The released model's numerics are pinned by
test_s2_v1_bundle; these tests need no bundle."""
import json
import os

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("zarr")

import georvq  # noqa: E402
from georvq.config import Config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference.mean_codec import RVQPCAMeanChannel  # noqa: E402
from georvq.s2 import safe as S  # noqa: E402
from georvq.s2 import v1  # noqa: E402
from tests.s2_safe_fixture import make_safe, synthetic_bands  # noqa: E402

N = 144                                   # 10 m pixels of the test crop
WIN = 36
CHUNK_PX = {"base": 72, "tier1": 72}
T10 = (10.0, 0.0, 399960.0, 0.0, -10.0, 5800020.0)


def tiny_channel():
    """An RVQ-PCA channel for the 26 S2 statistics with the release's kind
    of transform: means over 5 DN, log standard deviations times 10."""
    nb = len(S.BANDS)
    scale = np.concatenate([np.full(nb, 1e4 / 5), np.full(nb, 10.0)])
    lg = np.r_[np.zeros(nb, bool), np.ones(nb, bool)]
    mu0 = np.r_[np.full(nb, 400.0), np.full(nb, -45.0)]
    s = np.r_[np.full(nb, 300.0), np.full(nb, 8.0)]
    books = np.zeros((1, 2, 2 * nb), np.float32)
    return RVQPCAMeanChannel(books, mu0, np.eye(2 * nb), s,
                             np.full(2 * nb, 8), np.full(2 * nb, 3.0),
                             pre_scale=scale, pre_log=lg)


def tiny_codec():
    cfg = Config.model_validate({
        "data": {"source": "s2", "store_dir": "/tmp/unused"},
        "model": {"window": WIN, "bands": 13,
                  "tiers": ["1:16:32:3", "6:8:32:3"], "min_depths": [3, 0],
                  "ff_ratio": 2.0, "tokenizer": {"dim": 16, "depth": 1},
                  "decoder": {"dim": 32, "depth": 1, "heads": 4,
                              "head_feat": 8}},
        "log": {"run_tag": "s2v1test", "wandb": False}})
    torch.manual_seed(0)
    return v1.S2V1Codec(build_model(cfg, "cpu"), tiny_channel())


@pytest.fixture(scope="module")
def codec():
    return tiny_codec()


class _Crop:
    """The synthetic bands as a ProductSource."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.height = self.width = N
        self.transform = T10
        from rasterio.crs import CRS
        self.crs = CRS.from_epsg(32637)
        self.orientation = "north-up"
        self.product_id = "crop"
        self.source = "crop"
        self.product_meta = {"count": 13}
        self.aux = None

    def read_strip(self, y0, y1):
        return tuple(a[:, y0 // f:y1 // f]
                     for a, f in zip(self.arrays, v1.FACTORS))


def _assemble(blk, dec, f):
    """Decoded windows of one grid -> (C, H / f, W / f), zero elsewhere."""
    w = WIN // f
    gh, gw = blk.grid
    out = np.zeros((dec.shape[1], gh * w, gw * w), np.uint16)
    for k, (r, c) in enumerate(zip(blk.rows, blk.cols)):
        out[:, r * w:(r + 1) * w, c * w:(c + 1) * w] = dec[k]
    return out[:, :N // f, :N // f]


def direct_decode(codec, arrays, rung):
    """{grid: (C, h, w)} decoded from the codec's records of the crop,
    without a store."""
    torch.set_grad_enabled(False)
    try:
        blk = codec.encode_block(arrays, stats=False)
        rec = blk.records
        tiers = [rec[n].transpose(1, 0, 2)[:, :rung[i + 1]] if rung[i + 1]
                 else None for i, n in enumerate(codec.names[1:])]
        v = np.zeros((blk.grid[0] * WIN, blk.grid[1] * WIN), bool)
        v[:N, :N] = blk.valid
        vw = np.stack([v[r * WIN:(r + 1) * WIN, c * WIN:(c + 1) * WIN]
                       for r, c in zip(blk.rows, blk.cols)])
        dec = codec.decode_records(rec["base"][0], tiers, rung, vw)
    finally:
        torch.set_grad_enabled(True)
    return {g: _assemble(blk, dec[g], f) for g, f, _ in S.GRIDS}


def test_band_constants_match_the_source():
    from georvq.sources import s2
    assert list(S.BANDS_10M) == s2.BANDS_10M
    assert list(S.BANDS_20M) == s2.BANDS_20M
    assert list(S.BANDS_60M) == s2.BANDS_60M
    assert {g: (len(b), f) for g, f, b in S.GRIDS} == s2.GROUPS
    assert v1.MAX_VALUE == s2.MAX_VALUE
    assert sorted(S.S2_ORDER) == sorted(S.BANDS)


def test_raw_valid_matches_pixel_valid():
    from georvq.sources.s2 import pixel_valid
    arrays = synthetic_bands(N, seed=3)
    arrays[1][2, 10, 10] = 0
    arrays[2][0, 20, 3] = 0
    bv = {g: torch.from_numpy(a[None].astype(np.int32)) != 0
          for (g, _, _), a in zip(S.GRIDS, arrays)}
    assert np.array_equal(v1.raw_valid_np(arrays), pixel_valid(bv)[0].numpy())


def test_store_reads_three_grids(codec, tmp_path):
    arrays = synthetic_bands(N)
    out = str(tmp_path / "crop.georvq.zarr")
    torch.set_grad_enabled(False)
    try:
        v1.encode_product(_Crop(arrays), out, codec, chunk_px=CHUNK_PX,
                          strip_px=72, batch=5,
                          licence=v1.licence_of(2026))
    finally:
        torch.set_grad_enabled(True)
    with georvq.open(out, codec=codec) as ds:
        h = ds.header
        assert h["codec"]["format"] == "georvq-s2/v1"
        assert h["licence"]["attribution"] == \
            "Contains modified Copernicus Sentinel data 2026"
        assert h["output"]["band_names"] == list(S.S2_ORDER)
        assert h["quality"]["stats"] == ["rmse_10m", "rmse_20m", "rmse_60m"]
        assert ds.grids == ("10m", "20m", "60m")
        assert ds.grid_name == "10m"
        for g, f, names in S.GRIDS:
            d = ds.grid(g)
            assert (d.height, d.width, d.count) == (N // f, N // f, len(names))
            assert d.descriptions == names
            assert tuple(d.transform)[:6] == (10.0 * f, 0.0, T10[2], 0.0,
                                              -10.0 * f, T10[5])
            assert d.res == (10.0 * f, 10.0 * f)
        for rung in [(3, 0), (3, 2)]:
            want = direct_decode(codec, arrays, rung)
            got = ds.read_grids(quality=rung)
            for g, f, _ in S.GRIDS:
                assert np.array_equal(got[g], want[g]), (rung, g)
                assert np.array_equal(ds.grid(g).read(quality=rung), want[g])
                part = ds.grid(g).read(quality=rung, window=(
                    (12 // f, 96 // f), (6 // f, 138 // f)))
                assert np.array_equal(part, want[g][:, 12 // f:96 // f,
                                                    6 // f:138 // f])
            sub = ds.read_grids(((30, 101), (7, 144)), quality=rung)
            for g, f, _ in S.GRIDS:
                assert np.array_equal(sub[g], want[g][:, 30 // f:-(-101 // f),
                                                      7 // f:])
        valid = v1.raw_valid_np(arrays)
        for g, f, _ in S.GRIDS:
            m = ds.grid(g).dataset_mask() > 0
            vf = valid.reshape(N // f, f, N // f, f).all((1, 3))
            assert np.array_equal(m, vf), g
            a = ds.grid(g).read(quality=(3, 1))
            assert (a[:, ~vf] == 0).all() and (a[:, vf] >= 1).all()
        b = ds.grid("60m").read(1, quality="rmse_60m<=1e6")
        assert np.array_equal(b, direct_decode(codec, arrays, (3, 0))["60m"][0])


def test_grid_views_share_the_codec(codec, tmp_path):
    arrays = synthetic_bands(N, seed=1)
    out = str(tmp_path / "crop.georvq.zarr")
    torch.set_grad_enabled(False)
    try:
        v1.encode_product(_Crop(arrays), out, codec, chunk_px=CHUNK_PX,
                          stats=False)
    finally:
        torch.set_grad_enabled(True)
    calls = []

    def resolver(header):
        calls.append(1)
        return codec
    from georvq.inference.store_dataset import StoreDataset
    ds = StoreDataset(out, v1.SPEC, resolver=resolver, grid="20m")
    assert ds.grid_name == "20m" and ds.count == 6
    v = ds.grid("60m")
    v.read(quality=(3, 3))
    ds.read(quality=(3, 3))
    assert len(calls) == 1
    with pytest.raises(ValueError, match="no output grid"):
        ds.grid("30m")
    ds.close()


def test_single_grid_stores_refuse_a_grid(tmp_path):
    from tests.test_tessera_v2_format import _write_store
    from georvq.tessera import v2
    out = _write_store(str(tmp_path))
    from georvq.inference.store_dataset import StoreDataset
    with pytest.raises(ValueError, match="one output grid"):
        StoreDataset(out, v2.SPEC, grid="10m")
    ds = StoreDataset(out, v2.SPEC)
    assert ds.grids == () and ds.grid_name is None
    with pytest.raises(ValueError, match="one output grid"):
        ds.read_grids()
    ds.close()


# --------------------------------------------------------------------------
# SAFE
# --------------------------------------------------------------------------

def _tree(d):
    out = {}
    for r, dirs, fs in os.walk(d):
        for x in dirs:
            out[os.path.relpath(os.path.join(r, x), d) + "/"] = None
        for x in fs:
            p = os.path.join(r, x)
            out[os.path.relpath(p, d)] = open(p, "rb").read()
    return out


@pytest.fixture(scope="module")
def safe_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("safe")
    return make_safe(str(root), synthetic_bands(N), T10)


def test_safe_meta_zip(safe_dir, tmp_path):
    import zipfile
    a = S.safe_meta_zip(safe_dir)
    assert S.safe_meta_zip(safe_dir) == a
    names = zipfile.ZipFile(__import__("io").BytesIO(a)).namelist()
    assert names == sorted(names)
    assert not any(S.is_band_file(n) for n in names)
    assert any(n.endswith("_TCI.jp2") for n in names)
    assert any(n.endswith("_PVI.jp2") for n in names)
    assert "AUX_DATA/" in names and "manifest.safe" in names
    out = str(tmp_path / "x.SAFE")
    S.unzip_meta(a, out)
    want = {k: v for k, v in _tree(safe_dir).items()
            if not S.is_band_file(k.replace(os.sep, "/"))}
    assert _tree(out) == want
    with pytest.raises(FileExistsError):
        S.unzip_meta(a, out)


def test_unzip_refuses_paths_outside(tmp_path):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", b"x")
    with pytest.raises(ValueError, match="leaves"):
        S.unzip_meta(buf.getvalue(), str(tmp_path / "out"))


def test_manifest_rewrite(safe_dir, tmp_path):
    text = open(os.path.join(safe_dir, "manifest.safe")).read()
    ent = S.manifest_entries(text)
    files = S.band_files(safe_dir)
    assert set(files.values()) <= set(ent)
    for rel, (size, name, h) in ent.items():
        p = os.path.join(safe_dir, rel)
        assert size == os.path.getsize(p) and name == "SHA3-256"
        assert h == S.file_checksum(p, name)
    d = str(tmp_path / "copy")
    import shutil
    shutil.copytree(safe_dir, d)
    rel = files["B05"]
    with open(os.path.join(d, rel), "ab") as f:
        f.write(b"changed")
    new = S.rewrite_manifest(text, d, [rel], "a note")
    e2 = S.manifest_entries(new)
    assert e2[rel][0] == os.path.getsize(os.path.join(d, rel))
    assert e2[rel][2] == S.file_checksum(os.path.join(d, rel), "SHA3-256")
    assert {k: v for k, v in e2.items() if k != rel} == \
        {k: v for k, v in ent.items() if k != rel}
    assert new.splitlines()[1] == "<!-- a note -->"
    assert new.replace("<!-- a note -->\n", "").count("\n") == text.count("\n")
    with pytest.raises(ValueError, match="does not list"):
        S.rewrite_manifest(text, d, ["GRANULE/nope.jp2"], "x")


def test_safe_product_reads_the_bands(safe_dir):
    p = v1.SafeProduct(safe_dir)
    want = synthetic_bands(N)
    for a, b in zip(p.arrays, want):
        assert np.array_equal(a, b)
    assert (p.height, p.width) == (N, N) and p.transform == T10
    assert p.crs.to_epsg() == 32637 and p.year == 2026
    assert p.product_meta["processing_baseline"] == "05.12"
    assert p.product_meta["radio_add_offset"] == {b: -1000 for b in S.S2_ORDER}
    jp = p.aux["safe"]["jp2"]["B02"]
    assert jp["precision"] == 15 and jp["gml"] and jp["reversible"]
    assert list(p.aux["safe"]["band_files"]) == list(S.S2_ORDER)


def test_safe_round_trip(codec, safe_dir, tmp_path):
    store = str(tmp_path / "p.georvq.zarr")
    torch.set_grad_enabled(False)
    try:
        v1.encode_safe(safe_dir, store, codec, chunk_px=CHUNK_PX,
                       strip_px=72)
    finally:
        torch.set_grad_enabled(True)
    rung = (3, 2)
    root = v1.reconstruct_safe(store, str(tmp_path / "out"), quality=rung,
                               codec=codec)
    assert os.path.basename(root) == os.path.basename(safe_dir)
    a, b = _tree(safe_dir), _tree(root)
    assert sorted(b) == sorted(list(a) + [S.MARKER])
    files = S.band_files(safe_dir)
    bands = set(files.values())
    for k, v in a.items():
        if k not in bands and k != "manifest.safe":
            assert b[k] == v, k
    want = direct_decode(codec, synthetic_bands(N), rung)
    for g, f, names in S.GRIDS:
        for i, band in enumerate(names):
            p0 = os.path.join(safe_dir, files[band])
            p1 = os.path.join(root, files[band])
            with rasterio.open(p0) as s0, rasterio.open(p1) as s1:
                assert (s1.width, s1.height, s1.count, s1.dtypes) == \
                    (s0.width, s0.height, s0.count, s0.dtypes)
                assert s1.transform == s0.transform and s1.crs == s0.crs
                assert s1.nodata is None
                got = s1.read(1)
            assert np.array_equal(got, np.minimum(want[g][i], 32767)), band
            p = S.jp2_params(p1)
            q = S.jp2_params(p0)
            for k in ("tile_width", "tile_height", "levels", "precision",
                      "progression", "codeblock", "reversible", "gml"):
                assert p[k] == q[k], (band, k)
    man = open(os.path.join(root, "manifest.safe")).read()
    assert "georvq reconstruction" in man.splitlines()[1]
    for rel, (size, name, h) in S.manifest_entries(man).items():
        f = os.path.join(root, rel)
        assert size == os.path.getsize(f) and h == S.file_checksum(f, name)
    marker = json.load(open(os.path.join(root, S.MARKER)))
    assert marker["quality"] == [3, 2] and sorted(marker["band_images"]) == \
        sorted(bands)
    with georvq.open(store, codec=codec) as ds:
        assert ds.assets["safe_meta"] == S.safe_meta_zip(safe_dir)
        assert ds.header["product"]["safe"] == os.path.basename(safe_dir)


def test_onnx_decode_matches_torch(codec, tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from georvq.inference.store_codec import export_onnx, make_onnx_codec
    arrays = synthetic_bands(N, seed=2)
    out = str(tmp_path / "crop.georvq.zarr")
    torch.set_grad_enabled(False)
    try:
        v1.encode_product(_Crop(arrays), out, codec, chunk_px=CHUNK_PX,
                          stats=False)
        meta = export_onnx(codec, str(tmp_path / "onnx"))
    finally:
        torch.set_grad_enabled(True)
    assert meta["bitstream_id"] == codec.bitstream_id
    oc = make_onnx_codec(v1.SPEC, str(tmp_path / "onnx"))
    assert (oc.bitstream_id, oc.decoder_id) == (codec.bitstream_id,
                                                codec.decoder_id)
    for rung in [(3, 0), (3, 3)]:
        with georvq.open(out, codec=codec) as a, georvq.open(out, codec=oc) as b:
            ta, tb = a.read_grids(quality=rung), b.read_grids(quality=rung)
        for g, _, _ in S.GRIDS:
            d = np.abs(ta[g].astype(int) - tb[g].astype(int))
            assert d.max() <= 1 and (d == 0).mean() > 0.99, (rung, g)
