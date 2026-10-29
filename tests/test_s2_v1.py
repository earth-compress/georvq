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
from georvq.s2 import safe as S  # noqa: E402
from georvq.s2 import v1  # noqa: E402
from tests.s2_safe_fixture import make_safe, synthetic_bands  # noqa: E402

N = 144                                   # 10 m pixels of the test crop
WIN = 36
CHUNK_PX = {"base": 72, "tier1": 72}
T10 = (10.0, 0.0, 399960.0, 0.0, -10.0, 5800020.0)


def tiny_cfg():
    return Config.model_validate({
        "data": {"source": "s2", "store_dir": "/tmp/unused"},
        "model": {"window": WIN, "bands": 13,
                  "tiers": ["1:16:32:3", "6:8:32:3"], "min_depths": [3, 0],
                  "ff_ratio": 2.0, "tokenizer": {"dim": 16, "depth": 1},
                  "vq": {"book_bits": 8},
                  "decoder": {"dim": 32, "depth": 1, "heads": 4,
                              "head_feat": 8}},
        "log": {"run_tag": "s2v1test", "wandb": False}})


def tiny_codec():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = build_model(cfg, "cpu")
    for rvq in model.rvqs:
        rvq.project_books()
    return v1.S2V1Codec(model, v1.stats_channel())


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
    assert not any(S.is_band_file(n) or S.is_tci_file(n) for n in names)
    assert not any(n.endswith("_TCI.jp2") for n in names)
    assert any(n.endswith("_PVI.jp2") for n in names)
    assert "AUX_DATA/" in names and "manifest.safe" in names
    out = str(tmp_path / "x.SAFE")
    S.unzip_meta(a, out)
    want = {k: v for k, v in _tree(safe_dir).items()
            if not S.is_band_file(k.replace(os.sep, "/"))
            and not S.is_tci_file(k.replace(os.sep, "/"))}
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


def test_manifest_drop(safe_dir):
    import xml.etree.ElementTree as ET
    text = open(os.path.join(safe_dir, "manifest.safe")).read()
    ent = S.manifest_entries(text)
    tci = [r for r in ent if S.is_tci_file(r)]
    assert len(tci) == 1
    new = S.drop_from_manifest(text, tci)
    root = ET.fromstring(new)
    assert S.manifest_entries(new) == {k: v for k, v in ent.items()
                                       if k not in tci}
    objs = {o.get("ID") for o in root.iter("dataObject")}
    ptrs = [p.get("dataObjectID") for p in root.iter("dataObjectPointer")]
    assert sorted(ptrs) == sorted(objs) and len(objs) == len(ent) - 1
    assert text.count("\n") - new.count("\n") == 9 and "TCI" not in new
    # the one-line layout of products of baselines before 04.00
    flat = "".join(line.strip() for line in text.splitlines())
    new = S.drop_from_manifest(flat, tci)
    assert new == "".join(line.strip() for line in
                          S.drop_from_manifest(text, tci).splitlines())
    assert "TCI" not in new
    with pytest.raises(ValueError, match="lists"):
        S.drop_from_manifest(text, ["GRANULE/nope_TCI.jp2"])


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
    tci = [k for k in a if S.is_tci_file(k.replace(os.sep, "/"))]
    assert len(tci) == 1
    assert sorted(b) == sorted([k for k in a if k not in tci] + [S.MARKER])
    files = S.band_files(safe_dir)
    bands = set(files.values())
    for k, v in a.items():
        if k not in bands and k not in tci and k != "manifest.safe":
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
    import xml.etree.ElementTree as ET
    ET.fromstring(man)
    ent = S.manifest_entries(man)
    assert sorted(ent) == sorted(k for k in b if not k.endswith("/") and k
                                 not in ("manifest.safe", S.MARKER))
    for rel, (size, name, h) in ent.items():
        f = os.path.join(root, rel)
        assert size == os.path.getsize(f) and h == S.file_checksum(f, name)
    marker = json.load(open(os.path.join(root, S.MARKER)))
    assert marker["quality"] == [3, 2] and sorted(marker["band_images"]) == \
        sorted(bands)
    assert marker["omitted"] == tci
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
    # the export stores the int8 books, and they are the model's bit for bit
    with np.load(tmp_path / "onnx" / "codebooks.npz") as z:
        assert z["rvqs.0.levels.0.q"].dtype == np.int8
    for books, rvq in zip(oc.books, codec.model.rvqs):
        for b, lev in zip(books, rvq.levels):
            w = lev.codebook.weight.detach().numpy()
            assert np.array_equal(b.view(np.int32), w.view(np.int32))
    assert (oc.bitstream_id, oc.decoder_id) == (codec.bitstream_id,
                                                codec.decoder_id)
    for rung in [(3, 0), (3, 3)]:
        with georvq.open(out, codec=codec) as a, georvq.open(out, codec=oc) as b:
            ta, tb = a.read_grids(quality=rung), b.read_grids(quality=rung)
        for g, _, _ in S.GRIDS:
            d = np.abs(ta[g].astype(int) - tb[g].astype(int))
            assert d.max() <= 1 and (d == 0).mean() > 0.99, (rung, g)


def test_header_names_quality_and_licence(codec, tmp_path):
    out = str(tmp_path / "crop.georvq.zarr")
    torch.set_grad_enabled(False)
    try:
        v1.encode_product(_Crop(synthetic_bands(N, seed=3)), out, codec,
                          chunk_px=CHUNK_PX, licence=v1.licence_of(2026))
    finally:
        torch.set_grad_enabled(True)
    attrs = json.load(open(os.path.join(out, "zarr.json")))["attributes"]
    h = attrs["georvq"]
    assert h["quality"]["name"] == "rmse_dn"
    assert h["licence"]["attribution"] == \
        "Contains modified Copernicus Sentinel data 2026"
    assert attrs["proj:code"] == "EPSG:32637" and "proj:wkt2" in attrs
    assert "proj:projjson" not in attrs


def test_decoder_only_bundle(codec, tmp_path):
    from georvq.inference.bundle import write_bundle
    full = tmp_path / "full"
    write_bundle(str(full), codec.model, tiny_cfg(), codec.channel, v1.SPEC,
                 extra={"licence_stores": v1.STORES_LICENCE})
    fm = json.load(open(full / "bundle.json"))
    assert fm["variant"] == "full" and fm["release"] == "s2-v1"
    assert fm["licence_weights"] == "Apache-2.0"
    assert fm["licence_stores"] == v1.STORES_LICENCE
    assert fm["book_bits"] == 8 and fm["decoder_bundle"] == "decoder"
    assert "8-bit codebooks" in open(full / "README.md").read()
    dec = full / "decoder"
    meta = json.load(open(dec / "bundle.json"))
    assert meta["variant"] == "decoder" and meta["onnx"] is None
    assert (meta["bitstream_id"], meta["decoder_id"]) == (
        fm["bitstream_id"], fm["decoder_id"])
    with np.load(dec / "weights.npz", allow_pickle=False) as z, \
            np.load(full / "weights.npz", allow_pickle=False) as zf:
        assert not any(k.startswith("tokenizer.") for k in z.files)
        assert set(z.files) < set(zf.files)
        assert all(np.array_equal(z[k], zf[k]) and z[k].dtype == zf[k].dtype
                   for k in z.files)
    a, b = v1.load_bundle(str(full)), v1.load_bundle(str(dec))
    assert b.decode_only and not a.decode_only
    store = str(tmp_path / "crop.georvq.zarr")
    torch.set_grad_enabled(False)
    try:
        v1.encode_product(_Crop(synthetic_bands(N, seed=5)), store, a,
                          chunk_px=CHUNK_PX)
        with pytest.raises(RuntimeError, match="decoder-only"):
            v1.encode_product(_Crop(synthetic_bands(N, seed=5)),
                              str(tmp_path / "x.georvq.zarr"), b,
                              chunk_px=CHUNK_PX)
    finally:
        torch.set_grad_enabled(True)
    # the same decoder: same reads, and the store's quality is served
    with georvq.open(store, codec=a) as da, georvq.open(store, codec=b) as db:
        assert db.reader.quality_decoder == fm["decoder_id"]
        ta, tb = da.read_grids(quality=(3, 3)), db.read_grids(quality=(3, 3))
        qa = da.read_grids(quality="rmse_10m<=1000")
        qb = db.read_grids(quality="rmse_10m<=1000")
    for g, _, _ in S.GRIDS:
        assert np.array_equal(ta[g], tb[g]) and np.array_equal(qa[g], qb[g])


def test_deblocked_window_matches_whole_product(codec, tmp_path):
    from georvq.s2 import deblock as D
    from georvq.sources.s2_deblock import S2BoundaryDeblocker, deblock_s2_mosaic
    store = str(tmp_path / "crop.georvq.zarr")
    torch.set_grad_enabled(False)
    try:
        v1.encode_product(_Crop(synthetic_bands(N, seed=6)), store, codec,
                          chunk_px=CHUNK_PX, stats=False)
        torch.manual_seed(1)
        net = S2BoundaryDeblocker(width=8, cond_dim=8, depth=3,
                                  window=WIN).eval()
        for p in net.parameters():          # not the identity
            p.add_(0.05 * torch.randn_like(p))
        recipe = {"frame": 72, "halo": 36}
        entry = {"recipe": recipe, "trained_against": {
            "bitstream_id": codec.bitstream_id,
            "decoder_id": codec.decoder_id}}
        with georvq.open(store, codec=codec) as ds:
            full = ds.read_grids(quality="3-2")
            x = {g: torch.from_numpy(a.astype(np.float32) / D.SCALE)[None]
                 for g, a in full.items()}
            y = deblock_s2_mosaic(net, x, torch.tensor([[3, 2]]), recipe)
            got = D.read_deblocked(ds, (net, entry), ((36, 108), (72, 144)),
                                   quality="3-2")
            with pytest.raises(ValueError, match="fixed rung"):
                D.read_deblocked(ds, (net, entry), quality="rmse_10m<=5")
            bad = dict(entry, trained_against={"bitstream_id": "x",
                                               "decoder_id": codec.decoder_id})
            with pytest.raises(ValueError, match="trained against"):
                D.read_deblocked(ds, (net, bad), ((0, 36), (0, 36)), "3-2")
            ok = dict(bad, applies_to=[entry["trained_against"]])
            assert np.array_equal(
                D.read_deblocked(ds, (net, ok), ((36, 108), (72, 144)),
                                 quality="3-2")["10m"], got["10m"])
    finally:
        torch.set_grad_enabled(True)
    changed = 0
    for g, f, _ in S.GRIDS:
        a = y[g][0].numpy() * D.SCALE
        want = np.where((full[g] != 0).any(0), np.clip(np.rint(a), 1, 65535),
                        0).astype(np.uint16)[:, 36 // f:108 // f, 72 // f:144 // f]
        assert np.array_equal(got[g], want), g
        changed += int((want != full[g][:, 36 // f:108 // f,
                                         72 // f:144 // f]).sum())
    assert changed > 0


def test_index(codec, tmp_path):
    import pyarrow.parquet as pq
    from georvq.inference.store_index import CORE
    from georvq.s2.index import build_index
    name = "S2C_MSIL1C_20260503T084641_N0512_R107_T37UDB_20260503T104912"
    src = _Crop(synthetic_bands(N, seed=7))
    src.product_id = name
    src.product_meta = {"count": 13, "safe": name + ".SAFE",
                        "spacecraft": "Sentinel-2C",
                        "sensing_start": "2026-05-03T08:46:41.024Z",
                        "processing_baseline": "05.12"}
    root = tmp_path / "sample"
    root.mkdir()
    torch.set_grad_enabled(False)
    try:
        v1.encode_product(src, str(root / (name + ".georvq.zarr")), codec,
                          chunk_px=CHUNK_PX, stats=False)
    finally:
        torch.set_grad_enabled(True)
    out = str(tmp_path / "s2_georvq_index.parquet")
    assert v1.main(["index", str(root), out]) == 0
    t = pq.read_table(out)
    assert t.schema.names[:len(CORE.names)] == CORE.names
    row = t.to_pylist()[0]
    assert row["path"] == name + ".georvq.zarr"
    assert row["crs"] == "EPSG:32637" and row["product_id"] == name
    assert (row["mgrs_tile"], row["relative_orbit"]) == ("37UDB", 107)
    assert row["safe"] == name + ".SAFE" and row["spacecraft"] == "Sentinel-2C"
    assert row["sensing_start"].year == 2026
    assert b'"primary_column": "geometry"' in t.schema.metadata[b"geo"]


def test_cli_encode_records_source_urls(codec, safe_dir, tmp_path):
    from georvq.inference.bundle import write_bundle
    bundle = tmp_path / "bundle"
    write_bundle(str(bundle), codec.model, tiny_cfg(), codec.channel, v1.SPEC,
                 extra={"licence_stores": v1.STORES_LICENCE})
    name = S.safe_name(safe_dir)
    url = "https://example.org/Products(1234)"
    srcs = tmp_path / "sources.json"
    srcs.write_text(json.dumps({name + ".SAFE": url}))
    out = tmp_path / "sample"
    torch.set_grad_enabled(False)
    try:
        empty = tmp_path / "empty.json"
        empty.write_text("{}")
        with pytest.raises(SystemExit):
            v1.main(["encode", str(out), safe_dir, "--bundle", str(bundle),
                     "--device", "cpu", "--sources", str(empty)])
        assert not out.exists()
        assert v1.main(["encode", str(out), safe_dir, "--bundle", str(bundle),
                        "--device", "cpu", "--sources", str(srcs)]) == 0
    finally:
        torch.set_grad_enabled(True)
    p = json.load(open(out / (name + ".georvq.zarr") / "zarr.json"))
    assert p["attributes"]["georvq"]["product"]["source"] == url
    assert p["attributes"]["georvq"]["product"]["id"] == name
