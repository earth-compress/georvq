"""The shared store codec and reader under a release that is not AE: 8
float bands with NaN nodata, a uniform mean channel, four tiers (named
tier0..tier3), float output with its own nodata value, an L1 quality
error with a p90 statistic, and a north-up product whose size is not a
multiple of the window. Also the `assets` group, and `georvq.open` with
no arguments through a bundle in the cache."""
import json
import os
import warnings

import numpy as np
import pytest
import torch

pytest.importorskip("rasterio")
zarr = pytest.importorskip("zarr")
from rasterio.crs import CRS  # noqa: E402
from rasterio.windows import Window  # noqa: E402

import georvq  # noqa: E402
from georvq.config import load_config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference import bundle as bundle_mod  # noqa: E402
from georvq.inference.api import Codec  # noqa: E402
from georvq.inference.mean_codec import MeanChannel  # noqa: E402
from georvq.inference.records import RecordReader  # noqa: E402
from georvq.inference.store_codec import (ReleaseSpec, StoreCodec,  # noqa: E402
                                          encode_product, quality_code,
                                          quality_value)
from georvq.inference.store_dataset import open_dataset  # noqa: E402

H, WD, WIN, BANDS = 50, 37, 16, 8
OUT_NODATA = -9999.0
FORMAT = "georvq-test/v1"


def _prep(model, arrays, device):
    a = torch.from_numpy(np.ascontiguousarray(arrays[0])).to(device)
    valid = torch.isfinite(a).all(1)
    x = torch.nan_to_num(a, nan=0.0) * valid[:, None]
    return x.float(), valid


def _raw_valid(arrays):
    return np.isfinite(arrays[0]).all(0)


def _invalidate(arrays, mask):
    a = arrays[0]
    a[:, mask] = np.nan
    return (a,)


def _post_np(out, valid, scale=1.0):
    o = (np.asarray(out, np.float32) * np.float32(scale)).astype(np.float32)
    o[np.broadcast_to(~valid[:, None], o.shape)] = OUT_NODATA
    return o


def _post(out, valid, scale=1.0):
    return _post_np(out.cpu().numpy(), valid.cpu().numpy(), scale)


def _l1(out, x):
    return (out - x).abs().mean(1)


SPEC = ReleaseSpec(
    name="test-v1", codec_format=FORMAT, product_kind="test-tile",
    bundle_format="georvq-test/v1-bundle", onnx_meta_key="test_v1",
    bands=BANDS, prep=_prep, raw_valid=_raw_valid, invalidate=_invalidate,
    pad=(np.nan,), postprocess=_post, postprocess_np=_post_np,
    out_dtype="float32", out_nodata=OUT_NODATA, chunk_px=(32, 32, 16),
    strip_px=32, quality_stats=("mean", "p90"), quality_error=_l1,
    quality_error_text="mean |decoded - source| over bands",
    output_options={"scale": 1.0},
    band_names=tuple(f"b{i}" for i in range(BANDS)),
    licence={"spdx": "CC0-1.0"})


class Product:
    """A north-up float product as a ProductSource."""

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[1:]
        self.transform = (10.0, 0.0, 300000.0, 0.0, -10.0, 5000000.0)
        self.crs = CRS.from_epsg(32633)
        self.orientation = "north-up"
        self.product_id = "synthetic"
        self.source = "synthetic.npy"
        self.product_meta = {"count": BANDS, "dtype": "float32", "nodata": None}
        self.aux = None

    def read_strip(self, y0, y1):
        return (self.a[:, y0:y1].copy(),)


def _pixels():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:WD] / 9.0
    a = np.stack([np.sin(yy * (1 + k / 4)) * np.cos(xx * (1 + k / 3))
                  for k in range(BANDS)]).astype(np.float32)
    a += 0.05 * rng.normal(size=a.shape).astype(np.float32)
    a[:, 3:9, 4:30] = np.nan          # nodata inside windows
    a[:, 16:32, 16:32] = np.nan       # one whole window
    a[2, 40, 5] = np.nan              # one band missing: the pixel is invalid
    return a


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("storecodec")
    torch.manual_seed(0)
    cfg = load_config("configs/smoke.yaml", [
        "data.source=tessera", f"model.bands={BANDS}", f"model.window={WIN}",
        "model.tiers=['1:16:32:2','2:8:32:2','4:8:16:2','8:8:16:2']",
        "model.min_depths=[1,0,0,0]", "model.vq.book_bits=8",
        "model.stats_cond={subtract_mean: true, cond_encoder: true, "
        "cond_decoder: true, bits_per_band: 4}"])
    model = build_model(cfg, "cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    for rvq in model.rvqs:
        rvq.project_books()
    model.eval()
    a = _pixels()
    x, v = _prep(model, (a[None, :, :48, :32]
                         .reshape(1, BANDS, 3, WIN, 2, WIN)
                         .transpose(0, 2, 4, 1, 3, 5)
                         .reshape(-1, BANDS, WIN, WIN),), "cpu")
    means = model.patch_stats(x, v).numpy()[v.reshape(len(v), -1).any(1).numpy()]
    channel = MeanChannel.fit(means, 4)
    codec = StoreCodec(model, channel, spec=SPEC)
    blob = os.urandom(3000)
    out = tmp / "synthetic.georvq.zarr"
    encode_product(Product(a), out, codec, release="test-v1",
                   uri="hf://earth-compress/test",
                   assets={"meta": (blob, "application/zip", "meta.zip"),
                           "empty": (b"", "text/plain", "empty.txt")})
    bundle = tmp / "bundle"
    bundle_mod.write_bundle(bundle, model, cfg, channel, SPEC)
    return {"tmp": tmp, "codec": codec, "a": a, "store": out, "blob": blob,
            "bundle": bundle, "model": model, "channel": channel}


def _direct(built, depths):
    """Codec on the coded mean, window by window, then the spec's output."""
    codec, a = built["codec"], built["a"]
    gh, gw = -(-H // WIN), -(-WD // WIN)
    p = np.full((BANDS, gh * WIN, gw * WIN), np.nan, np.float32)
    p[:, :H, :WD] = a
    win = p.reshape(BANDS, gh, WIN, gw, WIN).transpose(1, 3, 0, 2, 4)
    x, v = _prep(None, (win.reshape(-1, BANDS, WIN, WIN),), "cpu")
    mu = codec.model.patch_stats(x, v).numpy()
    m = torch.from_numpy(codec.channel.round_trip(mu))
    c = Codec(codec.model)
    y = c.decode(c.encode(x, valid=v, mean=m), depths=list(depths), valid=v)
    out = _post(y, v).reshape(gh, gw, BANDS, WIN, WIN).transpose(2, 0, 3, 1, 4)
    out = out.reshape(BANDS, gh * WIN, gw * WIN)[:, :H, :WD]
    return out, x, v, y


def test_header_carries_what_a_reader_needs(built):
    with RecordReader(built["store"]) as rd:
        h = rd.header
    assert h["codec"]["format"] == FORMAT
    assert h["codec"]["mean_channel"] == {"kind": "uniform", "bits": 32}
    assert h["output"] == {"dtype": "float32", "nodata": OUT_NODATA,
                           "count": BANDS,
                           "band_names": [f"b{i}" for i in range(BANDS)]}
    assert h["licence"] == {"spdx": "CC0-1.0"}
    assert h["quality"]["stats"] == ["mean", "p90"]
    assert [t["name"] for t in h["layout"]["tiers"]] == [
        "tier0", "tier1", "tier2", "tier3"]
    assert h["layout"]["product_px"] == [H, WD]


@pytest.mark.parametrize("depths", [(2, 2, 2, 2), (1, 0, 0, 0), (2, 2, 1, 0)])
def test_store_read_equals_direct_decode(built, depths):
    want = _direct(built, depths)[0]
    with open_dataset(built["store"], SPEC, codec=built["codec"]) as ds:
        assert ds.dtypes == ("float32",) * BANDS and ds.nodata == OUT_NODATA
        assert ds.descriptions == tuple(f"b{i}" for i in range(BANDS))
        got = ds.read(quality=depths)
    assert got.dtype == np.float32 and got.shape == (BANDS, H, WD)
    invalid = ~np.isfinite(built["a"]).all(0)
    assert (got[:, invalid] == OUT_NODATA).all()
    assert (got[:, ~invalid] != OUT_NODATA).all()
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


def test_window_read_and_options(built):
    with open_dataset(built["store"], SPEC, codec=built["codec"]) as ds:
        full = ds.read()
        part = ds.read([2, 5], window=Window(3, 20, 30, 25))
        np.testing.assert_allclose(part, full[[1, 4], 20:45, 3:33], atol=1e-5)
        m = ds.read(masked=True)
        assert m.mask[:, 5, 10].all() and not m.mask[:, 45, 30].any()
    with open_dataset(built["store"], SPEC, codec=built["codec"],
                      options={"scale": 2.0}) as ds:
        twice = ds.read()
    valid = full != OUT_NODATA
    np.testing.assert_allclose(twice[valid], 2 * full[valid], rtol=1e-6)
    with pytest.raises(TypeError, match="unknown output option"):
        open_dataset(built["store"], SPEC, options={"units": "x"})


def test_quality_uses_the_spec_error(built):
    codec = built["codec"]
    with open_dataset(built["store"], SPEC, codec=codec) as ds:
        gh, gw = ds.reader.window_grid
        q = ds.quality_codes(0, gh, 0, gw).reshape(gh * gw, 2, -1)
        n = len(ds.rungs)
        assert q.shape[-1] == n
        live = ~ds.reader.dead.reshape(-1)
        for k, r in enumerate(ds.rungs):
            _, x, v, y = _direct(built, r)
            e = _l1(y, x).reshape(len(x), -1)
            e = torch.where(v.reshape(len(x), -1), e, torch.nan)
            want_mean = quality_code(e.nanmean(1).numpy())
            want_p90 = quality_code(torch.nanquantile(e, 0.9, dim=1).numpy())
            vw = v.reshape(len(x), -1).any(1).numpy() & live
            assert np.abs(q[vw, 0, k].astype(int) - want_mean[vw]).max() <= 1
            assert np.abs(q[vw, 1, k].astype(int) - want_p90[vw]).max() <= 1
        # a bound on the spec's own statistic resolves per window
        rungs = ds.window_rungs("p90<=0.5")
        assert rungs.shape == (gh, gw)
        ds.read(quality="p90<=0.5")
        with pytest.raises(ValueError, match="p99"):
            ds.read(quality="p99<=0.5")


def test_bound_reads_are_conservative(built):
    """Quality codes round up, so a window read to a bound that its
    quality values say a rung meets has an error within the bound."""
    a = built["a"]
    gh, gw = -(-H // WIN), -(-WD // WIN)

    def window_errors(out):
        p = np.full((BANDS, gh * WIN, gw * WIN), np.nan, np.float32)
        q = np.full_like(p, np.nan)
        p[:, :H, :WD], q[:, :H, :WD] = a, out
        e = np.abs(p - q).mean(0)                    # NaN where invalid
        with warnings.catch_warnings():              # windows with no valid pixel
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(e.reshape(gh, WIN, gw, WIN).transpose(0, 2, 1, 3)
                              .reshape(gh, gw, -1), -1)

    with open_dataset(built["store"], SPEC, codec=built["codec"]) as ds:
        n = len(ds.rungs)
        errs = np.stack([window_errors(ds.read(quality=r)) for r in ds.rungs], -1)
        live = ~np.isnan(errs[..., 0])
        codes = ds.quality_codes(0, gh, 0, gw)[..., :n]
        assert (quality_value(codes)[live] >= errs[live] * (1 - 1e-5)).all()
        for bound in np.quantile(errs[live], [0.2, 0.5, 0.8]):
            k = ds.window_rungs(f"mean<={bound}")
            met = (np.take_along_axis(quality_value(codes), k[..., None], -1)[..., 0]
                   <= bound) & live
            assert met.any()
            got = window_errors(ds.read(quality=f"mean<={bound}"))
            assert (got[met] <= bound * (1 + 1e-5)).all()


def test_a_read_of_several_strips_is_fetched_once(built, monkeypatch):
    """A read spanning several decode strips fetches its records with one
    zarr-python read per array, and decodes as it does strip by strip."""
    from georvq.inference import store_dataset
    from georvq.inference.records import Trace
    monkeypatch.setattr(store_dataset, "STRIP_PX", WIN)
    tr = Trace()
    with open_dataset(built["store"], SPEC, codec=built["codec"], trace=tr) as ds:
        monkeypatch.setattr(ds, "_prefetch", lambda *a: None)
        nopf = ds.read(quality=(2, 2, 1, 0))
    assert sum(e["kind"] == "get" for e in tr.events) > 4
    tr = Trace()
    with open_dataset(built["store"], SPEC, codec=built["codec"], trace=tr) as ds:
        got = ds.read(quality=(2, 2, 1, 0))
        gets = [e["label"] for e in tr.events if e["kind"] == "get"]
    assert np.array_equal(got, nopf)
    assert len(gets) == len(set(gets)) <= 4          # base, mask, tier1, tier2


def test_blank_window_reads_nothing(built):
    """Window (1, 1) has no valid pixel and shares its base chunk with
    valid windows: its records are zero, reading it fetches nothing, and
    it decodes to nodata."""
    with open_dataset(built["store"], SPEC, codec=built["codec"]) as ds:
        g = ds.reader.group
        assert not g["base"][1, 1].any() and not g["tier1"][1, 1].any()
        assert g["base"][0, 0].any()
        assert ds.reader.plan(1, 2, 1, 2, quality=True) == []
        got = ds.read(window=Window(16, 16, 16, 16))
        assert ds.reader.stats["values"] == 0
        assert (got == OUT_NODATA).all()


def test_strip_without_valid_pixels_encodes(built, tmp_path):
    """A whole encode strip (rows 0..32) without a valid pixel, as at a
    swath edge: its windows are stored blank and decode to nodata, and
    the rest of the product reads as when the strip held data."""
    a = built["a"].copy()
    a[:, :32] = np.nan
    out = tmp_path / "edge.georvq.zarr"
    encode_product(Product(a), out, built["codec"], release="test-v1")
    with open_dataset(out, SPEC, codec=built["codec"]) as ds, \
            open_dataset(built["store"], SPEC, codec=built["codec"]) as full:
        g = ds.reader.group
        assert not g["base"][:2].any() and not g["tier1"][:2].any()
        assert ds.reader.plan(0, 2, 0, 3, quality=True) == []
        assert (ds.read(window=Window(0, 0, WD, 32)) == OUT_NODATA).all()
        rest = Window(0, 32, WD, H - 32)
        assert np.array_equal(ds.read(window=rest), full.read(window=rest))


def test_assets_round_trip_and_are_checked(built, tmp_path):
    import shutil
    with open_dataset(built["store"], SPEC) as ds:
        assert sorted(ds.assets) == ["empty", "meta"]
        assert ds.assets.info("meta") == {
            "media_type": "application/zip", "filename": "meta.zip",
            "size": 3000, "sha256": __import__("hashlib").sha256(
                built["blob"]).hexdigest()}
        assert ds.assets["meta"] == built["blob"]
        assert ds.assets["empty"] == b""
        with pytest.raises(KeyError):
            ds.assets["other"]
    g = zarr.open_group(str(built["store"]), mode="r")
    a = g["assets"]["meta"]
    assert a.shape == (3000,) and a.chunks == (3000,) and a.dtype == np.uint8
    assert a.metadata.to_dict()["codecs"][0]["name"] == "bytes"
    assert len(a.metadata.to_dict()["codecs"]) == 1
    bad = tmp_path / "bad.georvq.zarr"
    shutil.copytree(built["store"], bad)
    chunk = bad / "assets" / "meta" / "c" / "0"
    raw = bytearray(chunk.read_bytes())
    raw[10] ^= 1
    chunk.write_bytes(bytes(raw))
    with open_dataset(bad, SPEC) as ds:
        with pytest.raises(ValueError, match="sha256"):
            ds.assets["meta"]


def test_georvq_open_needs_no_arguments(built, tmp_path, monkeypatch):
    from georvq import releases
    from georvq.inference.bundle_resolve import BundleError
    monkeypatch.setitem(releases.RELEASES, FORMAT, "tests.test_store_codec")
    root = tmp_path / "cache"
    entry = root / "bundles" / built["codec"].bitstream_id
    entry.parent.mkdir(parents=True)
    import shutil
    shutil.copytree(built["bundle"], entry)
    monkeypatch.setenv("GEORVQ_CACHE", str(root))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    want = _direct(built, (2, 2, 1, 0))[0]
    with georvq.open(str(built["store"])) as ds:
        assert ds.spec is SPEC and ds.count == BANDS
        assert ds.crs == CRS.from_epsg(32633)
        got = ds.read(quality="2-2-1-0")
        assert type(ds.codec) is StoreCodec
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)
    with georvq.open(str(built["store"]), quality="2-2-1-0", scale=2.0) as ds:
        valid = got != OUT_NODATA
        np.testing.assert_allclose(ds.read()[valid], 2 * got[valid], rtol=1e-6)
    # a decoder other than the store's is refused unless forced
    import copy
    other = copy.copy(built["codec"])
    other.decoder_id = "f" * 16
    with pytest.raises(ValueError, match="force=True"):
        georvq.open(str(built["store"]), codec=other)
    with georvq.open(str(built["store"]), codec=other, force=True) as ds:
        assert ds.read(quality="1-0-0-0").shape == (BANDS, H, WD)
    # an unknown format is refused, and nothing is imported from the store
    monkeypatch.delitem(releases.RELEASES, FORMAT)
    with pytest.raises(ValueError, match="not one georvq reads"):
        georvq.open(str(built["store"]))
    monkeypatch.setitem(releases.RELEASES, FORMAT, "tests.test_store_codec")
    monkeypatch.setenv("GEORVQ_CACHE", str(tmp_path / "empty"))
    try:
        import huggingface_hub
    except ImportError:
        pass
    else:
        def offline(*a, **kw):
            raise OSError("offline")
        monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    with georvq.open(str(built["store"])) as ds:
        assert ds.height == H                     # metadata needs no bundle
        with pytest.raises(BundleError, match="hf://earth-compress/test"):
            ds.read()


def test_bundle_round_trip(built):
    c = bundle_mod.load_bundle(built["bundle"], SPEC)
    assert (c.bitstream_id, c.decoder_id) == (built["codec"].bitstream_id,
                                              built["codec"].decoder_id)
    assert c.channel.kind == "uniform"
    meta = json.load(open(built["bundle"] / "bundle.json"))
    assert meta["format"] == "georvq-test/v1-bundle"
    assert meta["mean_channel"] == {"kind": "uniform", "bits": 32}
    assert [tuple(r["tuple"]) for r in meta["rungs"]][0] == (1, 0, 0, 0)


_CRS_PROBE = """
import json, sys, threading
import numpy as np
import rasterio
from georvq.inference.store_codec import crs_from_epsg

def write():
    with rasterio.open(sys.argv[1], "w", driver="GTiff", height=2, width=2,
                       count=1, dtype="uint8", crs="EPSG:32629",
                       transform=rasterio.Affine(10, 0, 0, 0, -10, 0)) as ds:
        ds.write(np.ones((1, 2, 2), np.uint8))

if sys.argv[2] == "read-first":
    t = threading.Thread(target=write)
    t.start()
    t.join()
    with rasterio.open(sys.argv[1]) as ds:
        ds.crs
print(json.dumps(crs_from_epsg(32653).to_dict(projjson=True), sort_keys=True))
"""


def test_crs_from_epsg_does_not_depend_on_earlier_reads(tmp_path):
    """Reading a GeoTIFF first leaves WGS 84 cached as a plain datum in
    the thread's PROJ context, which changed the Tessera golden store's
    PROJJSON when its test ran after another that opened a landmask. Each
    case runs in a fresh interpreter, since the cache lives as long as the
    thread."""
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.dirname(os.path.dirname(georvq.__file__))]
        + os.environ.get("PYTHONPATH", "").split(os.pathsep)))
    out = [subprocess.run([sys.executable, "-c", _CRS_PROBE,
                           str(tmp_path / "t.tif"), case], env=env,
                          check=True, capture_output=True, text=True).stdout
           for case in ("clean", "read-first")]
    assert out[0] == out[1]
    assert "datum_ensemble" in json.loads(out[0])["base_crs"]
