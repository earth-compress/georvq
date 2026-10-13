"""AE v2 bundle resolution: `georvq.ae.open(path)` finds the bundle a file
names, through GEORVQ_AE_BUNDLE, the bundle cache, or a download from a
trusted release, and refuses one for another bitstream. The hub is
mocked; no test touches the network."""
import json
import os
import shutil

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
huggingface_hub = pytest.importorskip("huggingface_hub")

import georvq.ae  # noqa: E402
from georvq.ae import v2, v2_resolve  # noqa: E402
from georvq.ae.v2_bundle import build_bundle, load_bundle  # noqa: E402
from georvq.ae.v2_resolve import BundleError  # noqa: E402
from georvq.config import load_config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference.georvq_parquet import RVQParquetReader  # noqa: E402
from georvq.inference.mean_codec import RVQPCAMeanChannel  # noqa: E402
from georvq.repro import save_checkpoint  # noqa: E402

H, WD = 40, 48
CHUNK_PX = {"base": 64, "mid": 32, "fine": 16}
RUNG = "3-2-1"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No inherited bundle or cache settings, a private default cache,
    a hub that fails if it is reached, and no GPU, so the default device
    is cpu."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for k in (v2_resolve.ENV_BUNDLE, v2_resolve.ENV_CACHE):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    def offline(*a, **kw):
        raise AssertionError("snapshot_download called")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("aev2resolve")
    torch.manual_seed(0)
    cfg = load_config("configs/smoke.yaml", [
        "model.bands=64", "model.window=16",
        "model.tiers=['1:16:32:3','2:8:32:2','4:8:16:2']",
        "model.min_depths=[1,0,0]", "model.vq.book_bits=8",
        "model.stats_cond={subtract_mean: true, cond_encoder: true, "
        "cond_decoder: true, bits_per_band: 6}"])
    model = build_model(cfg, "cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    for rvq in model.rvqs:
        rvq.project_books()
    model.eval()
    ckpt = tmp / "tiny.pt"
    save_checkpoint(ckpt, model, cfg, 7)
    rng = np.random.default_rng(0)
    means = rng.normal(size=(256, 64)).astype(np.float32) * 0.05
    books = means[:8][None].copy()
    a = ((means[:, None] - books[0][None]) ** 2).sum(-1).argmin(1)
    channel = RVQPCAMeanChannel.fit_residual(books, means - books[0][a],
                                             total_bits=96)
    npz = tmp / "mean.npz"
    channel.save(npz)
    bundle = tmp / "bundle"
    build_bundle(ckpt, npz, bundle)
    codec = load_bundle(bundle)

    src = tmp / "tile.tif"
    t = rasterio.transform.Affine(10.0, 0.0, 500000.0, 0.0, 10.0, 4000000.0)
    px = rng.integers(-127, 128, size=(64, H, WD), dtype=np.int8)
    px[:, :5, :7] = v2.NODATA
    with rasterio.open(src, "w", driver="GTiff", width=WD, height=H, count=64,
                       dtype="int8", nodata=v2.NODATA, crs="EPSG:32633",
                       transform=t) as ds:
        ds.write(px)

    def encode(name, **kw):
        out = tmp / name
        v2.encode_tile(src, out, codec, stats=False, chunk_px=CHUNK_PX,
                       strip_px=16, **kw)
        return out

    released = encode("released.georvq.parquet", revision="abc123")
    with v2.open_dataset(released, codec=codec) as ds:
        want = ds.read(quality=RUNG)
    return {"tmp": tmp, "ckpt": ckpt, "npz": npz, "bundle": bundle,
            "codec": codec, "bid": codec.bitstream_id, "encode": encode,
            "released": released, "want": want}


def _cache_with(tiny, tmp_path, bundle=None):
    root = tmp_path / "cache"
    entry = root / "bundles" / tiny["bid"]
    entry.parent.mkdir(parents=True)
    shutil.copytree(bundle or tiny["bundle"], entry)
    return root, entry


def _fake_hub(source, calls):
    """A snapshot_download that copies `source` into local_dir (below
    the first allow pattern's directory, when there is one)."""
    def snapshot_download(repo_id, repo_type=None, revision=None,
                          local_dir=None, allow_patterns=None, **kw):
        calls.append({"repo_id": repo_id, "repo_type": repo_type,
                      "revision": revision, "allow_patterns": allow_patterns})
        dst = local_dir
        if allow_patterns:
            dst = os.path.join(local_dir, allow_patterns[0].rsplit("/", 1)[0])
        shutil.copytree(source, dst, dirs_exist_ok=True)
        return local_dir
    return snapshot_download


def test_header_names_the_release(tiny):
    with RVQParquetReader(tiny["released"]) as rd:
        rel = rd.header["codec"]["release"]
    assert rel == {"name": "ae-v2", "bitstream_id": tiny["bid"],
                   "decoder_id": tiny["codec"].decoder_id,
                   "uri": "hf://earth-compress/georvq-ae-v2",
                   "revision": "abc123"}


def test_open_from_the_cache_with_no_codec_arguments(tiny, tmp_path):
    root, _ = _cache_with(tiny, tmp_path)
    with georvq.ae.open(tiny["released"], cache_dir=root) as ds:
        assert isinstance(ds.codec, v2.AEV2Codec)
        assert ds.codec.device.type == "cpu"
        assert np.array_equal(ds.read(quality=RUNG), tiny["want"])


def test_default_backend_and_device(monkeypatch):
    assert v2_resolve.default_backend() == "torch"
    assert v2_resolve.default_device("torch") == "cpu"
    assert v2_resolve.default_device("onnx") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert v2_resolve.default_device("torch") == "cuda"
    assert v2_resolve.default_device("onnx") == "cpu"


def test_default_cache_follows_xdg_and_georvq_cache(tiny, tmp_path,
                                                    monkeypatch):
    assert v2_resolve.bundle_path("x") == str(
        tmp_path / "xdg" / "georvq" / "bundles" / "x")
    root, _ = _cache_with(tiny, tmp_path)
    monkeypatch.setenv(v2_resolve.ENV_CACHE, str(root))
    with georvq.ae.open(tiny["released"], device="cpu") as ds:
        assert np.array_equal(ds.read(quality=RUNG), tiny["want"])


def test_open_from_the_environment(tiny, monkeypatch):
    monkeypatch.setenv(v2_resolve.ENV_BUNDLE, str(tiny["bundle"]))
    with georvq.ae.open(tiny["released"], device="cpu") as ds:
        assert np.array_equal(ds.read(quality=RUNG), tiny["want"])


def test_metadata_needs_no_bundle(tiny):
    with georvq.ae.open(tiny["released"]) as ds:
        assert (ds.height, ds.width, ds.count) == (H, WD, 64)


def test_bundle_for_another_bitstream_is_refused(tiny, tmp_path,
                                                 monkeypatch):
    other = tmp_path / "other"
    shutil.copytree(tiny["bundle"], other)
    meta = json.loads((other / "bundle.json").read_text())
    meta["bitstream_id"] = "0123456789abcdef"
    (other / "bundle.json").write_text(json.dumps(meta))

    root, _ = _cache_with(tiny, tmp_path, other)
    with georvq.ae.open(tiny["released"], cache_dir=root) as ds:
        with pytest.raises(BundleError, match="0123456789abcdef.*cannot "
                                              "decode this file"):
            ds.read()
    with pytest.raises(BundleError, match="given as bundle="):
        georvq.ae.open(tiny["released"], bundle=other)
    monkeypatch.setenv(v2_resolve.ENV_BUNDLE, str(other))
    with georvq.ae.open(tiny["released"], cache_dir=root) as ds:
        with pytest.raises(BundleError, match="GEORVQ_AE_BUNDLE"):
            ds.read()


def test_untrusted_release_needs_trust_remote(tiny, tmp_path, monkeypatch):
    f = tiny["encode"]("untrusted.georvq.parquet",
                       uri="hf://someone-else/georvq-ae-v2")
    root = tmp_path / "cache"
    with georvq.ae.open(f, cache_dir=root) as ds:
        with pytest.raises(BundleError, match="trust_remote=True") as err:
            ds.read()
    assert "GEORVQ_AE_BUNDLE" in str(err.value)
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        _fake_hub(tiny["bundle"], calls))
    with georvq.ae.open(f, cache_dir=root, trust_remote=True,
                        device="cpu") as ds:
        assert np.array_equal(ds.read(quality=RUNG), tiny["want"])
    assert calls[0]["repo_id"] == "someone-else/georvq-ae-v2"


def test_trusted_release_is_downloaded_into_the_cache(tiny, tmp_path,
                                                      monkeypatch):
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        _fake_hub(tiny["bundle"], calls))
    root = tmp_path / "cache"
    with georvq.ae.open(tiny["released"], cache_dir=root) as ds:
        assert np.array_equal(ds.read(quality=RUNG), tiny["want"])
    assert calls == [{"repo_id": "earth-compress/georvq-ae-v2",
                      "repo_type": "model", "revision": "abc123",
                      "allow_patterns": None}]
    entry = root / "bundles" / tiny["bid"]
    assert json.loads((entry / "bundle.json").read_text())["bitstream_id"] \
        == tiny["bid"]
    assert sorted(p.name for p in entry.parent.iterdir()) == [tiny["bid"]]
    # the second open finds the cache entry
    with georvq.ae.open(tiny["released"], cache_dir=root) as ds:
        ds.read(quality=RUNG)
    assert len(calls) == 1


def test_release_below_the_repository_root(tiny, tmp_path, monkeypatch):
    f = tiny["encode"]("subdir.georvq.parquet",
                       uri="hf://earth-compress/bundles/ae/v2")
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        _fake_hub(tiny["bundle"], calls))
    with georvq.ae.open(f, cache_dir=tmp_path / "c", device="cpu") as ds:
        assert np.array_equal(ds.read(quality=RUNG), tiny["want"])
    assert calls[0]["repo_id"] == "earth-compress/bundles"
    assert calls[0]["allow_patterns"] == ["ae/v2/*"]


def test_tampered_download_is_not_cached(tiny, tmp_path, monkeypatch):
    bad = tmp_path / "bad"
    shutil.copytree(tiny["bundle"], bad)
    with open(bad / "mean_channel.npz", "ab") as fh:
        fh.write(b"\0")
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        _fake_hub(bad, []))
    root = tmp_path / "cache"
    with georvq.ae.open(tiny["released"], cache_dir=root) as ds:
        with pytest.raises(BundleError, match="sha256"):
            ds.read()
    assert list((root / "bundles").iterdir()) == []


def test_file_without_a_release(tiny, tmp_path):
    f = tiny["encode"]("legacy.georvq.parquet", release=None)
    with RVQParquetReader(f) as rd:
        assert rd.header["codec"]["release"] is None
    with georvq.ae.open(f, model=str(tiny["ckpt"]), channel=str(tiny["npz"]),
                        device="cpu") as ds:
        assert np.array_equal(ds.read(quality=RUNG), tiny["want"])
    with georvq.ae.open(f, bundle=tiny["bundle"], device="cpu") as ds:
        assert np.array_equal(ds.read(quality=RUNG), tiny["want"])
    with georvq.ae.open(f, cache_dir=tmp_path / "empty") as ds:
        with pytest.raises(BundleError, match="names no release"):
            ds.read()
    root, _ = _cache_with(tiny, tmp_path)
    with georvq.ae.open(f, cache_dir=root, device="cpu") as ds:
        assert np.array_equal(ds.read(quality=RUNG), tiny["want"])


def test_bundle_without_onnx_refuses_the_onnx_backend(tiny):
    with pytest.raises(BundleError, match="no ONNX export"):
        georvq.ae.open(tiny["released"], bundle=tiny["bundle"],
                       backend="onnx")


@pytest.fixture(scope="module")
def onnx_bundle(tiny):
    pytest.importorskip("onnxruntime")
    out = tiny["tmp"] / "bundle_onnx"
    meta = build_bundle(tiny["ckpt"], tiny["npz"], out, onnx=True)
    return out, meta


def test_bundle_with_onnx_serves_both_backends(tiny, onnx_bundle, tmp_path):
    out, meta = onnx_bundle
    assert meta["onnx"] == "onnx" and meta["bitstream_id"] == tiny["bid"]
    assert "onnx/decoder.onnx" in meta["files"]
    root, _ = _cache_with(tiny, tmp_path, out)
    with georvq.ae.open(tiny["released"], cache_dir=root,
                        backend="onnx") as ds:
        assert isinstance(ds.codec, v2.AEV2OnnxCodec)
        got = ds.read(quality=RUNG)
    d = np.abs(got.astype(int) - tiny["want"].astype(int))
    assert d.max() <= 1 and (d > 0).mean() < 1e-3
    with georvq.ae.open(tiny["released"], cache_dir=root,
                        device="cpu") as ds:
        assert np.array_equal(ds.read(quality=RUNG), tiny["want"])


def test_cli_fetch_and_decode_without_model(tiny, tmp_path, monkeypatch,
                                            capsys):
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        _fake_hub(tiny["bundle"], calls))
    root = tmp_path / "cache"
    entry = str(root / "bundles" / tiny["bid"])
    assert v2.main(["fetch", str(tiny["released"]),
                    "--cache-dir", str(root)]) == 0
    assert capsys.readouterr().out.strip() == entry
    assert v2.main(["fetch", tiny["bid"], "--cache-dir", str(root)]) == 0
    assert capsys.readouterr().out.strip() == entry
    assert len(calls) == 1
    assert v2.main(["info", str(tiny["released"]),
                    "--cache-dir", str(root)]) == 0
    assert f"bundle     {entry}" in capsys.readouterr().out
    out = tmp_path / "a.npy"
    assert v2.main(["decode", str(tiny["released"]), str(out), "--device",
                    "cpu", "--cache-dir", str(root), "--quality", RUNG]) == 0
    assert np.array_equal(np.load(out), tiny["want"])


def test_cli_reports_a_missing_bundle(tiny, tmp_path, capsys):
    f = tiny["encode"]("legacy_cli.georvq.parquet", release=None)
    assert v2.main(["decode", str(f), str(tmp_path / "a.npy"),
                    "--cache-dir", str(tmp_path / "none")]) == 1
    assert "names no release" in capsys.readouterr().err


@pytest.mark.parametrize("uri", ["https://x/y", "hf://org", "hf://a/../b",
                                 "hf://org/re po"])
def test_malformed_uris_are_refused(uri):
    with pytest.raises(BundleError):
        v2_resolve.parse_hf_uri(uri)


def test_bitstream_id_must_be_a_plain_name():
    with pytest.raises(BundleError, match="cache key"):
        v2_resolve.file_bitstream_id({"codec": {"bitstream_id": "../x"}})


def test_without_torch_the_onnx_export_is_used(tiny, onnx_bundle, tmp_path):
    import subprocess
    import sys
    from tests.test_onnx_stats import NO_TORCH
    out, _ = onnx_bundle
    root, _ = _cache_with(tiny, tmp_path, out)
    with georvq.ae.open(tiny["released"], cache_dir=root,
                        backend="onnx") as ds:
        np.save(tmp_path / "want.npy", ds.read(quality=RUNG))
    code = NO_TORCH + f"""
import numpy as np
import georvq.ae
from georvq.ae import v2
with georvq.ae.open({str(tiny["released"])!r}, cache_dir={str(root)!r}) as ds:
    assert isinstance(ds.codec, v2.AEV2OnnxCodec)
    got = ds.read(quality={RUNG!r})
assert np.array_equal(got, np.load({str(tmp_path / "want.npy")!r}))
assert "torch" not in sys.modules
print("ok")
"""
    env = {k: v for k, v in os.environ.items()
           if k not in (v2_resolve.ENV_BUNDLE, v2_resolve.ENV_CACHE)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=os.getcwd(), env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
