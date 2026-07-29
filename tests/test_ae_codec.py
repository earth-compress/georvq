"""The deployment executables: ONNX export parity, container round-trips,
and the patch/tile CLIs end-to-end on CPU (tiny model, no store needed)."""
import json
import os

import numpy as np
import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.inference.api import Codec
from georvq.ae.cli import (codec_patch_main, codec_tile_main,
                                  export_onnx_main, prep_ae_np,
                                  read_container, requant_ae_np,
                                  write_container)
from georvq.repro import save_checkpoint

BANDS, W = 8, 16


@pytest.fixture()
def tiny_ckpt(tiny_raw_cfg, tmp_path):
    cfg = Config.model_validate(tiny_raw_cfg)
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    # spread codebooks out (random init would leave near-ties everywhere)
    for rvq in model.rvqs:
        for lev in rvq.levels:
            torch.nn.init.normal_(lev.codebook.weight, std=1.0)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, cfg, step=0)
    return str(path), model


def _windows(n=6, seed=3):
    rng = np.random.default_rng(seed)
    s = rng.integers(-127, 128, (n, BANDS, W, W), dtype=np.int8)
    s[0, 0, :4, :4] = -128        # a nodata corner exercises the mask path
    return s


def test_prep_requant_roundtrip():
    s = _windows()
    x, valid = prep_ae_np(s)
    assert x.dtype == np.float32 and valid.shape == (6, W, W)
    n = np.linalg.norm(x, axis=1)
    assert np.allclose(n[valid], 1.0, atol=1e-5)
    assert not valid[0, 0, 0] and valid[0, 8, 8]
    # requant of a prepped field is idempotent through one more prep
    s2 = requant_ae_np(x, valid)
    assert (s2[0, :, :4, :4] == -128).all()
    x2, _ = prep_ae_np(s2)
    assert np.abs(x2 - x)[np.broadcast_to(valid[:, None], x.shape)].max() \
        < 0.02


def test_container_roundtrip(tiny_ckpt, tmp_path):
    _, model = tiny_ckpt
    codec = Codec(model)
    x, _ = prep_ae_np(_windows())
    packets = codec.encode(torch.from_numpy(x))
    path = tmp_path / "c.gvq.npz"
    write_container(path, packets, [(sp.cells, sp.depth, sp.codes)
                                    for sp in model.specs],
                    {"kind": "patch", "window": W})
    header, back, _ = read_container(path)
    assert header["n_windows"] == 6
    for p, q in zip(packets, back):
        for a, b in zip(p.indices, q.indices):
            assert np.array_equal(a, b)


def test_container_mask_roundtrip(tiny_ckpt, tmp_path):
    _, model = tiny_ckpt
    codec = Codec(model)
    s = _windows()
    x, valid = prep_ae_np(s)
    packets = codec.encode(torch.from_numpy(x))
    sig = [(sp.cells, sp.depth, sp.codes) for sp in model.specs]
    path = tmp_path / "m.gvq.npz"
    write_container(path, packets, sig, {"kind": "patch"}, valid=valid)
    header, _, valid2 = read_container(path)
    assert header["mask_shape"] == list(valid.shape)
    assert np.array_equal(valid, valid2)
    # an all-valid mask is elided entirely
    write_container(path, packets, sig, {"kind": "patch"},
                    valid=np.ones_like(valid))
    header, _, valid2 = read_container(path)
    assert valid2 is None and "mask_shape" not in header


@pytest.fixture()
def onnx_dir(tiny_ckpt, tmp_path):
    pytest.importorskip("onnxruntime")
    ckpt, _ = tiny_ckpt
    out = str(tmp_path / "onnx")
    export_onnx_main(["--ckpt", ckpt, "--out", out])
    return out


def test_onnx_export_parity(tiny_ckpt, onnx_dir):
    """Same packets decode to (near-)identical pixels in both backends;
    fp32 index assignment matches torch on a spread codebook."""
    from georvq.ae.cli import OnnxBackend, TorchBackend
    ckpt, _ = tiny_ckpt
    tb, ob = TorchBackend(ckpt), OnnxBackend(onnx_dir)
    assert json.load(open(os.path.join(onnx_dir, "meta.json")))["window"] == W
    x, _ = prep_ae_np(_windows())
    pt, po = tb.encode(x), ob.encode(x)
    for a, b in zip(pt, po):
        for t in range(len(tb.specs_sig)):
            assert np.mean(np.asarray(a.indices[t])
                           == np.asarray(b.indices[t])) > 0.99
    assert np.abs(tb.decode(pt) - ob.decode(pt)).max() < 1e-3
    # truncated tuples flow through the exported decoder's Ls input
    assert np.abs(tb.decode(pt, [2, 1]) - ob.decode(pt, [2, 1])).max() < 1e-3


def test_patch_cli_roundtrip(tiny_ckpt, tmp_path):
    ckpt, model = tiny_ckpt
    s = _windows()
    np.save(tmp_path / "in.npy", s)
    gvq, out = str(tmp_path / "p.gvq.npz"), str(tmp_path / "out.npy")
    codec_patch_main(["encode", "--model", ckpt,
                      str(tmp_path / "in.npy"), gvq])
    codec_patch_main(["decode", "--model", ckpt, gvq, out])
    recon = np.load(out)
    assert recon.shape == (6, BANDS, W, W) and recon.dtype == np.float32
    # matches the API decode of the container's own packets (incl. the
    # quantized 8* sidecar) on valid pixels, zero vectors on nodata
    codec = Codec(model)
    x, valid = prep_ae_np(s)
    _, packets, _ = read_container(gvq)
    assert packets[0].coarse_residual is not None     # sidecar on by default
    ref = codec.decode(packets).numpy()
    vb = np.broadcast_to(valid[:, None], ref.shape)
    assert np.abs(recon - ref)[vb].max() < 1e-5
    assert (recon[~vb] == 0).all()
    # --int8 restores the source's exact fill value at nodata pixels
    codec_patch_main(["decode", "--model", ckpt, "--int8", gvq, out])
    r8 = np.load(out)
    assert r8.dtype == np.int8
    assert (r8[0, :, :4, :4] == -128).all()
    assert (r8[vb] != -128).all()   # requant clips data to [-127, 127]
    # truncated decode differs from full but stays unit-norm on valid px
    codec_patch_main(["decode", "--model", ckpt, "--tuple", "1,0", gvq, out])
    low = np.load(out)
    assert np.allclose(np.linalg.norm(low, axis=1)[valid], 1.0, atol=1e-4)
    assert np.abs(low - recon).max() > 1e-3


def test_tile_cli_roundtrip(tiny_ckpt, tmp_path):
    """A non-multiple-of-W raster pads, encodes to COG assets, decodes,
    and unpads back to shape, with nodata pixels restored exactly."""
    pytest.importorskip("rasterio")
    ckpt, _ = tiny_ckpt
    rng = np.random.default_rng(0)
    H, Wd = 3 * W - 5, 2 * W + 3
    tile = rng.integers(-127, 128, (BANDS, H, Wd), dtype=np.int8)
    tile[:, :6, :9] = -128
    np.save(tmp_path / "tile.npy", tile)
    adir, out = str(tmp_path / "assets"), str(tmp_path / "rec.npy")
    codec_tile_main(["encode", "--model", ckpt,
                     str(tmp_path / "tile.npy"), adir])
    from georvq.ae.cog import read_meta
    meta = read_meta(adir)
    assert meta["grid"] == [3, 3] and meta["shape"] == [BANDS, H, Wd]
    for f in ("coarse.tif", "mid.tif", "star.tif", "mask.tif",
              "metadata.json"):
        assert os.path.exists(os.path.join(adir, f)), f
    # decoder-specific quality raster: per-window mean dcos per point
    import rasterio
    from georvq.ae.cli import TorchBackend
    did = TorchBackend(ckpt).decoder_id
    with rasterio.open(os.path.join(adir, f"quality/{did}.tif")) as ds:
        assert ds.count == 3 and ds.descriptions[0] == "3-1"
        qv = ds.read() * float(ds.tags()["QUALITY_SCALE"])
        assert (qv >= 0).all() and qv.max() < 2.0
        assert qv[0].mean() >= qv[2].mean() - 1e-6   # deeper -> better
    codec_tile_main(["decode", "--model", ckpt, adir, out])
    rec = np.load(out)
    assert rec.shape == (BANDS, H, Wd) and rec.dtype == np.int8
    assert (rec[:, :6, :9] == -128).all()          # nodata restored
    assert (rec[0, 6:, 9:] != -128).all()
    codec_tile_main(["decode", "--model", ckpt, "--float32", adir, out])
    assert np.load(out).dtype == np.float32


def test_tile_cli_onnx_backend_matches(tiny_ckpt, onnx_dir, tmp_path):
    """Assets encoded by torch decode identically under the onnx backend."""
    pytest.importorskip("rasterio")
    ckpt, _ = tiny_ckpt
    rng = np.random.default_rng(1)
    tile = rng.integers(-127, 128, (BANDS, 2 * W, 2 * W), dtype=np.int8)
    np.save(tmp_path / "tile.npy", tile)
    adir = str(tmp_path / "assets")
    codec_tile_main(["encode", "--model", ckpt,
                     str(tmp_path / "tile.npy"), adir])
    a, b = str(tmp_path / "a.npy"), str(tmp_path / "b.npy")
    codec_tile_main(["decode", "--model", ckpt, adir, a])
    codec_tile_main(["decode", "--model", onnx_dir, "--backend", "onnx",
                     adir, b])
    ra, rb = np.load(a), np.load(b)
    assert (ra == rb).mean() > 0.999   # int8 quantization absorbs fp noise


def test_empty_windows_skipped(tiny_ckpt, tmp_path):
    """Fully-nodata windows are not encoded; decode rebuilds them as pure
    fill from the mask alone."""
    ckpt, _ = tiny_ckpt
    s = _windows()
    s[2] = -128                     # window 2 entirely nodata
    np.save(tmp_path / "in.npy", s)
    gvq, out = str(tmp_path / "e.gvq.npz"), str(tmp_path / "out.npy")
    codec_patch_main(["encode", "--model", ckpt,
                      str(tmp_path / "in.npy"), gvq])
    header, packets, valid = read_container(gvq)
    assert header["n_windows"] == 5 and header["n_total"] == 6
    assert not valid[2].any()
    codec_patch_main(["decode", "--model", ckpt, "--int8", gvq, out])
    r8 = np.load(out)
    assert r8.shape == (6, BANDS, W, W)
    assert (r8[2] == -128).all()            # rebuilt as pure fill
    assert (r8[1] != -128).all()            # neighbours decoded normally
    # float output: empty window is all-zero vectors
    codec_patch_main(["decode", "--model", ckpt, gvq, out])
    assert (np.load(out)[2] == 0).all()


def test_tile_empty_windows_skipped(tiny_ckpt, tmp_path):
    pytest.importorskip("rasterio")
    ckpt, _ = tiny_ckpt
    rng = np.random.default_rng(2)
    tile = rng.integers(-127, 128, (BANDS, 2 * W, 2 * W), dtype=np.int8)
    tile[:, :W, :W] = -128          # top-left window entirely nodata
    np.save(tmp_path / "tile.npy", tile)
    adir, out = str(tmp_path / "assets"), str(tmp_path / "rec.npy")
    codec_tile_main(["encode", "--model", ckpt,
                     str(tmp_path / "tile.npy"), adir])
    from georvq.ae.cog import read_assets
    _, packets, kept, _ = read_assets(adir)
    assert len(packets) == 3                # 4 grid cells, 1 skipped
    codec_tile_main(["decode", "--model", ckpt, adir, out])
    rec = np.load(out)
    assert (rec[:, :W, :W] == -128).all()
    assert (rec[:, W:, W:] != -128).all()


def test_tile_geotiff_exact_filetype(tiny_ckpt, tmp_path):
    """A bottom-up GeoTIFF in the AEF source.coop layout round-trips to the
    exact same file type: driver, dtype, blocking, compression, interleave,
    nodata tag, CRS, transform, and orientation."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine
    ckpt, _ = tiny_ckpt
    rng = np.random.default_rng(4)
    H = Wd = 2 * W
    tile = rng.integers(-127, 128, (BANDS, H, Wd), dtype=np.int8)
    tile[:, :5, :7] = -128
    src = str(tmp_path / "src.tiff")
    prof = {"driver": "GTiff", "height": H, "width": Wd, "count": BANDS,
            "dtype": "int8", "nodata": -128.0, "tiled": True,
            "blockxsize": 16, "blockysize": 16, "compress": "deflate",
            "interleave": "band", "crs": "EPSG:32755",
            "transform": Affine(10.0, 0.0, 254240.0, 0.0, 10.0, 5740160.0)}
    with rasterio.open(src, "w", **prof) as ds:
        ds.write(tile)                        # transform.e > 0: bottom-up
    gvq, out = str(tmp_path / "t.gvq.npz"), str(tmp_path / "rec.tiff")
    codec_tile_main(["encode", "--model", ckpt, src, gvq])
    codec_tile_main(["decode", "--model", ckpt, gvq, out])
    with rasterio.open(src) as s, rasterio.open(out) as r:
        for k in ("driver", "nodata", "blockxsize", "blockysize", "tiled",
                  "compress", "interleave", "count", "dtype",
                  "width", "height"):
            assert s.profile.get(k) == r.profile.get(k), k
        assert s.crs == r.crs and s.transform == r.transform
        rec = r.read()
    assert (rec[:, :5, :7] == -128).all()     # nodata exact, in FILE order
    assert ((rec[0] == -128) == (tile[0] == -128)).all()


def test_wrong_model_refused(tiny_ckpt, tiny_raw_cfg, tmp_path):
    """A blob encoded with model A must not silently decode with a
    same-architecture model B (different codebooks)."""
    ckpt, _ = tiny_ckpt
    cfg = Config.model_validate(tiny_raw_cfg)
    torch.manual_seed(99)
    other = build_model(cfg).eval()
    for rvq in other.rvqs:
        for lev in rvq.levels:
            torch.nn.init.normal_(lev.codebook.weight, std=1.0)
    other_path = str(tmp_path / "other.pt")
    save_checkpoint(other_path, other, cfg, step=0)
    np.save(tmp_path / "in.npy", _windows())
    gvq = str(tmp_path / "p.gvq.npz")
    codec_patch_main(["encode", "--model", ckpt,
                      str(tmp_path / "in.npy"), gvq])
    with pytest.raises(SystemExit, match="bitstream"):
        codec_patch_main(["decode", "--model", other_path, gvq,
                          str(tmp_path / "out.npy")])


def test_progressive_prefix_read(tiny_ckpt, tmp_path):
    """Depth-limited reads load only the ladder prefix and decode
    identically to truncating a full read."""
    from georvq.ae.cli import container_read_bytes
    ckpt, model = tiny_ckpt
    codec = Codec(model)
    s = _windows(8)
    x, valid = prep_ae_np(s)
    packets = codec.encode(torch.from_numpy(x))
    sig = [(sp.cells, sp.depth, sp.codes) for sp in model.specs]
    path = tmp_path / "p.gvq.npz"
    write_container(path, packets, sig, {"kind": "patch"}, valid=valid)
    # zip members exist per (tier, level), ladder-ordered
    import zipfile
    names = [i.filename for i in zipfile.ZipFile(path).infolist()]
    lvl = [n for n in names if n.startswith("t")]
    assert lvl == [f"t{t}l{j}" + ".npy" for t in range(2) for j in range(3)]
    # prefix read: only the needed members counted
    rd, tot = container_read_bytes(path, sig, [2, 0])
    assert rd < tot
    _, part, _ = read_container(path, depths=[2, 0])
    assert part[0].indices[0].shape == (1, 2)
    assert part[0].indices[1].shape == (16, 0)
    _, full, _ = read_container(path)
    for p, q in zip(part, full):
        assert np.array_equal(p.indices[0], q.indices[0][:, :2])
    # decoded output matches the full-read truncated decode
    a = codec.decode(part, [2, 0])
    b = codec.decode(full, [2, 0])
    assert torch.allclose(a, b, atol=1e-6)


def test_star_sidecar(tiny_ckpt, tmp_path):
    """8* coarse-residual: pack/unpack accuracy, quality lift over the
    no-sidecar decode, ignored at truncated coarse depth, off by flag."""
    from georvq.ae.cli import pack_star, unpack_star
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.3, (8, 1, 16)).astype(np.float32)
    packed, step = pack_star(r)
    assert packed.shape == (8, 4) and step.dtype == np.float16
    r2 = unpack_star(packed, step, 1, 16)
    err = np.abs(r2 - r)
    inside = np.abs(r) < 1.5 * step.astype(np.float32)[:, None, None]
    assert err[inside].max() <= 0.5 * step.astype(np.float32).max() + 1e-4

    ckpt, model = tiny_ckpt
    s = _windows()
    np.save(tmp_path / "in.npy", s)
    on, off = str(tmp_path / "on.gvq.npz"), str(tmp_path / "off.gvq.npz")
    codec_patch_main(["encode", "--model", ckpt, str(tmp_path / "in.npy"), on])
    codec_patch_main(["encode", "--model", ckpt, "--no-coarse-residual",
                      str(tmp_path / "in.npy"), off])
    import zipfile
    assert "t0star.npy" in zipfile.ZipFile(on).namelist()
    assert "t0star.npy" not in zipfile.ZipFile(off).namelist()
    # star is applied at full depth: on/off decodes differ (quality gain
    # is a trained-model property, verified on the release model)
    outs = {}
    for name, gvq in (("on", on), ("off", off)):
        dst = str(tmp_path / f"{name}.npy")
        codec_patch_main(["decode", "--model", ckpt, gvq, dst])
        outs[name] = np.load(dst)
    assert np.abs(outs["on"] - outs["off"]).max() > 1e-4
    # truncated coarse depth: star not applied, decodes equal on/off
    for name, gvq in (("t_on", on), ("t_off", off)):
        dst = str(tmp_path / f"{name}.npy")
        codec_patch_main(["decode", "--model", ckpt, "--tuple", "2,1",
                          gvq, dst])
    assert np.array_equal(np.load(tmp_path / "t_on.npy"),
                          np.load(tmp_path / "t_off.npy"))


def test_star_cross_backend(tiny_ckpt, onnx_dir, tmp_path):
    """ONNX encode carries the sidecar; both backends decode a starred
    container identically (same dequantized correction)."""
    ckpt, _ = tiny_ckpt
    from georvq.ae.cli import OnnxBackend, TorchBackend
    tb, ob = TorchBackend(ckpt), OnnxBackend(onnx_dir)
    x, _ = prep_ae_np(_windows())
    po = ob.encode(x, coarse_residual=True)
    assert po[0].coarse_residual is not None
    a = tb.decode(po)
    b = ob.decode(po)
    assert np.abs(a - b).max() < 1e-3


def test_quality_stats_and_control(tiny_ckpt, tmp_path, capsys):
    """Encode-time stats land in header + sidecar json; --quality picks the
    cheapest satisfying ladder point; unsatisfiable bounds fail clearly."""
    ckpt, _ = tiny_ckpt
    np.save(tmp_path / "in.npy", _windows())
    gvq, out = str(tmp_path / "q.gvq.npz"), str(tmp_path / "out.npy")
    codec_patch_main(["encode", "--model", ckpt, str(tmp_path / "in.npy"),
                      gvq])
    from georvq.ae.cli import _read_header
    stats = _read_header(gvq)["quality_stats"]
    assert os.path.exists(gvq + ".stats.json")
    assert set(stats) == {"3-1", "3-2", "3-3"}
    for v in stats.values():
        assert 0 < v["p50"] <= v["p90"] <= v["p99"] <= 2.0
        assert v["mean"] > 0 and v["max"] >= v["p99"]
    assert stats["3-3"]["star"] is True          # sidecar priced at full depth
    # monotone: deeper is better, costlier
    assert stats["3-1"]["mean"] >= stats["3-3"]["mean"]
    assert stats["3-1"]["bits"] < stats["3-3"]["bits"]
    # quality-control decode: loose bound -> cheapest point
    codec_patch_main(["decode", "--model", ckpt, "--quality", "mean<=2.0",
                      gvq, out])
    assert "-> 3-1" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="best available"):
        codec_patch_main(["decode", "--model", ckpt, "--quality",
                          "max<=1e-9", gvq, out])
    with pytest.raises(SystemExit, match="mutually exclusive"):
        codec_patch_main(["decode", "--model", ckpt, "--quality",
                          "mean<=1", "--tuple", "3,1", gvq, out])


def test_fused_stats_match_generic(tiny_ckpt):
    """The fused encode+stats pass produces the same packets and (near-)
    identical stats as encode-then-decode-per-point."""
    from georvq.ae.cli import (TorchBackend, QualityStats,
                                      collect_stats, quality_ladder)
    ckpt, _ = tiny_ckpt
    be = TorchBackend(ckpt)
    x, valid = prep_ae_np(_windows(8))
    ladder = quality_ladder(be.specs_sig)
    pk_f, dcos = be.encode_and_stats(x, valid, ladder, coarse_residual=True)
    pk_g = be.encode(x, coarse_residual=True)
    for a, b in zip(pk_f, pk_g):
        for u, v in zip(a.indices, b.indices):
            assert np.array_equal(u, v)
        assert np.allclose(a.coarse_residual, b.coarse_residual, atol=1e-6)
    # generic path must measure the 2-bit sidecar decode sees, like fused
    from georvq.ae.cli import pack_star, unpack_star
    res = np.stack([p.coarse_residual for p in pk_g])
    deq = unpack_star(*pack_star(res), res.shape[1], res.shape[2])
    for p_, r_ in zip(pk_g, deq):
        p_.coarse_residual = r_
    qs_f = QualityStats(ladder)
    for pt, (d, wm) in dcos.items():
        qs_f.add(pt, d, win_mean=wm)
    qs_g = collect_stats(be, x, valid, pk_g, 8)
    sf = qs_f.summary(be.specs_sig, be.star_cost_bits)
    sg = qs_g.summary(be.specs_sig, be.star_cost_bits)
    for k in sf:
        for m in ("mean", "p50", "p90", "p99", "max"):
            assert abs(sf[k][m] - sg[k][m]) < 2e-3, (k, m, sf[k][m], sg[k][m])
