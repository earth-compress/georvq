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
from georvq.inference.cli import (codec_patch_main, codec_tile_main,
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


def test_container_raw_roundtrip(tiny_ckpt, tmp_path):
    _, model = tiny_ckpt
    codec = Codec(model)
    x, _ = prep_ae_np(_windows())
    packets = codec.encode(torch.from_numpy(x))
    path = tmp_path / "c.gvq.npz"
    write_container(path, packets, [(sp.cells, sp.depth, sp.codes)
                                    for sp in model.specs],
                    {"kind": "patch", "window": W}, raw=True)
    header, back, _ = read_container(path)
    assert header["coded"] is False and header["n_windows"] == 6
    for p, q in zip(packets, back):
        for a, b in zip(p.indices, q.indices):
            assert np.array_equal(a, b)


def test_container_coded_roundtrip(tiny_ckpt, tmp_path):
    pytest.importorskip("compressai")
    _, model = tiny_ckpt
    codec = Codec(model)
    x, _ = prep_ae_np(_windows(12))
    packets = codec.encode(torch.from_numpy(x))
    path = tmp_path / "c.gvq.npz"
    header = write_container(path, packets,
                             [(sp.cells, sp.depth, sp.codes)
                              for sp in model.specs],
                             {"kind": "patch", "window": W})
    assert header["coded"] is True
    _, back, _ = read_container(path)
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
    write_container(path, packets, sig, {"kind": "patch"}, raw=True,
                    valid=valid)
    header, _, valid2 = read_container(path)
    assert header["mask_shape"] == list(valid.shape)
    assert np.array_equal(valid, valid2)
    # an all-valid mask is elided entirely
    write_container(path, packets, sig, {"kind": "patch"}, raw=True,
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
    from georvq.inference.cli import OnnxBackend, TorchBackend
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
    codec_patch_main(["encode", "--model", ckpt, "--raw",
                      str(tmp_path / "in.npy"), gvq])
    codec_patch_main(["decode", "--model", ckpt, gvq, out])
    recon = np.load(out)
    assert recon.shape == (6, BANDS, W, W) and recon.dtype == np.float32
    # matches the model's own full-depth reconstruction on valid pixels,
    # zero vectors on nodata pixels
    codec = Codec(model)
    x, valid = prep_ae_np(s)
    ref = codec.decode(codec.encode(torch.from_numpy(x))).numpy()
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
    """A non-multiple-of-W raster pads, encodes, decodes, and unpads back
    to shape, with nodata pixels restored exactly."""
    ckpt, _ = tiny_ckpt
    rng = np.random.default_rng(0)
    H, Wd = 3 * W - 5, 2 * W + 3
    tile = rng.integers(-127, 128, (BANDS, H, Wd), dtype=np.int8)
    tile[:, :6, :9] = -128
    np.save(tmp_path / "tile.npy", tile)
    gvq, out = str(tmp_path / "t.gvq.npz"), str(tmp_path / "rec.npy")
    codec_tile_main(["encode", "--model", ckpt, "--raw",
                     str(tmp_path / "tile.npy"), gvq])
    header, _, valid = read_container(gvq)
    assert header["grid"] == [3, 3] and header["shape"] == [BANDS, H, Wd]
    assert header["nodata_fill"] == -128 and valid.shape == (H, Wd)
    assert not valid[0, 0] and valid[10, 10]
    codec_tile_main(["decode", "--model", ckpt, gvq, out])
    rec = np.load(out)
    assert rec.shape == (BANDS, H, Wd) and rec.dtype == np.int8
    assert (rec[:, :6, :9] == -128).all()          # nodata restored
    assert (rec[0, 6:, 9:] != -128).all()
    codec_tile_main(["decode", "--model", ckpt, "--float32", gvq, out])
    assert np.load(out).dtype == np.float32


def test_tile_cli_onnx_backend_matches(tiny_ckpt, onnx_dir, tmp_path):
    """A file encoded by torch decodes identically under the onnx backend."""
    ckpt, _ = tiny_ckpt
    rng = np.random.default_rng(1)
    tile = rng.integers(-127, 128, (BANDS, 2 * W, 2 * W), dtype=np.int8)
    np.save(tmp_path / "tile.npy", tile)
    gvq = str(tmp_path / "t.gvq.npz")
    codec_tile_main(["encode", "--model", ckpt, "--raw",
                     str(tmp_path / "tile.npy"), gvq])
    a, b = str(tmp_path / "a.npy"), str(tmp_path / "b.npy")
    codec_tile_main(["decode", "--model", ckpt, gvq, a])
    codec_tile_main(["decode", "--model", onnx_dir, "--backend", "onnx",
                     gvq, b])
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
    codec_patch_main(["encode", "--model", ckpt, "--raw",
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
    ckpt, _ = tiny_ckpt
    rng = np.random.default_rng(2)
    tile = rng.integers(-127, 128, (BANDS, 2 * W, 2 * W), dtype=np.int8)
    tile[:, :W, :W] = -128          # top-left window entirely nodata
    np.save(tmp_path / "tile.npy", tile)
    gvq, out = str(tmp_path / "t.gvq.npz"), str(tmp_path / "rec.npy")
    codec_tile_main(["encode", "--model", ckpt, "--raw",
                     str(tmp_path / "tile.npy"), gvq])
    header, _, _ = read_container(gvq)
    assert header["n_windows"] == 3         # 4 grid cells, 1 skipped
    codec_tile_main(["decode", "--model", ckpt, gvq, out])
    rec = np.load(out)
    assert (rec[:, :W, :W] == -128).all()
    assert (rec[:, W:, W:] != -128).all()
