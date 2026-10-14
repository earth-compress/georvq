"""The external entry point must be the same door, not a second one.

`georvq.tessera.io` exists so the Tessera team never has to reproduce
the chip store's internal 130-band packing. That is only safe if it
produces bit-identical tensors to the store path -- otherwise the codec
sees a slightly different field than it was trained and measured on.
"""
import numpy as np
import pytest
import torch

from georvq.sources.tessera import TesseraSource
from georvq.tessera import io


def _fake(bands=128, h=64, w=64, seed=0, n=2):
    g = np.random.default_rng(seed)
    q = g.integers(-127, 127, (n, bands, h, w), dtype=np.int8)
    scale = np.full((n, h, w), 0.065, np.float16)
    q[0, :, 0, 0] = 0                      # nodata: all-zero vector
    if n > 1:
        scale[1, 5, 5] = 0.0               # nodata: zero scale
        scale[1, 6, 6] = 9.0               # nodata: absurd scale
    return q, scale


def _store_layout(q, scale):
    """What ChipStore hands the loader: 128 bands + the f16 scale plane
    reinterpreted as two trailing int8 bands."""
    sb = scale.view(np.int8).reshape(scale.shape[0], scale.shape[1],
                                     scale.shape[2], 2).transpose(0, 3, 1, 2)
    return np.concatenate([q, sb], axis=1)


class _Cfg:
    bands = 128


def test_io_prep_matches_the_store_path_exactly():
    q, scale = _fake()
    src = TesseraSource(_Cfg())
    x_ref, v_ref = src.prep(torch.from_numpy(_store_layout(q, scale)), "cpu")
    x_new, v_new = io.prep(q, scale, device="cpu")
    assert torch.equal(v_new, v_ref)
    assert torch.equal(x_new, x_ref)


def test_io_prep_reproduces_every_nodata_convention():
    q, scale = _fake()
    _, valid = io.prep(q, scale, device="cpu")
    assert not valid[0, 0, 0]      # all-zero vector
    assert not valid[1, 5, 5]      # zero scale
    assert not valid[1, 6, 6]      # absurd scale
    assert valid[0, 30, 30]        # ordinary pixel survives


def test_int8_without_a_scale_is_an_error_not_a_guess():
    q, _ = _fake()
    with pytest.raises(ValueError, match="scale plane"):
        io.prep(q)


def test_float_input_takes_the_dequantized_path():
    x = np.random.default_rng(1).standard_normal((128, 64, 64)).astype(np.float32)
    got, valid = io.prep(x)
    assert torch.allclose(got[0], torch.from_numpy(x) / TesseraSource.XNORM)
    with pytest.raises(ValueError, match="already dequantized"):
        io.prep(x, scale=np.ones((64, 64), np.float32))


def test_mismatched_scale_plane_is_rejected():
    q, _ = _fake()
    with pytest.raises(ValueError, match="does not match"):
        io.prep(q, np.ones((2, 32, 32), np.float32))


def test_windowing_round_trips_and_crops_rather_than_pads():
    x = torch.randn(1, 4, 150, 200)
    valid = torch.ones(1, 150, 200, dtype=torch.bool)
    xw, vw, grid = io.to_windows(x, valid, window=64)
    assert grid == (1, 2, 3) and xw.shape == (6, 4, 64, 64)
    assert vw.shape == (6, 64, 64)
    back = io.from_windows(xw, grid, window=64)
    assert torch.equal(back, x[:, :, :128, :192])     # cropped, never padded


def test_a_tile_smaller_than_one_window_is_an_error():
    with pytest.raises(ValueError, match="smaller than one"):
        io.to_windows(torch.randn(1, 4, 32, 32),
                      torch.ones(1, 32, 32, dtype=torch.bool), window=64)


def test_requantize_returns_to_source_units():
    q, scale = _fake(n=1)
    x, _ = io.prep(q, scale, device="cpu")
    orig = io.requantize(x)
    assert torch.allclose(orig, x * TesseraSource.XNORM)
    _, codes = io.requantize(x, scale[0].astype(np.float32))
    assert codes.dtype == torch.int8
    ok = torch.from_numpy(q[0]).ne(0).any(0)          # ignore nodata pixels
    assert torch.equal(codes[0][:, ok], torch.from_numpy(q[0])[:, ok])


def test_prep_accepts_the_layout_geotessera_actually_returns():
    """geotessera hands out (H, W, bands); the store is (bands, H, W).
    Both must land on identical tensors, or the demo path and the
    training path are quietly different codecs."""
    q, scale = _fake(n=1)
    chw = q[0]                                  # (bands, H, W)
    hwc = np.ascontiguousarray(chw.transpose(1, 2, 0))   # (H, W, bands)
    x_a, v_a = io.prep(chw, scale[0], device="cpu")
    x_b, v_b = io.prep(hwc, scale[0], device="cpu")
    assert torch.equal(x_a, x_b) and torch.equal(v_a, v_b)


def test_prep_refuses_to_guess_an_ambiguous_layout():
    """A tile whose width equals the band count has no detectable
    layout, so guessing would silently transpose someone's data."""
    g = np.random.default_rng(0)
    amb = g.integers(-127, 127, (128, 64, 128), dtype=np.int8)
    with pytest.raises(ValueError, match="cannot tell the layout"):
        io.prep(amb, np.ones((64, 128), np.float32))
    io.prep(amb, np.ones((64, 128), np.float32), channels_last=False)


@pytest.fixture(scope="module")
def tiny_release(tmp_path_factory):
    from tests.test_tessera_release import _tiny_bundle
    from georvq.tessera import release as rel
    torch.manual_seed(0)
    return rel.load(str(_tiny_bundle(tmp_path_factory.mktemp("tess"))))


def _field(c=8, h=32, w=96, seed=0):
    g = np.random.default_rng(seed)
    x = g.normal(size=(c, h, w)).astype(np.float32) * 0.1
    x[:, :, 64:] = 0                       # nodata: all-zero vectors
    return x


def test_decode_tile_restores_nodata(tiny_release):
    R = tiny_release
    x = _field()
    packets, grid = io.encode_tile(R, x)
    assert tuple(grid) == (1, 32 // R.window, 96 // R.window)
    out = io.decode_tile(R, packets, grid)
    assert (out[0, :, :, 64:] == 0).all()
    assert (out[0, :, :, :64].abs().sum(0) > 0).all()
    # a plain tuple grid carries no validity: nodata decodes as the mean
    plain = io.decode_tile(R, packets, tuple(grid))
    assert plain[0, :, :, 64:].abs().max() > 0
    assert torch.equal(plain[0, :, :, :64], out[0, :, :, :64])
    assert torch.equal(io.decode_tile(R, packets, tuple(grid), valid=grid.valid),
                       out)


def test_encode_tile_takes_channels_last(tiny_release):
    R = tiny_release
    x = _field(w=128)                      # (8, 32, 128): W equals 128 bands
    pk, grid = io.encode_tile(R, x, channels_last=False)
    pk_t, grid_t = io.encode_tile(R, np.ascontiguousarray(x.transpose(1, 2, 0)),
                                  channels_last=True)
    assert tuple(grid) == tuple(grid_t)
    assert torch.equal(io.decode_tile(R, pk, grid), io.decode_tile(R, pk_t, grid_t))
