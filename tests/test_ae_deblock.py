"""The AE deblock decoder variant: recipe, locality contract, identity.

The canonical-recipe properties (tiled == whole to ~1e-5, stitch
residuals) were measured on trained weights against val data; what unit
tests can and do pin down exactly:

  - the port is identity at zero-init (head + FiLM zero),
  - the seam mask generalizes to rectangular clipped-halo frames,
  - the locality contract: a frame's core output depends ONLY on its
    support (core + halo) — bitwise, which is what makes the recipe
    scheduler-independent,
  - decoder_id folds post-processor weights + recipe into the identity.

An integration test against the release bundle's packaged deblocker
runs when the bundle is present (skipped elsewhere).
"""
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from georvq.ae.deblock import (RECIPE, AEDeblocker, boundary_mask,
                               deblock_mosaic, frame_support,
                               load_deblocker)

from .conftest import RELEASE_BUNDLE

DEBLOCK_PT = os.path.join(RELEASE_BUNDLE, "decoders", "deblock.pt")


def _unit(b, c, h, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(b, c, h, w, generator=g), dim=1,
                       eps=1e-6)


def _tiny_net(perturb=0.0, seed=0):
    torch.manual_seed(seed)
    net = AEDeblocker(bands=8, width=16, cond_dim=8, groups=4).eval()
    if perturb:
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * perturb)
    return net


def test_identity_at_init():
    """Zero-init head + zero-init FiLM -> the untrained deblocker is an
    exact pass-through (up to the final renorm of already-unit input)."""
    net = _tiny_net()
    x = _unit(2, 8, 128, 192)
    y = net(x, torch.tensor([[8, 8, 8], [8, 1, 0]]))
    assert torch.allclose(y, x, atol=1e-5)


def test_boundary_mask_rectangular():
    m = boundary_mask(128, 192, 64, torch.device("cpu"))[0, 0]
    assert m.shape == (128, 192)
    assert m[64].all() and m[:, 64].all() and m[:, 128].all()
    assert m[0, :64].sum() == 0 and m[1, 1] == 0
    # square case matches the research repo's formula
    sq = boundary_mask(128, 128, 64, torch.device("cpu"))[0, 0]
    idx = torch.arange(128)
    on = ((idx % 64 == 0) & (idx > 0)).float()
    assert torch.equal(sq, (on[None, :] + on[:, None]).clamp(max=1.0))


def test_locality_contract():
    """A frame's core output depends on its support windows and nothing
    else — BITWISE. This is the property that makes the canonical recipe
    scheduler-independent (per-frame, slab, or viewport decodes agree)."""
    net = _tiny_net(perturb=0.05)
    Ls = torch.tensor([[8, 8, 8]])
    x = _unit(1, 8, 256, 256)
    y = deblock_mosaic(net, x, Ls, frame=128, halo=64)
    # perturb strictly outside frame (0,0)'s support (its core is
    # [0:128), support [0:192) on both axes)
    x2 = x.clone()
    x2[:, :, 192:, :] = _unit(1, 8, 64, 256, seed=9)
    x2[:, :, :, 192:] = _unit(1, 8, 256, 64, seed=10)
    y2 = deblock_mosaic(net, x2, Ls, frame=128, halo=64)
    assert torch.equal(y[:, :, :128, :128], y2[:, :, :128, :128])
    # ... and the perturbation did change things elsewhere
    assert not torch.allclose(y[:, :, 128:, 128:], y2[:, :, 128:, 128:])


def test_tiling_partitions_exactly():
    """Hard crop covers every output pixel exactly once; output stays
    unit-norm (odd-sized trailing frames included)."""
    net = _tiny_net(perturb=0.05)
    x = _unit(1, 8, 320, 192)
    y = deblock_mosaic(net, x, torch.tensor([[8, 4, 2]]),
                       frame=128, halo=64)
    assert y.shape == x.shape
    assert torch.allclose(y.norm(dim=1), torch.ones(1, 320, 192),
                          atol=1e-4)


def test_frame_support_grid():
    sup = list(frame_support(24, 24, 64))       # 512/64 = 8-window frames
    assert len(sup) == 9
    (r, c, ((sr0, sr1), (sc0, sc1))) = sup[0]
    assert r == (0, 8) and c == (0, 8) and (sr0, sr1) == (0, 9)
    (r, c, ((sr0, sr1), (sc0, sc1))) = sup[-1]
    assert r == (16, 24) and (sr0, sr1) == (15, 24)


def test_decoder_id_variants(tiny_raw_cfg):
    from georvq.config import Config
    from georvq.factory import build_model
    from georvq.inference.release import decoder_id
    torch.manual_seed(0)
    model = build_model(Config.model_validate(tiny_raw_cfg)).eval()
    net = _tiny_net()
    base = decoder_id(model)
    with_post = decoder_id(model, extra_state=net.state_dict(),
                           recipe=RECIPE)
    other_recipe = decoder_id(model, extra_state=net.state_dict(),
                              recipe={**RECIPE, "frame": 256})
    assert len({base, with_post, other_recipe}) == 3
    # stable across calls
    assert with_post == decoder_id(model, extra_state=net.state_dict(),
                                   recipe=RECIPE)


@pytest.mark.skipif(not os.path.exists(DEBLOCK_PT),
                    reason="release bundle with packaged deblocker "
                           "not available")
def test_release_deblocker_tiled_vs_whole():
    """Trained weights: the canonical tiling must track whole-mosaic
    inference (GroupNorm makes them differ; the recipe sweep measured
    ~1e-5 mean dcos — gate at an order of slack)."""
    net, entry = load_deblocker(DEBLOCK_PT)
    assert entry["recipe"] == RECIPE
    x = _unit(1, 64, 256, 256, seed=3)
    Ls = torch.tensor([[8, 8, 8]])
    whole = net(x, Ls)
    tiled = deblock_mosaic(net, x, Ls, frame=128, halo=64)
    dev = 1.0 - (whole * tiled).sum(1)
    assert float(dev.mean()) < 1e-4 and float(dev.max()) < 1e-3


@pytest.mark.skipif(not os.path.exists(RELEASE_BUNDLE),
                    reason="release bundle not available")
def test_stats_rows_matches_numpy_reference():
    """The fused device stats (dcos + int-step secondary metric) must
    equal a straightforward numpy implementation."""
    from georvq.ae.cli import NODATA, TorchBackend, prep_ae_np

    be = TorchBackend(RELEASE_BUNDLE, "cpu")
    rng = np.random.default_rng(0)
    raw = rng.integers(-127, 128, (64, 128, 128)).astype(np.int8)
    raw[:, :10, :10] = NODATA                    # a nodata patch
    rows = F.normalize(torch.from_numpy(
        rng.standard_normal((64, 128, 128), dtype=np.float32)), dim=0)

    dvals, wm, hist, wint = be.stats_rows(rows.numpy(), raw)

    x, v = prep_ae_np(raw[None])
    x, v = x[0], v[0]
    y = rows.numpy()
    d = np.clip(1.0 - (x * y).sum(0), 1e-9, 2.0)
    np.testing.assert_allclose(np.sort(dvals), np.sort(d[v]), atol=1e-5)
    # requant via the same torch ops (numpy libm differs by ULPs at
    # exact rounding boundaries — the documented backend-consistent
    # slack); the test pins the histogram/quantile/window logic
    q = torch.clamp(torch.round(torch.sign(rows) * torch.sqrt(
        rows.abs()) * 127.5), -127, 127).to(torch.int16).numpy()
    dabs = np.abs(q - raw.astype(np.int16)).astype(np.float64)
    for fi, dint in enumerate((dabs.max(0), dabs.mean(0))):
        assert hist[fi].sum() == int(v.sum())
        np.testing.assert_array_equal(
            hist[fi], np.bincount(
                np.round(dint[v] * 256).astype(np.int64),
                minlength=be.INT_HIST_BINS))
        # per-window stats: mean + max on a fully-valid window, and the
        # nodata window's stats must ignore invalid pixels
        wd = dint[64:128, 64:128]
        assert wint[1, 1, fi, 0] == pytest.approx(wd.mean(), rel=1e-5)
        assert wint[1, 1, fi, 4] == pytest.approx(wd.max(), rel=1e-6)
        wd0 = dint[:64, :64][v[:64, :64]]
        assert wint[0, 0, fi, 4] == pytest.approx(wd0.max(), rel=1e-6)
        assert wint[0, 0, fi, 0] == pytest.approx(wd0.mean(), rel=1e-5)


ONNX_DIR = os.path.join(os.path.dirname(RELEASE_BUNDLE.rstrip("/")),
                        "onnx")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ONNX_DIR, "deblock.onnx")),
    reason="release onnx export with deblock.onnx not available")
def test_onnx_deblock_parity():
    """deblock.onnx must match the torch deblocker (fp32 both) on a
    frame size different from the export sample (dynamic H/W)."""
    pytest.importorskip("onnxruntime")
    from georvq.ae.cli import OnnxBackend, TorchBackend

    tb = TorchBackend(RELEASE_BUNDLE, "cpu")
    tb.load_deblock()
    ob = OnnxBackend(ONNX_DIR, "cpu")
    ob.load_deblock()
    assert ob.deblock_decoder_id == tb.deblock_decoder_id
    rng = np.random.default_rng(7)
    m = rng.standard_normal((64, 256, 128)).astype(np.float32)
    m /= np.linalg.norm(m, axis=0, keepdims=True)
    yt = tb.deblock_frame(m, [8, 8, 8])
    yo = ob.deblock_frame(m, [8, 8, 8])
    assert float((1.0 - (yt * yo).sum(0)).max()) < 1e-5
