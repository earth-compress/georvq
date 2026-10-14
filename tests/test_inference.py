import numpy as np
import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.inference.api import Codec
from georvq.inference.stats_codec import StatsPacker, fit_stats_pca


def _codec(tiny_raw_cfg, **model_over):
    raw = {**tiny_raw_cfg, "model": {**tiny_raw_cfg["model"], **model_over}}
    cfg = Config.model_validate(raw)
    torch.manual_seed(0)
    model = build_model(cfg)
    model.eval()
    return Codec(model), cfg


def _x(B=4):
    torch.manual_seed(1)
    return torch.nn.functional.normalize(torch.randn(B, 8, 16, 16), dim=1)


def test_encode_decode_roundtrip(tiny_raw_cfg):
    """decode(encode(x)) must equal the model's own cumsum-based recon."""
    codec, _ = _codec(tiny_raw_cfg)
    x = _x()
    packets = codec.encode(x)
    assert len(packets) == 4
    assert packets[0].indices[0].shape == (1, 3)
    assert packets[0].indices[1].shape == (16, 3)
    with torch.no_grad():
        _, cums = codec.model.level_cumsums(x)
        for depths in ([3, 3], [3, 2], [1, 0]):
            ref = codec.model.eval_tuple(cums, list(depths), x)
            out = codec.decode(packets, list(depths))
            assert torch.allclose(ref, out, atol=1e-6), depths


def test_packet_bits(tiny_raw_cfg):
    codec, _ = _codec(tiny_raw_cfg)
    assert codec.packet_bits([1, 0]) == pytest.approx(np.log2(32))
    assert codec.packet_bits() == pytest.approx((3 + 16 * 3) * np.log2(32))


def test_cascade_roundtrip(tiny_raw_cfg):
    codec, _ = _codec(tiny_raw_cfg,
                      conditioning={"kind": "cascade", "location": None,
                                    "hidden": 32, "layers": 2})
    x = _x()
    packets = codec.encode(x)
    with torch.no_grad():
        _, cums = codec.model.level_cumsums(x)
        ref = codec.model.eval_tuple(cums, [3, 3], x)
        out = codec.decode(packets, [3, 3])
    assert torch.allclose(ref, out, atol=1e-5)


def test_stats_packer_roundtrip():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 26)) @ np.diag(np.linspace(3, 0.01, 26))
    pca = fit_stats_pca(base)
    packer = StatsPacker(data=pca, tolerance=0.05)
    for s in base[:50].astype(np.float32):
        b = packer.pack(s)
        r = packer.unpack(b)
        assert np.abs(r - s).max() < 0.05 or len(b) == 1 + 52
        assert len(b) <= 1 + 2 * 26
    # raw fallback is exact to f16
    raw = packer.pack_raw(base[0].astype(np.float32))
    assert np.allclose(packer.unpack(raw), base[0].astype(np.float16),
                       atol=1e-3)
