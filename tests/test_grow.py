"""Growing a decoder must not change what it computes.

The experiment this supports only means anything if the grown model
STARTS as its ancestor: otherwise a quality change confounds "more
capacity, trained" with "weights partially thrown away".
"""
import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.train.grow import grow_decoder, verify_growth


def _cfg(depth, ff_ratio, head_feat):
    return Config.model_validate({
        "data": {"source": "ae", "store_dir": "/tmp/unused"},
        "model": {
            "window": 16, "bands": 8,
            "tiers": ["1:16:32:2", "4:16:32:2"],
            "ff_ratio": 2.0,               # tokenizer: never grows
            "tokenizer": {"dim": 32, "depth": 2},
            "decoder": {"dim": 32, "depth": depth, "heads": 4,
                        "head_feat": head_feat, "ff_ratio": ff_ratio},
            "vq": {"warmup_steps": 0},
        },
        "log": {"run_tag": "grow", "wandb": False},
    })


def _prepped(model, n=4):
    torch.manual_seed(1)
    raw = torch.randint(-127, 128, (n, 64, 16, 16), dtype=torch.int8)
    return model.source.prep(raw, "cpu")


def test_growth_preserves_the_function():
    """Deeper AND wider at once, which is the case that actually runs."""
    torch.manual_seed(0)
    src = build_model(_cfg(depth=3, ff_ratio=2.0, head_feat=8))
    dst = build_model(_cfg(depth=7, ff_ratio=3.0, head_feat=12))
    src.eval(); dst.eval()

    x, valid = _prepped(src)
    # run the source once so any lazy codebook init happens BEFORE the
    # state dict is taken; otherwise the two models quantize differently
    with torch.no_grad():
        src.encode_levels(x)
    grow_decoder(dst, src.state_dict(), verbose=False)

    r = verify_growth(src, dst, x, valid=valid, atol=1e-4)
    assert r["ok"], f"grown decoder diverged: {r}"
    assert r["rel"] < 1e-4


def test_growth_is_not_a_no_op():
    """Guard against the preservation test passing because nothing grew."""
    torch.manual_seed(0)
    src = build_model(_cfg(depth=3, ff_ratio=2.0, head_feat=8))
    dst = build_model(_cfg(depth=7, ff_ratio=3.0, head_feat=12))
    n_src = sum(p.numel() for p in src.decoder.parameters())
    n_dst = sum(p.numel() for p in dst.decoder.parameters())
    assert n_dst > 1.5 * n_src
    assert len(dst.decoder.blocks) == len(src.decoder.blocks) + 4

    rep = grow_decoder(dst, src.state_dict(), verbose=False)
    assert rep["expanded"], "nothing was expanded"
    assert rep["exact"] > 0


def test_growing_the_encoder_side_is_refused():
    """The bitstream must not move: this is a decoder-side experiment."""
    torch.manual_seed(0)
    src = build_model(_cfg(depth=3, ff_ratio=2.0, head_feat=8))
    wide = _cfg(depth=3, ff_ratio=2.0, head_feat=8)
    wide.model.tokenizer.dim = 48            # encoder side
    dst = build_model(wide)
    with pytest.raises(ValueError, match="frozen encoder"):
        grow_decoder(dst, src.state_dict(), verbose=False)
