"""The shared conv pyramid in the tokenizer.

Without it the coarsest tier reads from a CLS parameter that starts
identical for every window, so everything it carries has to arrive
through attention. With it, each tier is a strided conv over the tier
below, and the coarse token arrives with a convolutional read of the
image. The token sequence is unchanged either way, which is what lets
the trunk, the positional embedding and the per-tier readouts stay as
they are.
"""
import torch

from georvq.config import Config
from georvq.factory import build_model


def _cfg(pyramid):
    return Config.model_validate({
        "data": {"source": "ae", "store_dir": "/tmp/georvq-nonexistent"},
        "model": {
            "window": 16, "bands": 8, "ff_ratio": 2.0,
            "tiers": ["1:16:32:3", "4:8:32:3"],
            "min_depths": [1, 0],
            "tokenizer": {"dim": 32, "depth": 1, "stem_pyramid": pyramid},
            "decoder": {"dim": 32, "depth": 1, "heads": 4, "head_feat": 8},
            "vq": {},
        },
        "train": {"batch_windows": 4, "steps": 10, "amp": False,
                  "tf32": False},
        "log": {"run_tag": "test", "wandb": False},
    })


def test_both_paths_emit_the_same_latent_geometry():
    x = torch.randn(2, 8, 16, 16)
    shapes = []
    for p in (False, True):
        model = build_model(_cfg(p))
        zs = model.encode_levels(x)
        shapes.append([tuple(z.shape) for z in zs])
    assert shapes[0] == shapes[1] == [(2, 1, 16), (2, 16, 8)]


def test_pyramid_drops_the_cls_parameter():
    """The coarse token comes from a conv now, so a CLS parameter would
    be dead weight -- and its absence is what makes the two paths
    incompatible state dicts."""
    tok = build_model(_cfg(True)).tokenizer
    names = dict(tok.named_parameters())
    assert not any(n.startswith("cls") for n in names)
    assert any(n.startswith("stages.") for n in names)

    old = dict(build_model(_cfg(False)).tokenizer.named_parameters())
    assert any(n.startswith("cls") for n in old)
    assert any(n.startswith("patches.") for n in old)


def test_the_coarse_token_depends_on_the_input():
    """The point of the change: with a CLS parameter the coarse token is
    the same vector for every window before attention runs, and with the
    pyramid it is not."""
    model = build_model(_cfg(True))
    tok = model.tokenizer
    a = torch.randn(1, 8, 16, 16)
    b = torch.randn(1, 8, 16, 16)
    fa, fb = tok.stages[0](a), tok.stages[0](b)
    for st in tok.stages[1:]:
        fa, fb = st(fa), st(fb)
    assert not torch.allclose(fa, fb), \
        "coarse stage output is input-independent"


def test_default_is_off():
    tok = build_model(_cfg(False)).tokenizer
    assert tok.pyramid is False
