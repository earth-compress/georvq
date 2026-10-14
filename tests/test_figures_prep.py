"""Figure harvesters decode through the run's prep and validity mask,
with deterministic prep, and always restore the model's mode."""
import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.sources.prep import Prep
from georvq.train.figures import (error_hist_harvest, rd_err_harvest,
                                  recon_harvest)


class _HalfMasked(Prep):
    """A non-default prep: scale by 3, mark the left half invalid."""

    def forward(self, raw, meta, device):
        x, _ = self.source.prep(raw, device)
        x = 3 * x
        valid = torch.ones(x.shape[0], *x.shape[2:], dtype=torch.bool)
        valid[..., : x.shape[-1] // 2] = False
        self.saw_det = getattr(self.source, "_prep_deterministic", False)
        return x, valid, meta

    def inverse(self, y, meta):
        return y / 3


@pytest.fixture
def rig(tiny_raw_cfg):
    torch.manual_seed(0)
    cfg = Config.model_validate(tiny_raw_cfg)
    model = build_model(cfg, torch.device("cpu"))
    model.train()
    calls = []
    real = model.level_cumsums

    def spy(x, loc=None, valid=None):
        calls.append(valid)
        return real(x, loc=loc, valid=valid)
    model.level_cumsums = spy
    val = torch.randint(-100, 100, (8, 8, 16, 16), dtype=torch.int8)
    return model, model.source, val, _HalfMasked(model.source, cfg.model), \
        calls


def test_harvesters_use_the_run_prep_and_mask(rig):
    model, source, val, prep, calls = rig
    p = recon_harvest(model, source, val, n=4, tuples=[[3, 0]], prep=prep)
    assert prep.saw_det and model.training
    assert not getattr(source, "_prep_deterministic", False)
    want, _, _ = prep(val[torch.linspace(0, 7, 4).long()], {}, "cpu")
    assert torch.equal(p["rows"]["orig"], want)
    assert not p["valid"][..., :8].any() and p["valid"][..., 8:].all()

    h = error_hist_harvest(model, source, val, tuples=[[3, 0]], chunk=4,
                           prep=prep)
    assert h["absu"] is not None                  # inverted through prep
    r = rd_err_harvest(model, source, val, [[3, 0], [3, 3]], px=256,
                       chunk=4, prep=prep)
    # errors are taken over valid pixels only: 8 tokens x 8 bands x 16x8
    assert r["errs"]["3-0"].numel() == 8 * 8 * 16 * 8
    assert all(v is not None and not v[..., :8].any() for v in calls)


def test_harvest_restores_mode_on_error(rig):
    model, source, val, prep, _ = rig

    def boom(*a, **kw):
        raise RuntimeError("decode failed")
    model.level_cumsums = boom
    for fn in (lambda: recon_harvest(model, source, val, n=2, prep=prep),
               lambda: error_hist_harvest(model, source, val, prep=prep),
               lambda: rd_err_harvest(model, source, val, [[3, 0]], px=256,
                                      prep=prep)):
        with pytest.raises(RuntimeError, match="decode failed"):
            fn()
        assert model.training
        assert not getattr(source, "_prep_deterministic", False)
