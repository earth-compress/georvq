import sys

import pytest
import torch
import torch.nn as nn

from georvq.config import Config
from georvq.factory import build_model
from georvq.model.conditioning import CodebookSelector, TierPredictor, ancestor_map
from georvq.model.vq import ResidualVQ



def _specs3():
    from georvq.config import TierSpec
    return [TierSpec(grid=1, dim=8, codes=16, depth=2),
            TierSpec(grid=4, dim=8, codes=16, depth=2),
            TierSpec(grid=16, dim=8, codes=16, depth=2)]


def _rvqs(specs, seed=0):
    torch.manual_seed(seed)
    rvqs = nn.ModuleList([ResidualVQ(num_levels=sp.depth, num_codes=sp.codes,
                                     dim=sp.dim) for sp in specs])
    rvqs.eval()
    return rvqs


def _zs(specs, B=4, seed=1):
    torch.manual_seed(seed)
    return [torch.randn(B, sp.cells, sp.dim) for sp in specs]


def test_ancestor_map():
    m = ancestor_map(4, 2)
    assert m.tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]


def test_child_slot_map():
    from georvq.model.conditioning import child_slot_map
    m = child_slot_map(4, 2)
    # each 2x2 block of children enumerates slots 0..3 in raster order
    assert m.tolist() == [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3]


def test_child_pos_distinguishes_siblings():
    """Without child_pos all children of one parent get identical inputs,
    so their predictions are identical; with it they differ."""
    specs = _specs3()
    zs = _zs(specs)
    B = zs[0].shape[0]
    recons = [zs[0]]
    depths = [torch.full((B,), specs[0].depth, dtype=torch.long)]
    for flag in (False, True):
        torch.manual_seed(3)
        tp = TierPredictor(specs, loc_dim=0, child_pos=flag)
        for p in tp.predictors:      # defeat the zero-init on the last layer
            nn.init.normal_(p[-1].weight, std=0.05)
        pred = tp._predict(1, recons, depths, None, B)
        same = torch.allclose(pred[:, 0], pred[:, 1], atol=1e-7)
        assert same != flag


def test_cascade_cumulative_vs_masked():
    """Full-depth masked recon must equal the cumulative recon's last slice."""
    specs = _specs3()
    rvqs = _rvqs(specs)
    torch.manual_seed(2)
    tp = TierPredictor(specs, loc_dim=6, coarse_prior=nn.Linear(6, 8))
    tp.eval()
    zs = _zs(specs)
    loc = torch.randn(4, 6)
    full = torch.full((4, 3), 2, dtype=torch.long)
    with torch.no_grad():
        masked, _ = tp.quantize(rvqs, specs, zs, loc, full)
        cums = tp.cumsums(rvqs, specs, zs, loc)
    for m, c, sp in zip(masked, cums, specs):
        assert torch.allclose(m, c[:, :, sp.depth], atol=1e-6)


def test_cascade_zero_init_matches_plain(tiny_raw_cfg):
    """With no prior and zero-init predictors, the cascade must reduce to
    plain per-tier RVQ."""
    specs = _specs3()
    rvqs = _rvqs(specs)
    tp = TierPredictor(specs, loc_dim=0, coarse_prior=None)
    tp.eval()
    zs = _zs(specs)
    Ls = torch.tensor([[2, 1, 0], [1, 1, 1], [2, 2, 2], [2, 0, 0]])
    with torch.no_grad():
        casc, _ = tp.quantize(rvqs, specs, zs, None, Ls)
        from georvq.model.layers import thermometer
        for i, (z, rvq, sp) in enumerate(zip(zs, rvqs, specs)):
            per, _, _ = rvq.all_levels(z)
            m = thermometer(Ls[:, i], sp.depth)[:, None, :, None]
            assert torch.allclose(casc[i], (per * m).sum(-2), atol=1e-6)


def test_selector_full_subset_recall(tiny_raw_cfg):
    """With subset_frac=1 the selector can never exclude the best code."""
    raw = {**tiny_raw_cfg,
           "model": {**tiny_raw_cfg["model"],
                     "conditioning": {"kind": "select", "proj_dim": 8,
                                      "subset_frac": 1.0}}}
    cfg = Config.model_validate(raw)
    torch.manual_seed(0)
    model = build_model(cfg)
    model.eval()
    x = torch.nn.functional.normalize(torch.randn(6, 8, 16, 16), dim=1)
    valid = torch.ones(6, 16, 16, dtype=torch.bool)
    loc = torch.randn(6, 256)
    d = model.selector.diagnostics(model, x, loc, valid)
    for i in range(2):
        assert d[f"sel_recall_p{i}_mean"] == pytest.approx(1.0)
        assert d[f"sel_div_p{i}"] == pytest.approx(1.0)
    assert "sel_gated_cos" in d and "sel_ungated_cos" in d


def test_select_quantize_and_cumsums(tiny_raw_cfg):
    raw = {**tiny_raw_cfg,
           "model": {**tiny_raw_cfg["model"],
                     "conditioning": {"kind": "select", "proj_dim": 8,
                                      "subset_frac": 0.5}}}
    cfg = Config.model_validate(raw)
    torch.manual_seed(0)
    model = build_model(cfg)
    model.eval()
    x = torch.nn.functional.normalize(torch.randn(4, 8, 16, 16), dim=1)
    loc = torch.randn(4, 256)
    Ls = torch.tensor([[3, 2]]).expand(4, -1)
    zs = model.encode_levels(x)
    zqs, commit, select = model.quantize(zs, Ls, loc=loc)
    assert torch.isfinite(select) and select > 0
    with torch.no_grad():
        _, cums = model.level_cumsums(x, loc=loc)
        out = model.eval_tuple(cums, [3, 2], x)
    assert out.shape == x.shape
