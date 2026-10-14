"""One codebook update per training step.

The rate model's pass re-runs every VQ level on a sub-batch after the
main quantize pass. It must leave the books, the step counter and a
pending reset alone. Gradient books (`grad_codebook`) must still count
steps and run their dead-code resets, which used to sit inside the
EMA-only branch.
"""
import copy
import warnings

import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.model.vq import ResidualVQ, VectorQuantizer
from georvq.model.vq_quant import QuantizedResidualVQ


def _raw(**model_extra):
    r = {"data": {"source": "ae", "store_dir": "/tmp/x"},
         "model": {"window": 16, "bands": 8, "ff_ratio": 2.0,
                   "tiers": ["1:16:32:3", "4:8:32:3"],
                   "tokenizer": {"dim": 32, "depth": 1},
                   "decoder": {"dim": 32, "depth": 1, "heads": 4,
                               "head_feat": 8},
                   "vq": {}},
         "train": {"amp": False}, "log": {"wandb": False}}
    r["model"].update(model_extra)
    return r


def _state(rvqs):
    out = []
    for rvq in rvqs:
        for lev in rvq.levels:
            sd = {k: v.clone() for k, v in lev.state_dict().items()}
            if hasattr(lev, "ema_sse"):
                sd["ema_sse"] = lev.ema_sse.clone()
            out.append((sd, lev.reset_pending))
    return out


def _same(a, b):
    assert len(a) == len(b)
    for (sa, pa), (sb, pb) in zip(a, b):
        assert pa == pb
        for k in sa:
            assert torch.equal(sa[k], sb[k]), k


@pytest.mark.parametrize("grad_codebook", [False, True])
def test_rate_pass_leaves_books_and_counters_alone(grad_codebook):
    torch.manual_seed(0)
    vq = {"grad_codebook": grad_codebook, "reset_mode": "split"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # rate with EMA books warns
        m = build_model(Config.model_validate(_raw(
            vq=vq, rate={"model_dim": 16, "blocks": 1, "ctx_dim": 8})))
    m.train()
    x = torch.nn.functional.normalize(torch.randn(4, 8, 16, 16), dim=1)
    zs = m.encode_levels(x)
    m.quantize(zs, torch.tensor([[3, 3]] * 4))
    for rvq in m.rvqs:
        for lev in rvq.levels:
            assert int(lev.step) == 1
            lev.reset_pending = True       # must survive the rate pass
    before = _state(m.rvqs)
    m.train_rate_bits([z[:2].float() for z in zs])
    m.rate.level_bits([z.detach() for z in zs], m.rvqs)
    _same(before, _state(m.rvqs))


@pytest.mark.parametrize("cls", [ResidualVQ, QuantizedResidualVQ])
def test_update_false_matches_assignment_without_state_change(cls):
    torch.manual_seed(1)
    rvq = cls(num_levels=2, num_codes=16, dim=4, kmeans_init=True)
    rvq.train()
    z = torch.randn(64, 4)
    rvq.all_levels(z)                      # initialise and update once
    ref = copy.deepcopy(rvq)
    for lev in rvq.levels:
        lev.reset_pending = True
    before = _state([rvq])
    out = rvq.levels_with_rate(z[:16])
    _same(before, _state([rvq]))
    ref.eval()
    _, idx_eval = ref.all_levels(z[:16])[:2]
    for o, i in zip(out, idx_eval):
        assert torch.equal(o["idx"], i)


def _dead_book(cls_kw, grad_codebook, mode):
    torch.manual_seed(2)
    vq = VectorQuantizer(num_codes=8, dim=2, decay=0.5, kmeans_init=False,
                         reset_interval=0, dead_threshold=0.01,
                         grad_codebook=grad_codebook, reset_mode=mode,
                         **cls_kw)
    # two codes sit on the data, six far away and never chosen
    W = torch.full((8, 2), 50.0)
    W[0] = torch.tensor([0.0, 0.0])
    W[1] = torch.tensor([1.0, 1.0])
    vq.codebook.weight.data.copy_(W)
    vq.ema_count.fill_(0.1)
    vq.ema_sum.copy_(W * 0.1)
    return vq


@pytest.mark.parametrize("mode", ["random", "split"])
def test_grad_codebook_counts_steps_and_resets_dead_codes(mode):
    vq = _dead_book({}, True, mode)
    vq.train()
    z = torch.randn(256, 2) * 0.3 + 0.5
    for _ in range(8):                     # decay 0.5: unused counts fall below 0.01
        vq(z)
    assert int(vq.step) == 8
    assert (vq.ema_count[2:] < 0.01).all()
    assert vq.ema_usage() == pytest.approx(2 / 8)
    w_before = vq.codebook.weight.detach().clone()
    vq.reset_pending = True
    vq(z)
    assert not vq.reset_pending
    assert int(vq.step) == 9
    assert vq.last_reset["n_reset"] == 6
    assert (vq.codebook.weight[2:] - w_before[2:]).abs().max() > 1.0
    assert (vq.codebook.weight[2:].abs() < 10).all()


def test_grad_codebook_leaves_codewords_to_the_optimizer():
    vq = _dead_book({}, True, "random")
    vq.train()
    w0 = vq.codebook.weight.detach().clone()
    s0 = vq.ema_sum.clone()
    vq(torch.randn(64, 2))
    assert torch.equal(vq.codebook.weight.detach(), w0)
    assert torch.equal(vq.ema_sum, s0)


def test_grad_codebook_gated_path_counts_steps_and_resets():
    vq = _dead_book({}, True, "random")
    vq.train()
    z = (torch.randn(4, 16, 2) * 0.3 + 0.5)
    scores = torch.randn(4, 8)
    for _ in range(8):
        vq.forward_gated(z, scores, subset_k=8)
    assert int(vq.step) == 8
    vq.reset_pending = True
    vq.forward_gated(z, scores, subset_k=8)
    assert not vq.reset_pending and vq.last_reset["n_reset"] > 0


def test_grad_codebook_quantized_book_counts_steps_and_resets():
    rvq = QuantizedResidualVQ(num_levels=1, num_codes=8, dim=2, decay=0.5,
                              kmeans_init=False, reset_interval=0,
                              grad_codebook=True)
    lev = rvq.levels[0]
    W = torch.full((8, 2), 50.0)
    W[0] = torch.tensor([0.0, 0.0])
    W[1] = torch.tensor([1.0, 1.0])
    lev.codebook.weight.data.copy_(W)
    lev.ema_count.fill_(0.1)
    rvq.train()
    z = torch.randn(256, 2) * 0.3 + 0.5
    for _ in range(8):
        rvq.all_levels(z)
    assert int(lev.step) == 8
    lev.reset_pending = True
    rvq.all_levels(z)
    assert not lev.reset_pending and lev.last_reset["n_reset"] == 6
