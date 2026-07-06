"""Numerical parity of the VQ primitives against the original research
implementation (runs only where that repo is present)."""
import sys

import pytest
import torch

from tests.conftest import RINGSPACE_DIR

pytestmark = pytest.mark.ringspace


def _pair(K=32, D=8, **kw):
    sys.path.insert(0, RINGSPACE_DIR)
    from ring_space.compression import VectorQuantizer as RefVQ
    from georvq.model.vq import VectorQuantizer
    torch.manual_seed(0)
    ref = RefVQ(num_codes=K, dim=D, **kw)
    torch.manual_seed(0)
    new = VectorQuantizer(num_codes=K, dim=D, **kw)
    new.load_state_dict(ref.state_dict())
    return ref, new


@pytest.mark.parametrize("kw", [{}, {"grad_codebook": True}])
def test_vq_forward_parity(kw):
    ref, new = _pair(**kw)
    z = torch.randn(64, 8)
    for model in (ref, new):
        model.train()
        torch.manual_seed(1)          # identical data-init draws
        model._out = model(z)
    (q1, i1, c1), (q2, i2, c2) = ref._out, new._out
    assert torch.equal(q1, q2) and torch.equal(i1, i2)
    assert torch.equal(c1, c2)
    assert torch.equal(ref.ema_count, new.ema_count)
    assert torch.equal(ref.ema_sum, new.ema_sum)
    assert torch.equal(ref.codebook.weight, new.codebook.weight)


def test_vq_eval_parity():
    ref, new = _pair()
    z = torch.randn(64, 8)
    ref.train(), new.train()
    torch.manual_seed(1); ref(z)
    torch.manual_seed(1); new(z)
    ref.eval(), new.eval()
    z2 = torch.randn(32, 8)
    (q1, i1, _), (q2, i2, _) = ref(z2), new(z2)
    assert torch.equal(q1, q2) and torch.equal(i1, i2)


def test_rvq_parity():
    sys.path.insert(0, RINGSPACE_DIR)
    from ring_space.compression import ResidualVQ as RefRVQ
    from georvq.model.vq import ResidualVQ
    torch.manual_seed(0)
    ref = RefRVQ(num_levels=3, num_codes=16, dim=8)
    torch.manual_seed(0)
    new = ResidualVQ(num_levels=3, num_codes=16, dim=8)
    new.load_state_dict(ref.state_dict())
    z = torch.randn(4, 9, 8)
    ref.train(), new.train()
    torch.manual_seed(1); per1, idx1, c1 = ref.all_levels(z)
    torch.manual_seed(1); per2, idx2, c2 = new.all_levels(z)
    assert torch.equal(per1, per2) and torch.equal(c1, c2)
    for a, b in zip(idx1, idx2):
        assert torch.equal(a, b)

    ref.eval(), new.eval()
    r1 = ref.levels_with_rate(z)
    r2 = new.levels_with_rate(z)
    for a, b in zip(r1, r2):
        for k in ("d", "q", "ctx"):
            assert torch.equal(a[k], b[k])


def test_kmeans_parity():
    sys.path.insert(0, RINGSPACE_DIR)
    from ring_space.ae_codec import kmeans_init_rvq as ref_kmeans
    from ring_space.compression import ResidualVQ as RefRVQ
    from georvq.model.vq import ResidualVQ, kmeans_init_rvq
    torch.manual_seed(0)
    ref = RefRVQ(num_levels=2, num_codes=16, dim=8)
    torch.manual_seed(0)
    new = ResidualVQ(num_levels=2, num_codes=16, dim=8)
    new.load_state_dict(ref.state_dict())
    z = torch.randn(400, 8)
    torch.manual_seed(2); ref_kmeans(ref, z, iters=3, expected_batch=64)
    torch.manual_seed(2); kmeans_init_rvq(new, z, iters=3, expected_batch=64)
    for lr, ln in zip(ref.levels, new.levels):
        assert torch.equal(lr.codebook.weight, ln.codebook.weight)
        assert torch.equal(lr.ema_count, ln.ema_count)
