"""Quantize-aware (b-bit) codebook units. The default fp32 VQ path is
covered by the golden-checkpoint gate in test_compat.py and must be
unaffected by any of this."""
import numpy as np
import torch

from georvq.model.vq import ResidualVQ
from georvq.model.vq_quant import QuantizedBookVQ, QuantizedResidualVQ


def _on_grid(W, bits):
    """Every codeword must be scale * integer with |int| <= 2^(b-1)-1."""
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(1, keepdim=True) / qmax
    ints = W / s.clamp_min(1e-12)
    return torch.allclose(ints, ints.round(), atol=1e-4)


def test_books_stay_on_grid_through_training():
    torch.manual_seed(0)
    vq = QuantizedBookVQ(num_codes=32, dim=16, book_bits=8)
    vq.train()
    for i in range(8):
        vq.reset_pending = i % 3 == 2      # exercise dead resets too
        z_q, idx, commit = vq(torch.randn(256, 16))
        assert torch.isfinite(z_q).all() and torch.isfinite(commit)
        assert _on_grid(vq.codebook.weight.data, 8)
    # z_q values are gathered from the on-grid book
    vq.eval()
    z = torch.randn(64, 16)
    z_q, idx, _ = vq(z)
    assert torch.allclose(z_q, vq.codebook(idx), atol=1e-6)


def test_rvq_quantized_levels_and_state_dict_layout():
    q = QuantizedResidualVQ(num_levels=3, num_codes=16, dim=8, book_bits=4)
    r = ResidualVQ(num_levels=3, num_codes=16, dim=8)
    assert set(q.state_dict().keys()) == set(r.state_dict().keys())
    q.train()
    z = torch.randn(128, 8)
    z_q, idxs, commit = q(z)
    assert len(idxs) == 3 and torch.isfinite(commit)
    for lev in q.levels:
        assert lev.book_bits == 4
        assert _on_grid(lev.codebook.weight.data, 4)


def test_project_books_snaps_external_writes():
    q = QuantizedResidualVQ(num_levels=2, num_codes=8, dim=4, book_bits=8)
    for lev in q.levels:      # simulate the k-means init writing fp32
        lev.codebook.weight.data.copy_(torch.randn(8, 4))
    q.project_books()
    for lev in q.levels:
        assert _on_grid(lev.codebook.weight.data, 8)


def test_int8_roundtrip_error_bounded():
    """Projection error per codeword is <= s/2 per dim (grid spacing)."""
    torch.manual_seed(1)
    vq = QuantizedBookVQ(num_codes=64, dim=32, book_bits=8)
    W = torch.randn(64, 32)
    vq.codebook.weight.data.copy_(W)
    vq.project_book()
    s = W.abs().amax(1, keepdim=True) / 127
    assert ((vq.codebook.weight.data - W).abs() <= s / 2 + 1e-6).all()
