"""Gain-shape VQ: sphere invariants, projection gain, RVQ behaviour."""
import math

import pytest
import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.model.vq import ResidualVQ
from georvq.model.vq_gainshape import GainShapeVQ, GainShapeResidualVQ


def _train_a_bit(vq, n=6, N=512, D=32, seed=0):
    torch.manual_seed(seed)
    vq.train()
    for _ in range(n):
        z = torch.randn(N, D) * torch.rand(N, 1).add(0.5)  # varied gains
        out = vq(z)
    return out


def test_book_stays_on_unit_sphere():
    vq = GainShapeVQ(num_codes=64, dim=32, gain_bits=4, reset_interval=3)
    _train_a_bit(vq)
    norms = vq.codebook.weight.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_assignment_is_max_cosine():
    vq = GainShapeVQ(num_codes=48, dim=16, gain_bits=4)
    _train_a_bit(vq, D=16)
    vq.eval()
    z = torch.randn(128, 16)
    _, idx, _ = vq(z)
    E = vq.codebook.weight.data
    cos = torch.cosine_similarity(z[:, None, :], E[None, :, :], dim=-1)
    assert torch.equal(idx, cos.argmax(-1))


def test_reconstruction_uses_projection_gain():
    """z_q = g_hat * u with g_hat ~ z.u (not ||z||), so the error is
    near-orthogonal to the chosen shape."""
    vq = GainShapeVQ(num_codes=64, dim=32, gain_bits=6)
    _train_a_bit(vq, n=8)
    vq.eval()
    z = torch.randn(256, 32)
    z_q, idx, _ = vq(z)
    u = vq.codebook.weight.data[idx]
    g_star = (z * u).sum(-1)
    g_hat = (z_q * u).sum(-1)
    # gain quantization is the only discrepancy (6 bits, log domain)
    assert (g_hat - g_star).abs().mean() < 0.06 * g_star.abs().mean()
    err = z - z_q
    resid_along_shape = (err * u).sum(-1).abs().mean()
    assert resid_along_shape < 0.06 * z.norm(dim=1).mean()


def test_projection_gain_beats_full_norm_gain():
    """The design claim: coding g* = z.u is materially better than coding
    ||z|| for the same shape (the 2/(1+cos) factor)."""
    vq = GainShapeVQ(num_codes=64, dim=32, gain_bits=8)
    _train_a_bit(vq, n=8)
    vq.eval()
    z = torch.randn(512, 32)
    z_q, idx, _ = vq(z)
    u = vq.codebook.weight.data[idx]
    err_proj = (z - z_q).pow(2).sum(-1).mean()
    err_norm = (z - z.norm(dim=1, keepdim=True) * u).pow(2).sum(-1).mean()
    assert err_proj < err_norm


def test_gain_book_is_finite_and_ordered_in_log_domain():
    vq = GainShapeVQ(num_codes=32, dim=16, gain_bits=5)
    _train_a_bit(vq, D=16, n=8)
    gb = vq.gain_book
    assert torch.isfinite(gb).all()
    assert gb.exp().min() > 0
    # quantile seeding keeps levels sorted; EMA-Lloyd should not scramble
    assert (gb[1:] >= gb[:-1] - 1e-3).all()


def test_state_dict_superset_of_parent():
    g = GainShapeResidualVQ(num_levels=3, num_codes=16, dim=8, gain_bits=4)
    r = ResidualVQ(num_levels=3, num_codes=16, dim=8)
    gk, rk = set(g.state_dict()), set(r.state_dict())
    assert rk < gk
    extra = {k.split(".")[-1] for k in gk - rk}
    assert extra == {"gain_book", "gain_count", "gain_init"}


def test_rvq_levels_reduce_residual_energy():
    torch.manual_seed(0)
    rvq = GainShapeResidualVQ(num_levels=4, num_codes=64, dim=32, gain_bits=5)
    rvq.train()
    for _ in range(10):
        z = torch.randn(512, 32)
        rvq(z)
    rvq.eval()
    z = torch.randn(512, 32)
    per, idxs, _ = rvq.all_levels(z)
    resid = z.clone()
    energies = [float(resid.pow(2).sum(-1).mean())]
    for l in range(per.shape[-2]):
        resid = resid - per[..., l, :]
        energies.append(float(resid.pow(2).sum(-1).mean()))
    assert all(b < a for a, b in zip(energies, energies[1:])), energies
    assert len(idxs) == 4


def test_bits_per_code_accounts_for_gain():
    vq = GainShapeVQ(num_codes=16384, dim=128, gain_bits=5)
    assert vq.bits_per_code() == pytest.approx(math.log2(16384) + 5)


def test_spherize_books_seeds_from_data():
    torch.manual_seed(0)
    rvq = GainShapeResidualVQ(num_levels=2, num_codes=32, dim=16, gain_bits=4)
    for lev in rvq.levels:            # simulate unconstrained k-means output
        lev.codebook.weight.data.copy_(torch.randn(32, 16) * 3)
        lev.initialized.fill_(True)
    rvq.spherize_books(torch.randn(4096, 16))
    for lev in rvq.levels:
        n = lev.codebook.weight.data.norm(dim=1)
        assert torch.allclose(n, torch.ones_like(n), atol=1e-4)
        assert bool(lev.gain_init) and torch.isfinite(lev.gain_book).all()


def test_model_end_to_end_gain_shape():
    cfg = load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3']",
        "model.vq={gain_shape: true, gain_bits: 4, warmup_steps: 0, "
        "dead_threshold: 0.01}"])
    m = build_model(cfg, "cpu")
    from georvq.model.vq_gainshape import GainShapeVQ as G
    assert all(isinstance(l, G) for rvq in m.rvqs for l in rvq.levels)
    m.train()
    W, B = cfg.model.window, 2
    x = torch.randn(B, cfg.model.bands, W, W)
    Ls = torch.tensor([[3, 3]] * B)
    out, commit, _ = m.forward_train(x, Ls)
    assert out.shape == x.shape and torch.isfinite(commit)
    (out.pow(2).mean() + commit).backward()
    assert m.tokenizer.patches[0].weight.grad is not None
    for rvq in m.rvqs:                # books stayed on the sphere
        for l in rvq.levels:
            n = l.codebook.weight.data.norm(dim=1)
            assert torch.allclose(n, torch.ones_like(n), atol=1e-4)


def test_eval_is_deterministic():
    cfg = load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3']",
        "model.vq={gain_shape: true, gain_bits: 4, warmup_steps: 0}"])
    m = build_model(cfg, "cpu")
    m.train()
    x = torch.randn(2, cfg.model.bands, cfg.model.window, cfg.model.window)
    m.forward_train(x, torch.tensor([[3, 3]] * 2))
    m.eval()
    with torch.no_grad():
        a = m.encode_levels(x)
        q1, i1, _ = m.rvqs[0](a[0])
        q2, i2, _ = m.rvqs[0](a[0])
    assert torch.equal(i1[0], i2[0]) and torch.allclose(q1, q2)
