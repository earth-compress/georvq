"""Periodic offline refit phases (georvq/train/refit.py)."""
import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.model.vq import _chunked_assign
from georvq.train.refit import refit_level, refit_phase


def _clustered(n=20000, d=8, k=4, seed=0):
    torch.manual_seed(seed)
    return torch.cat([torch.randn(n // k, d) + torch.randn(1, d) * 6
                      for _ in range(k)])


def _crippled_book(X, K=64, n_dup=32):
    """A book with many near-duplicate (effectively dead) entries."""
    C = X[torch.randperm(X.shape[0])[:K]].clone()
    C[K - n_dup:] = C[0] + 1e-3 * torch.randn(n_dup, X.shape[1])
    return C


def test_refit_reduces_residual_and_revives_dead_codes():
    X = _clustered()
    C = _crippled_book(X)
    Cn, _, st = refit_level(X, C)
    assert st["resid_after"] < st["resid_before"]
    assert st["perp_after"] > st["perp_before"]
    assert st["n_zero_after"] <= st["n_zero_before"]


def test_recycles_at_least_the_floor_when_nothing_is_dead():
    """A fully-healthy book must still churn recycle_frac of K, so
    high-distortion splitting keeps happening over a long run."""
    X = _clustered()
    K = 64
    C = X[torch.randperm(X.shape[0])[:K]].clone()
    for _ in range(6):                       # settle it first
        a = _chunked_assign(X, C)
        for j in range(K):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    _, _, st = refit_level(X, C, recycle_frac=0.05)
    assert st["n_recycled"] >= int(0.05 * K)


def test_split_pair_straddles_the_donor_centroid():
    """LBG geometry: the pair sits either side of the donor along its
    principal direction, NOT on the donor's worst-fitting outlier."""
    torch.manual_seed(0)
    X = torch.randn(4000, 4) * torch.tensor([6.0, 1.0, 1.0, 1.0])
    C = torch.zeros(8, 4)
    C[1:] = torch.randn(7, 4) * 0.01 + 40.0   # 7 far-away dead codes
    Cn, _, st = refit_level(X, C, lloyd_iters=0, recycle_frac=0.1)
    used = Cn[(Cn - 40).norm(dim=1) > 1.0]    # the ones that moved back
    assert used.shape[0] >= 2
    # they should straddle the data's principal axis, not sit on one tail
    proj = used[:, 0]
    assert proj.min() < 0 < proj.max(), proj


def test_zero_lloyd_iters_still_valid():
    X = _clustered()
    C = _crippled_book(X)
    Cn, _, st = refit_level(X, C, lloyd_iters=0)
    assert torch.isfinite(Cn).all()
    assert st["n_split"] > 0


def test_phase_updates_every_level_and_reports_cascade():
    cfg = load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3']",
        "model.vq={warmup_steps: 0, freeze_books: true}"])
    m = build_model(cfg, "cpu")
    m.train()
    x = torch.randn(8, cfg.model.bands, cfg.model.window, cfg.model.window)
    m.forward_train(x, torch.tensor([[3, 3]] * 8))
    before = [l.codebook.weight.data.clone()
              for rvq in m.rvqs for l in rvq.levels]
    with torch.no_grad():
        samples = [z.reshape(-1, z.shape[-1]).float().repeat(40, 1)
                   for z in m.encode_levels(x)]
    st = refit_phase(m, samples, cfg, lloyd_iters=1)
    after = [l.codebook.weight.data for rvq in m.rvqs for l in rvq.levels]
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), \
        "refit left every book untouched"
    for i, sp in enumerate(cfg.model.tiers):
        for L in range(1, sp.depth + 1):
            for k in ("resid_before", "resid_after", "perp_before",
                      "perp_after", "input_energy_before",
                      "input_energy_after", "n_recycled"):
                assert f"refit/pyr{i}_L{L}_{k}" in st, (i, L, k)


def test_cascade_ordering_is_visible():
    """Level l+1's input must reflect level l's NEW book: the after-chain
    input energies should differ from the before-chain ones past level 1."""
    cfg = load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:4','4:8:32:4']",
        "model.vq={warmup_steps: 0, freeze_books: true}"])
    m = build_model(cfg, "cpu")
    m.train()
    x = torch.randn(8, cfg.model.bands, cfg.model.window, cfg.model.window)
    m.forward_train(x, torch.tensor([[4, 4]] * 8))
    with torch.no_grad():
        samples = [z.reshape(-1, z.shape[-1]).float().repeat(60, 1)
                   for z in m.encode_levels(x)]
    st = refit_phase(m, samples, cfg, lloyd_iters=1)
    diffs = [abs(st[f"refit/pyr0_L{L}_input_energy_after"]
                 - st[f"refit/pyr0_L{L}_input_energy_before"])
             for L in range(2, 5)]
    assert max(diffs) > 0, "cascade effect not captured"


def test_config_defaults_off():
    cfg = load_config("configs/smoke.yaml", [])
    assert cfg.model.vq.refit_interval == 0
    assert cfg.model.vq.refit_offset == 1
