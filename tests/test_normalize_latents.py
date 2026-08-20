"""`model.norm_latents`: stage-2 must quantize the same geometry
stage-1 trained on.

The bug this guards against was silent. A stage-2 run warm-started from a
stage-1 checkpoint encoded with `encode_levels` (raw) while its codebooks
had been fitted on `encode_stage1` (L2-normalized * sqrt(dim)) latents.
Directions matched exactly, so nothing looked wrong -- but the books were
2-4x too long for the vectors they scored against, and training still
improved, because the decoder simply learned to compensate for a crippled
quantizer.
"""
import torch

from georvq.config import load_config
from georvq.factory import build_model


def _cfg(on):
    return load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3']",
        f"model.norm_latents={'true' if on else 'false'}"])


def test_default_is_off_so_existing_checkpoints_are_unaffected():
    cfg = load_config("configs/smoke.yaml", [])
    assert cfg.model.norm_latents is False


def test_normalized_latents_have_sqrt_dim_norm():
    cfg = _cfg(True)
    m = build_model(cfg, "cpu").eval()
    x = torch.randn(2, cfg.model.bands, cfg.model.window, cfg.model.window)
    with torch.no_grad():
        zs = m.encode_levels(x)
    for z in zs:
        n = z.norm(dim=-1)
        assert torch.allclose(n, torch.full_like(n, z.shape[-1] ** 0.5),
                              atol=1e-3)


def test_off_leaves_raw_latents_alone():
    cfg = _cfg(False)
    m = build_model(cfg, "cpu").eval()
    x = torch.randn(2, cfg.model.bands, cfg.model.window, cfg.model.window)
    with torch.no_grad():
        zs = m.encode_levels(x)
    # raw latents are not on the sqrt(dim) sphere (this is what stage-2
    # was accidentally quantizing)
    assert not all(
        torch.allclose(z.norm(dim=-1),
                       torch.full_like(z.norm(dim=-1), z.shape[-1] ** 0.5),
                       atol=1e-2) for z in zs)


def test_encode_stage1_is_idempotent_under_the_flag():
    """encode_stage1 normalizes encode_levels' output, so with the flag on
    the two must agree -- otherwise a stage-1 checkpoint and a stage-2 run
    would still disagree about the latent scale."""
    cfg = _cfg(True)
    m = build_model(cfg, "cpu").eval()
    x = torch.randn(2, cfg.model.bands, cfg.model.window, cfg.model.window)
    with torch.no_grad():
        a, b = m.encode_levels(x), m.encode_stage1(x)
    assert all(torch.allclose(p, q, atol=1e-5) for p, q in zip(a, b))


def test_direction_is_preserved():
    """The flag must only change magnitude: if it rotated latents, books
    fitted on stage-1 output would be invalid for a different reason."""
    x = torch.randn(2, 4, 32, 32)
    torch.manual_seed(0)
    off = build_model(_cfg(False), "cpu").eval()
    torch.manual_seed(0)
    on = build_model(_cfg(True), "cpu").eval()
    x = torch.randn(2, off.bands, off.window, off.window)
    with torch.no_grad():
        for a, b in zip(off.encode_levels(x), on.encode_levels(x)):
            cos = torch.cosine_similarity(a, b, dim=-1)
            assert torch.allclose(cos, torch.ones_like(cos), atol=1e-4)


def test_quantizing_the_wrong_geometry_is_worse_than_sending_zeros():
    """The measurement that exposed the bug, in miniature: fit books on
    normalized latents, then apply them to raw ones. Residual energy above
    1.0 means the quantizer hurts."""
    torch.manual_seed(0)
    D, K, L = 32, 64, 4
    direction = torch.randn(4096, D)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    nrm = direction * (D ** 0.5)
    # same directions, ~3.7x shorter -- the w32hd tier-0 ratio (5.3/19.6)
    raw = direction * (D ** 0.5) / 3.7

    books, r = [], nrm.clone()
    for _ in range(L):                          # crude residual k-means
        C = r[torch.randperm(r.shape[0])[:K]].clone()
        for _ in range(5):
            a = torch.cdist(r, C).argmin(1)
            for k in range(K):
                if (a == k).any():
                    C[k] = r[a == k].mean(0)
        r = r - C[torch.cdist(r, C).argmin(1)]
        books.append(C)

    def resid(z):
        rr = z.clone()
        e0 = rr.pow(2).sum(1).mean()
        for C in books:
            rr = rr - C[torch.cdist(rr, C).argmin(1)]
        return float(rr.pow(2).sum(1).mean() / e0)

    assert resid(nrm) < 0.5, "books should work in the space they were fit"
    assert resid(raw) > 1.0, "wrong scale should be worse than zeros"
