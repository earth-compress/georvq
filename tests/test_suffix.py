"""Narrow neighborhood-cross-attention suffix (suffix_dim trunks)."""
import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.model.suffix import NeighborhoodCrossAttention


def _halo_allow(g, b, r, Nb):
    """(g*g, Nb + g*g) bool: base always allowed; fine keys allowed iff
    inside the query's block halo."""
    allow = torch.zeros(g * g, Nb + g * g, dtype=torch.bool)
    allow[:, :Nb] = True
    for qi in range(g * g):
        qr, qc = divmod(qi, g)
        bi, bj = qr // b, qc // b
        for ki in range(g * g):
            kr, kc = divmod(ki, g)
            if (bi * b - r <= kr < bi * b + b + r
                    and bj * b - r <= kc < bj * b + b + r):
                allow[qi, Nb + ki] = True
    return allow


def test_halo_attention_matches_masked_dense():
    """Blocked-halo attention == dense attention under the halo mask."""
    torch.manual_seed(0)
    B, Nb, g, r, D = 2, 5, 4, 1, 32
    att = NeighborhoodCrossAttention(D, base_dim=16, grid=g, radius=r,
                                     heads=4)
    fine = torch.randn(B, g * g, D)
    base = torch.randn(B, Nb, 16)
    out = att(fine, base)

    q = att.q(fine).view(B, g * g, 4, D // 4).permute(0, 2, 1, 3)
    kf, vf = att.kv_f(fine).chunk(2, -1)
    kb, vb = att.kv_b(base).chunk(2, -1)
    K = torch.cat([kb, kf], 1).view(B, -1, 4, D // 4).permute(0, 2, 1, 3)
    V = torch.cat([vb, vf], 1).view(B, -1, 4, D // 4).permute(0, 2, 1, 3)
    logits = q @ K.transpose(-2, -1) * (D // 4) ** -0.5
    allow = _halo_allow(g, att.b, r, Nb)
    logits = logits.masked_fill(~allow, float("-inf"))
    ref = (logits.softmax(-1) @ V).permute(0, 2, 1, 3).reshape(B, g * g, D)
    ref = att.out(ref)
    assert torch.allclose(out, ref, atol=1e-4)


def test_halo_block_shapes():
    att = NeighborhoodCrossAttention(64, base_dim=32, grid=16, radius=2,
                                     heads=8)
    assert att.b == 4 and att.nb == 4 and att.halo == 8
    out = att(torch.randn(2, 256, 64), torch.randn(2, 17, 32))
    assert out.shape == (2, 256, 64) and torch.isfinite(out).all()


def _eff_cfg():
    return load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3','16:8:32:3']",
        "model.vq=null", "model.sigma={max: 4.0, freqs: 4}",
        "model.local_attn=[null, null, 1]",
        "model.tokenizer={dim: 32, depth: 4, late_join: 2, suffix_dim: 16}",
        "model.decoder={dim: 32, depth: 4, heads: 4, head_feat: 8, "
        "late_join: 2, suffix_dim: 16}",
        "train.grad_ckpt=true"])


def test_suffix_model_forward_backward():
    cfg = _eff_cfg()
    m = build_model(cfg, "cpu")
    m.train()
    W = cfg.model.window
    x = torch.randn(2, cfg.model.bands, W, W)
    rec = m.forward_stage1(x, torch.rand(2, 3) * 4)
    assert rec.shape == x.shape and torch.isfinite(rec).all()
    rec.pow(2).mean().backward()
    assert m.tokenizer.suffix[0].attn.q.weight.grad is not None
    assert m.decoder.suffix[0].attn.q.weight.grad is not None
    # narrow stream widths
    assert m.tokenizer.patches[-1].out_channels == 16
    assert m.decoder.projs[-1].out_features == 16
