"""Narrow neighborhood-cross-attention suffix (suffix_dim trunks)."""
import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.model.suffix import NeighborhoodCrossAttention, neighbor_index


def test_neighbor_index_geometry():
    idx = neighbor_index(grid=4, radius=1)
    assert idx.shape == (16, 9)
    # corner cell (0,0): only 4 valid neighbors (itself + right/down/diag)
    assert (idx[0] >= 0).sum() == 4
    # interior cell (1,1) = flat 5: all 9 valid, centered on itself
    assert (idx[5] >= 0).all() and idx[5][4] == 5


def test_neighborhood_attention_matches_masked_dense():
    """Gathered attention == dense attention under the equivalent mask."""
    torch.manual_seed(0)
    B, Nb, g, r, D = 2, 5, 4, 1, 32
    att = NeighborhoodCrossAttention(D, base_dim=16, grid=g, radius=r,
                                     heads=4)
    fine = torch.randn(B, g * g, D)
    base = torch.randn(B, Nb, 16)
    out = att(fine, base)

    # dense reference: same projections, full (Nf, Nb+Nf) with mask
    q = att.q(fine).view(B, g * g, 4, D // 4).permute(0, 2, 1, 3)
    kf, vf = att.kv_f(fine).chunk(2, -1)
    kb, vb = att.kv_b(base).chunk(2, -1)
    K = torch.cat([kb, kf], 1).view(B, -1, 4, D // 4).permute(0, 2, 1, 3)
    V = torch.cat([vb, vf], 1).view(B, -1, 4, D // 4).permute(0, 2, 1, 3)
    logits = q @ K.transpose(-2, -1) * (D // 4) ** -0.5
    idx = neighbor_index(g, r)
    allow = torch.zeros(g * g, Nb + g * g, dtype=torch.bool)
    allow[:, :Nb] = True
    for i in range(g * g):
        for j in idx[i][idx[i] >= 0]:
            allow[i, Nb + j] = True
    logits = logits.masked_fill(~allow, float("-inf"))
    ref = (logits.softmax(-1) @ V).permute(0, 2, 1, 3).reshape(B, g * g, D)
    ref = att.out(ref)
    assert torch.allclose(out, ref, atol=1e-5)


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
