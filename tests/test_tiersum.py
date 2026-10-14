"""Tier-sum decoder: per-tier maps summed, absent tiers truly absent,
coarse-only rows on the transformer-free path."""
import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model


@pytest.fixture
def ts_cfg(tiny_raw_cfg):
    raw = dict(tiny_raw_cfg)
    raw["model"] = dict(
        raw["model"],
        depth_sampling="path",
        decoder={"kind": "tiersum", "dim": 32, "depth": 2, "heads": 4,
                 "map_px": [4, 8], "solo_hidden": 64, "solo_layers": 2})
    return Config.model_validate(raw)


def _batch(cfg, B=4):
    W = cfg.model.window
    x = torch.randn(B, cfg.model.bands, W, W)
    v = torch.ones(B, W, W, dtype=torch.bool)
    return x, v


def test_mixed_presence_decode_and_absence(ts_cfg):
    torch.manual_seed(0)
    m = build_model(ts_cfg, torch.device("cpu")).eval()
    x, v = _batch(ts_cfg)
    with torch.no_grad():
        zs, cums = m.level_cumsums(x, valid=v)
        Ls = torch.tensor([[3, 0], [3, 1], [3, 3], [2, 0]])
        zqs = [c[torch.arange(4), :, Ls[:, i]] for i, c in enumerate(cums)]
        out = m.decode(zqs, Ls, valid=v)
        assert out.shape == x.shape and torch.isfinite(out).all()
        # a tier at depth 0 has NO influence, present tiers do
        z2 = [q.clone() for q in zqs]
        z2[1] = torch.randn_like(z2[1])
        out2 = m.decode(z2, Ls, valid=v)
        absent = Ls[:, 1] == 0
        assert torch.allclose(out[absent], out2[absent], atol=1e-6)
        assert not torch.allclose(out[~absent], out2[~absent], atol=1e-4)


def test_solo_rows_match_solo_only_batch(ts_cfg):
    """Group routing must not depend on batch composition."""
    torch.manual_seed(0)
    m = build_model(ts_cfg, torch.device("cpu")).eval()
    x, v = _batch(ts_cfg)
    with torch.no_grad():
        _, cums = m.level_cumsums(x, valid=v)
        Ls = torch.tensor([[3, 0], [3, 3], [3, 0], [3, 2]])
        zqs = [c[torch.arange(4), :, Ls[:, i]] for i, c in enumerate(cums)]
        mixed = m.decode(zqs, Ls, valid=v)
        alone = m.decode([q[0:1] for q in zqs], Ls[0:1], valid=v[0:1])
        assert torch.allclose(mixed[0], alone[0], atol=1e-6)


def test_feat_sum_render_keeps_semantics(ts_cfg):
    """sum_feat mode: absence semantics survive the shared renderer, and
    the FiLM zero-init means conditioning starts neutral but reaches the
    renderer's gradient."""
    raw = ts_cfg.model_dump(mode="json")
    raw["model"]["decoder"]["sum_feat"] = 16
    cfg = Config.model_validate(raw)
    torch.manual_seed(0)
    m = build_model(cfg, torch.device("cpu"))
    x, v = _batch(cfg)
    with torch.no_grad():
        _, cums = m.level_cumsums(x, valid=v)
        Ls = torch.tensor([[3, 0], [3, 1], [3, 3], [2, 0]])
        zqs = [c[torch.arange(4), :, Ls[:, i]] for i, c in enumerate(cums)]
        out = m.decode(zqs, Ls, valid=v)
        assert out.shape == x.shape
        z2 = [q.clone() for q in zqs]
        z2[1] = torch.randn_like(z2[1])
        out2 = m.decode(z2, Ls, valid=v)
        absent = Ls[:, 1] == 0
        assert torch.allclose(out[absent], out2[absent], atol=1e-6)
    out, commit, _ = m.forward_train(x, Ls, valid=v)
    (out.pow(2).mean() + commit).backward()
    assert m.decoder.rc1.weight.grad is not None
    assert m.decoder.film.weight.grad is not None


def test_forward_train_backward_reaches_all_paths(ts_cfg):
    torch.manual_seed(0)
    m = build_model(ts_cfg, torch.device("cpu"))
    x, v = _batch(ts_cfg, B=6)
    Ls = torch.tensor([[3, 0], [2, 0], [3, 1], [3, 2], [3, 3], [3, 3]])
    out, commit, _ = m.forward_train(x, Ls, valid=v)
    (out.pow(2).mean() + commit).backward()
    assert m.decoder.solo[0].weight.grad is not None
    assert m.decoder.heads[1].weight.grad is not None
    assert m.decoder.blocks[0].mlp[0].weight.grad is not None


def test_decoder_ff_ratio_widens_only_the_decoder(ts_cfg):
    def counts(ff):
        raw = ts_cfg.model_dump(mode="json")
        raw["model"]["decoder"]["ff_ratio"] = ff
        m = build_model(Config.model_validate(raw), torch.device("cpu"))
        return (sum(p.numel() for p in m.decoder.parameters()),
                sum(p.numel() for p in m.tokenizer.parameters()))
    unset, same, wide = counts(None), counts(2.0), counts(3.0)
    assert unset == same                 # None shares model.ff_ratio (2.0)
    assert wide[0] > same[0] and wide[1] == same[1]
