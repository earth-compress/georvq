"""Ladder decoder: per-prefix exits, deep supervision, absence semantics."""
import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model


@pytest.fixture
def ld_cfg(tiny_raw_cfg):
    raw = dict(tiny_raw_cfg)
    raw["model"] = dict(
        raw["model"],
        depth_sampling="path",
        decoder={"kind": "ladder", "dim": 32, "depth": 2, "heads": 4,
                 "head_feat": 8, "map_px": [4, 16], "solo_layers": 2})
    return Config.model_validate(raw)


def _batch(cfg, B=4):
    W = cfg.model.window
    x = torch.randn(B, cfg.model.bands, W, W)
    v = torch.ones(B, W, W, dtype=torch.bool)
    return x, v


def test_exits_and_aux_follow_the_prefix(ld_cfg):
    torch.manual_seed(0)
    m = build_model(ld_cfg, torch.device("cpu"))
    x, v = _batch(ld_cfg)
    Ls = torch.tensor([[2, 0], [3, 0], [3, 1], [3, 3]])
    m.train()
    out, commit, _ = m.forward_train(x, Ls, valid=v)
    assert out.shape == x.shape and torch.isfinite(out).all()
    aux = m.decoder.last_aux
    # the two rows with mid active get a coarse aux exit; nothing else
    assert len(aux) == 1
    rows, amap = aux[0]
    assert sorted(rows.tolist()) == [2, 3] and amap.shape[0] == 2


def test_absent_mid_has_no_influence(ld_cfg):
    torch.manual_seed(0)
    m = build_model(ld_cfg, torch.device("cpu")).eval()
    x, v = _batch(ld_cfg)
    with torch.no_grad():
        _, cums = m.level_cumsums(x, valid=v)
        Ls = torch.tensor([[3, 0], [3, 2], [2, 0], [3, 3]])
        zqs = [c[torch.arange(4), :, Ls[:, i]] for i, c in enumerate(cums)]
        out = m.decode(zqs, Ls, valid=v)
        z2 = [q.clone() for q in zqs]
        z2[1] = torch.randn_like(z2[1])
        out2 = m.decode(z2, Ls, valid=v)
        absent = Ls[:, 1] == 0
        assert torch.allclose(out[absent], out2[absent], atol=1e-6)
        assert not torch.allclose(out[~absent], out2[~absent], atol=1e-4)


def test_solo_rows_match_solo_only_batch(ld_cfg):
    torch.manual_seed(0)
    m = build_model(ld_cfg, torch.device("cpu")).eval()
    x, v = _batch(ld_cfg)
    with torch.no_grad():
        _, cums = m.level_cumsums(x, valid=v)
        Ls = torch.tensor([[3, 0], [3, 3], [2, 0], [3, 1]])
        zqs = [c[torch.arange(4), :, Ls[:, i]] for i, c in enumerate(cums)]
        mixed = m.decode(zqs, Ls, valid=v)
        alone = m.decode([q[0:1] for q in zqs], Ls[0:1], valid=v[0:1])
        assert torch.allclose(mixed[0], alone[0], atol=1e-6)


def test_deep_supervision_reaches_every_exit(ld_cfg):
    torch.manual_seed(0)
    m = build_model(ld_cfg, torch.device("cpu"))
    x, v = _batch(ld_cfg)
    Ls = torch.tensor([[3, 0], [3, 3], [3, 2], [2, 0]])
    m.train()
    out, commit, _ = m.forward_train(x, Ls, valid=v)
    loss = out.pow(2).mean() + commit
    for rows, amap in m.decoder.last_aux:
        loss = loss + amap.pow(2).mean()
    loss.backward()
    assert m.decoder.stem[0].mlp[0].weight.grad is not None
    assert m.decoder.heads[0].to_patch.weight.grad is not None
    assert m.decoder.heads[1].to_patch.weight.grad is not None
    assert m.decoder.stages[0][0].mlp[0].weight.grad is not None
    # activation telemetry populated for every stage that ran, plus the
    # shared-input probes (conditioning and normalized input tokens)
    assert {"stem", "stage1", "cond", "in0"} <= set(m.decoder.act_max)
    assert "stage2" not in m.decoder.act_max   # no 3rd tier in this cfg
    assert all(v > 0 for v in m.decoder.act_max.values())


def test_compile_trunks_reaches_stem_and_stages(ld_cfg):
    from georvq.factory import compile_trunks
    m = build_model(ld_cfg, torch.device("cpu"))
    compile_trunks(m)
    blocks = list(m.decoder.stem) + [b for st in m.decoder.stages for b in st]
    assert blocks and all("forward" in vars(b) for b in blocks)


def test_decoder_ff_ratio_widens_only_the_decoder(ld_cfg):
    def counts(ff):
        raw = ld_cfg.model_dump(mode="json")
        raw["model"]["decoder"]["ff_ratio"] = ff
        m = build_model(Config.model_validate(raw), torch.device("cpu"))
        return (sum(p.numel() for p in m.decoder.parameters()),
                sum(p.numel() for p in m.tokenizer.parameters()))
    unset, same, wide = counts(None), counts(2.0), counts(3.0)
    assert unset == same                 # None shares model.ff_ratio (2.0)
    assert wide[0] > same[0] and wide[1] == same[1]
