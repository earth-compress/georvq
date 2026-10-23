"""`model.stats_cond` on a multi-group source (S2): the model standardises
each window as S2's own normaliser does, conditions the conv encoder and
the decoder on the 26 statistics, and scores the loss in the same
standardised space. Models without stats_cond (the S2 v1 release) are
pinned by test_s2_v1_bundle and test_s2_v1."""
import os

import numpy as np
import pytest
import torch

from georvq.config import Config, load_config
from georvq.data.build import ShardWriter
from georvq.data.pool import GroupBatch
from georvq.factory import build_model
from georvq.model.stats_cond import (center, patch_stats, standardize,
                                     uncenter)
from georvq.sources.s2 import S2Normalizer, S2Source

SC = {"subtract_mean": True, "use_std": True, "cond_encoder": True,
      "cond_decoder": True, "std_floor": 1e-4, "bits_per_band": 10}


def _cfg(stats_cond=SC, window=24, tiers=("1:16:32:3", "4:16:32:3"),
         **extra):
    model = {"window": window, "bands": 13, "tiers": list(tiers),
             "min_depths": [3, 0], "depth_sampling": "path",
             "tokenizer": {"kind": "source", "dim": 16},
             "decoder": {"dim": 32, "depth": 1, "heads": 4, "head_feat": 8,
                         "cls_bridge": False}}
    if stats_cond is not None:
        model["stats_cond"] = stats_cond
    model.update(extra.pop("model", {}))
    return Config.model_validate({
        "data": {"source": "s2", "store_dir": "/tmp/unused"},
        "model": model, **extra})


def _raw(b=5, w=24, seed=0):
    """uint16 DN with nodata (0) in a different region of every band, one
    window with a band that is all nodata and one with a constant band."""
    g = torch.Generator().manual_seed(seed)
    raw = {}
    for name, (c, r) in {"10m": (4, 1), "20m": (6, 2), "60m": (3, 6)}.items():
        px = w // r
        t = torch.randint(1, 10000, (b, c, px, px), generator=g)
        for k in range(c):
            t[0, k, : (k % px) + 1] = 0          # swath edge, per band
        t[1, 0] = 0                                # a band with no data
        t[2, -1] = 1234                            # a constant band
        raw[name] = t.to(torch.int32)
    return raw


def test_group_stats_match_s2_normalizer_bit_for_bit():
    old = S2Source(_cfg(None).model)
    new = S2Source(_cfg().model)
    raw = _raw()
    xn, valid_old = old.prep(raw, "cpu")
    x, valid = new.prep(raw, "cpu")
    assert torch.equal(valid, valid_old)
    st = patch_stats(x, valid, use_std=True, std_floor=1e-4)
    assert st.shape == (5, 26)
    assert torch.equal(st, S2Normalizer.stats_vector(old.last_stats()))
    xc = center(x, st, valid)
    assert isinstance(xc, GroupBatch)
    for g in xn:
        assert torch.equal(xc[g], xn[g]), g


def test_uncenter_restores_the_field_on_valid_pixels():
    src = S2Source(_cfg().model)
    x, valid = src.prep(_raw(), "cpu")
    st = patch_stats(x, valid, use_std=True, std_floor=1e-4)
    back = uncenter(center(x, st, valid), st, valid)
    from georvq.sources.s2 import group_valid
    for g in x:
        m = group_valid(valid, g).expand_as(x[g])
        assert torch.allclose(back[g][m], x[g][m], atol=1e-6)
        assert (back[g][~m] == 0).all()
    # standardize is center without the mask
    s = standardize(x, st)
    for g in x:
        m = x[g] != 0
        assert torch.equal(s[g][m], center(x, st)[g][m])


def test_conditioning_starts_at_zero_and_receives_gradient():
    """The stats embeddings are zero-initialised, so the first forward
    pass is the unconditioned model's on the standardised field, and the
    first backward pass reaches both embeddings."""
    torch.manual_seed(0)
    m = build_model(_cfg())
    assert m.stats_enc is not None and m.stats_dec is not None
    x, valid = m.source.prep(_raw(), "cpu")
    st = m._stats_of(x, valid)
    zs = m.encode_levels(x, st, valid)
    ref = m.tokenizer(center(x, st, valid))
    for a, b in zip(zs, ref):
        assert torch.equal(a, b)
    Ls = torch.tensor([[3, 2]] * 5)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    def step():
        opt.zero_grad()
        zqs, commit, _ = m.quantize(m.encode_levels(x, st, valid), Ls)
        out = m.decode(zqs, Ls, st, valid)
        loss, _ = m.source.recon_stats(out.float(), x, valid)
        (loss + commit).backward()
    step()
    assert m.stats_enc.mlp[-1].weight.grad.abs().sum() > 0
    # the decoder's AdaLN modulation is zero-initialised too, so its
    # conditioning inputs receive gradient from the second step on
    opt.step()
    step()
    for emb in (m.stats_enc, m.stats_dec):
        assert emb.mlp[-1].weight.grad.abs().sum() > 0
    # once the embedding is non-zero the encoder reads the statistics
    with torch.no_grad():
        m.stats_enc.mlp[-1].weight.normal_()
        st2 = st.clone()
        st2[:, :13] += 0.05
        a = m.encode_levels(x, st, valid)[1]
        b = m.tokenizer(center(x, st, valid), stats_bias=m.stats_enc(st2))[1]
    assert not torch.allclose(a, b)


def test_loss_is_scored_in_the_standardised_space():
    """The stats_cond model's loss on a reflectance decode equals the S2
    v1 loss on the same decode in standardised units."""
    old = S2Source(_cfg(None).model)
    new = S2Source(_cfg().model)
    raw = _raw()
    xn, valid = old.prep(raw, "cpu")
    x, _ = new.prep(raw, "cpu")
    st = patch_stats(x, valid, use_std=True, std_floor=1e-4)
    torch.manual_seed(1)
    y = GroupBatch({g: t + 0.3 * torch.randn_like(t) for g, t in xn.items()})
    l_old, m_old = old.recon_stats(y, xn, valid)
    l_new, m_new = new.recon_stats(uncenter(y, st, valid), x, valid)
    assert torch.allclose(l_old, l_new, rtol=1e-5)
    spec_old, spec_new = old.report(None), new.report(None)
    for mo, mn in zip([spec_old.quality, *spec_old.extra],
                      [spec_new.quality, *spec_new.extra]):
        assert torch.allclose(mo(y, xn, valid),
                              mn(uncenter(y, st, valid), x, valid),
                              rtol=1e-4), mo.key


def test_rate_is_charged_once():
    """stats_cond charges the 26 statistics at bits_per_band; the source's
    own stats channel is not charged on top."""
    m = build_model(_cfg())
    assert m.side_bits() == 26 * 10
    m0 = build_model(_cfg(None))
    assert m0.side_bits() == 26 * 16 + 8          # f16 stats, as before


def test_s2_needs_the_standardising_options():
    with pytest.raises(ValueError, match="use_std"):
        build_model(_cfg({**SC, "use_std": False}))


def test_models_without_stats_cond_are_unchanged():
    """No new parameters, the normaliser still runs in prep, and the conv
    encoder's state dict keys are those of beta_arch_w36."""
    m = build_model(load_config("configs/s2/beta_arch_w36.yaml"))
    assert not any(k.startswith("stats_") for k in m.state_dict())
    assert m.stats_enc is None and m.stats_dec is None
    x, _ = m.source.prep(_raw(w=36), "cpu")
    assert m.source.last_stats() is not None
    assert abs(float(x["10m"][3].mean())) < 1e-4


def test_s2_v2_preset_builds():
    cfg = load_config("configs/s2/s2_v2.yaml")
    assert cfg.model.vq.book_bits == 8
    m = build_model(cfg)
    assert m.side_bits() == 260
    n = sum(p.numel() for k, p in m.named_parameters()
            if k.startswith("stats_"))
    # two StatsEmbed: LayerNorm(26) and 26 -> d -> d, for d = 128 and 384
    want = sum(2 * 26 + 26 * d + d + d * d + d for d in (128, 384))
    assert n == want
    x = GroupBatch({"10m": torch.rand(2, 4, 36, 36) + 0.01,
                    "20m": torch.rand(2, 6, 18, 18) + 0.01,
                    "60m": torch.rand(2, 3, 6, 6) + 0.01})
    zs = m.encode_levels(x, m._stats_of(x))
    assert [tuple(z.shape) for z in zs] == [(2, 1, 128), (2, 36, 128)]


GROUPS = {"10m": (4, 48), "20m": (6, 24), "60m": (3, 8)}


@pytest.fixture
def s2_store(tmp_path):
    rng = np.random.default_rng(0)
    w = ShardWriter(tmp_path / "s2", shard_size=4, dtype=np.uint16,
                    groups=GROUPS)
    for i in range(6):
        chip = {g: rng.integers(1, 10000, size=(b, px, px), dtype=np.uint16)
                for g, (b, px) in GROUPS.items()}
        w.add(chip, {"scene_id": f"s{i//4}", "patch_row": i, "patch_col": 0,
                     "crs": "EPSG:32601", "lon": float(i), "lat": 0.0,
                     "year": 0}, log_every=0)
    w.finalize("s2", nodata=0, grid_period=6)
    return tmp_path / "s2"


def test_training_with_stats_cond_and_int8_books(s2_store, tmp_path):
    """The trainer runs a stats_cond S2 model through warmup, book fit,
    quantised steps and validation, and the books stay on the int8
    grid."""
    from georvq.repro import load_checkpoint
    from georvq.train.loop import train
    cfg = _cfg(
        data={"source": "s2", "store_dir": str(s2_store), "val_frac": 0.0,
              "pool": {"size": 6, "workers": 0, "device": "cpu"}},
        model={"vq": {"warmup_steps": 3, "init_mode": "tree",
                      "init_per_code": 4, "reset_mode": "split",
                      "book_bits": 8, "passthrough_frac": 0.15}},
        train={"steps": 8, "batch_windows": 8, "amp": False,
               "val_interval": 4, "fig_interval": 4, "save_interval": 4},
        log={"run_tag": "sc", "wandb": False, "out_dir": str(tmp_path)})
    run = train(cfg, gpu=-1)
    ck = load_checkpoint(os.path.join(run, "model.pt"))
    m = build_model(cfg)
    m.load_state_dict(ck["model"])
    from georvq.inference.bundle import book_to_int
    for rvq in m.rvqs:
        for lev in rvq.levels:
            book_to_int(lev.codebook.weight.detach())  # raises if off-grid
