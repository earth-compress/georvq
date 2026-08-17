"""Stats conditioning: the mean really is removed, really is put back, and
cannot be silently forgotten."""
import pytest
import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.model.stats_cond import center, patch_mean, uncenter

TIERS = "model.tiers=['1:16:32:3','4:8:32:3']"


def _model(stats="{}", extra=()):
    cfg = load_config("configs/smoke.yaml", [
        TIERS, f"model.stats_cond={stats}", *extra])
    return cfg, build_model(cfg, "cpu")


def _batch(cfg, B=3, nodata=True):
    W = cfg.model.window
    x = torch.randn(B, cfg.model.bands, W, W) + 5.0
    v = torch.ones(B, W, W, dtype=torch.bool)
    if nodata:
        v[0, :, : W // 2] = False           # half of sample 0 is nodata
        x = x * v.unsqueeze(1)
    return x, v


def test_patch_mean_ignores_nodata():
    """nodata is exactly zero in the source; counting it would drag the
    mean toward zero by the nodata fraction, making the conditioning
    signal depend on cloud/edge geometry rather than on content."""
    cfg, _ = _model()
    x, v = _batch(cfg)
    mu = patch_mean(x, v)
    ref = x[0, :, :, cfg.model.window // 2:].mean(dim=(1, 2))
    assert torch.allclose(mu[0], ref, atol=1e-5)
    naive = x.mean(dim=(2, 3))
    assert not torch.allclose(mu[0], naive[0], atol=1e-3), \
        "test is vacuous unless nodata actually shifts the naive mean"


def test_center_uncenter_round_trip_on_valid_pixels():
    cfg, _ = _model()
    x, v = _batch(cfg)
    mu = patch_mean(x, v)
    back = uncenter(center(x, mu, v), mu, v)
    assert torch.allclose(back[v.unsqueeze(1).expand_as(x)],
                          x[v.unsqueeze(1).expand_as(x)], atol=1e-5)


def test_centered_field_has_zero_mean_and_nodata_stays_zero():
    cfg, _ = _model()
    x, v = _batch(cfg)
    mu = patch_mean(x, v)
    xc = center(x, mu, v)
    assert torch.allclose(patch_mean(xc, v), torch.zeros_like(mu), atol=1e-5)
    assert float(xc[0, :, :, : cfg.model.window // 2].abs().max()) == 0.0


def test_decode_without_stats_raises_rather_than_corrupting():
    """The failure this guards against is silent: decode would return a
    mean-REMOVED field, and every cosine/MSE metric downstream would be
    quietly wrong rather than erroring."""
    cfg, m = _model()
    x, v = _batch(cfg)
    st = m.patch_mean(x, v)
    zs = m.encode_levels(x, st, v)
    Ls = torch.tensor([[3, 3]] * x.shape[0])
    with pytest.raises(ValueError, match="stats_cond is enabled"):
        m.decode(zs, Ls)
    with pytest.raises(ValueError, match="stats_cond is enabled"):
        m.encode_levels(x)
    m.decode(zs, Ls, st, v)          # the same call WITH stats is fine


def test_decode_adds_the_mean_back():
    """Two decodes of the same latents differing only in the stats argument
    must differ by exactly that difference in mean on valid pixels."""
    cfg, m = _model()
    x, v = _batch(cfg, nodata=False)
    st = m.patch_mean(x, v)
    zs = m.encode_levels(x, st, v)
    Ls = torch.tensor([[3, 3]] * x.shape[0])
    m.stats_dec = None               # isolate the add-back from conditioning
    a = m.decode(zs, Ls, st, v)
    b = m.decode(zs, Ls, st + 2.0, v)
    assert torch.allclose(b - a, torch.full_like(a, 2.0), atol=1e-4)


def test_conditioning_actually_reaches_encoder_and_decoder():
    cfg, m = _model()
    x, v = _batch(cfg, nodata=False)
    st = m.patch_mean(x, v)
    # Two separate zero-inits stand between the stats and the output, and
    # both must be defeated for this test to mean anything: StatsEmbed's
    # last layer, and AdaLNBlock's `ada` projection (DiT AdaLN-Zero), which
    # makes EVERY conditioning signal -- including the existing depth
    # thermometer -- a no-op at initialisation.
    for emb in (m.stats_enc, m.stats_dec):
        torch.nn.init.normal_(emb.mlp[-1].weight, std=0.1)
    for blk in m.decoder.blocks:
        torch.nn.init.normal_(blk.ada[-1].weight, std=0.02)
    z0 = m.encode_levels(x, st, v)
    z1 = m.encode_levels(x, st + 1.0, v)
    assert not torch.allclose(z0[0], z1[0]), "encoder ignores the stats"
    Ls = torch.tensor([[3, 3]] * x.shape[0])
    d0 = m.decode(z0, Ls, st, v)
    # same latents, same add-back mean, different CONDITIONING only
    m2_stats = st.clone()
    d1 = m.decode(z0, Ls, m2_stats, v)
    assert torch.allclose(d0, d1)
    saved = m.stats_dec
    m.stats_dec = None
    d2 = m.decode(z0, Ls, st, v)
    m.stats_dec = saved
    assert not torch.allclose(d0, d2), "decoder ignores the stats"


def test_zero_init_starts_from_the_unconditioned_model():
    """The stats embeddings are zero-init so a run begins exactly where an
    unconditioned one does and the conditioning has to earn its keep."""
    cfg, m = _model()
    x, v = _batch(cfg)
    st = m.patch_mean(x, v)
    assert float(m.stats_enc(st).abs().max()) == 0.0
    assert float(m.stats_dec(st).abs().max()) == 0.0


def test_forward_train_end_to_end():
    cfg, m = _model()
    x, v = _batch(cfg)
    st = m.patch_mean(x, v)
    Ls = torch.tensor([[3, 3]] * x.shape[0])
    out, commit, _ = m.forward_train(x, Ls, stats=st, valid=v)
    assert out.shape == x.shape
    (out.pow(2).mean() + commit).backward()
    assert m.stats_enc.mlp[0].weight.grad is not None


def test_disabled_by_default_and_use_std_rejected():
    cfg, m = _model(stats="null")
    assert m.stats_cond is None and m.stats_enc is None
    x, v = _batch(cfg)
    m.encode_levels(x)               # no stats needed when off
    with pytest.raises(Exception, match="use_std"):
        load_config("configs/smoke.yaml",
                    [TIERS, "model.stats_cond={use_std: true}"])
