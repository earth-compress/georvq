"""Tessera release policy: the mean channel and the rate ladder."""
import numpy as np
import pytest

from georvq.config import TierSpec
from georvq.inference.mean_codec import MeanChannel
from georvq.tessera import release as rel


# The real TierSpec, deliberately: rate arithmetic lives on it, and a
# stub here would let the release's accounting drift from the codec's.
SPECS = [TierSpec(grid=1, dim=512, codes=256, depth=20),
         TierSpec(grid=4, dim=512, codes=256, depth=20),
         TierSpec(grid=16, dim=256, codes=256, depth=20)]
SIDE = 128 * 6          # bands x bits_per_band


def test_mean_channel_round_trips_within_its_step():
    g = np.random.default_rng(0)
    mu = g.standard_normal((500, 32)).astype(np.float32) * 3 + 7
    ch = MeanChannel.fit(mu, bits=6)
    back = ch.round_trip(mu)
    # Inside the clip the error is bounded by half a step; outside it the
    # quantizer saturates deliberately, and at 4 sigma a Gaussian corpus
    # puts a handful of samples there.
    step = 2 * ch.clip * ch.scale / (ch.levels - 1)
    inside = np.abs((mu - ch.mu0) / ch.scale) <= ch.clip
    assert inside.mean() > 0.999
    assert np.all(np.abs(back - mu)[inside] <= step[None].repeat(
        mu.shape[0], 0)[inside] * 0.5 + 1e-4)
    assert ch.cost_bits() == 32 * 6          # no per-window side info


def test_mean_channel_costs_exactly_the_charged_rate():
    """The rate ladder is computed with bands x bits_per_band; the wire
    format must not quietly add a per-window scale on top."""
    ch = MeanChannel(np.zeros(128, np.float32), np.ones(128, np.float32),
                     bits=6)
    assert ch.cost_bits() == 768
    assert ch.cost_bits() / 64 ** 2 == pytest.approx(0.1875)


def test_mean_channel_saturates_rather_than_wrapping():
    ch = MeanChannel(np.zeros(4, np.float32), np.ones(4, np.float32), bits=4)
    far = np.array([[1e6, -1e6, 0.0, 0.0]], np.float32)
    codes = ch.encode(far)
    assert codes.max() == ch.levels - 1 and codes.min() == 0
    assert np.all(np.isfinite(ch.decode(codes)))


def test_mean_channel_survives_a_constant_band():
    mu = np.ones((10, 3), np.float32)          # zero variance everywhere
    ch = MeanChannel.fit(mu, bits=6)
    assert np.all(np.isfinite(ch.round_trip(mu)))


def test_mean_channel_persists(tmp_path):
    g = np.random.default_rng(1)
    ch = MeanChannel.fit(g.standard_normal((50, 8)).astype(np.float32), 6)
    p = tmp_path / "mc.npz"
    ch.save(p)
    got = MeanChannel.load(p)
    assert got.bits == ch.bits and got.clip == ch.clip
    assert np.array_equal(got.mu0, ch.mu0)
    assert np.array_equal(got.scale, ch.scale)


def test_ladder_saturates_coarse_then_mid_then_fine():
    lad = rel.ladder(SPECS, SIDE)
    assert lad[0] == (20, 0, 0)
    assert lad[1] == (20, 20, 0)
    assert lad[-1] == (20, 20, 20)
    # monotone in rate, which is what makes it a ladder
    bits = [rel.tuple_bits(SPECS, t, SIDE) for t in lad]
    assert bits == sorted(bits)


def test_ladder_rates_match_the_published_table():
    W = 64
    for tup, bpp in [((20, 0, 0), 0.227), ((20, 20, 0), 0.852),
                     ((20, 20, 5), 3.352), ((20, 20, 20), 10.852)]:
        got = rel.tuple_bits(SPECS, tup, SIDE) / W ** 2
        assert got == pytest.approx(bpp, abs=5e-4)


def test_refinement_is_priced_only_at_full_depth():
    """A refinement codes the residual after ALL of its tier's levels, so
    it is meaningless — and must not be charged — against a prefix."""
    W = 64
    full = rel.tuple_bits(SPECS, (20, 20, 20), SIDE, rel.REFINEMENT) / W ** 2
    part = rel.tuple_bits(SPECS, (20, 20, 19), SIDE, rel.REFINEMENT) / W ** 2
    assert full == pytest.approx(10.852 + 16.0, abs=5e-4)
    assert part == pytest.approx(10.852 - 0.5, abs=5e-4)   # no refine charge


def test_release_refines_the_fine_tier_and_not_by_default():
    """Both are measured choices that differ from the AlphaEarth release;
    a silent flip of either would change the shipped rate."""
    assert rel.REFINEMENT.tier == 2
    assert rel.REFINEMENT.quantizer == "lloyd"
    assert rel.REFINE_BY_DEFAULT is False


def test_rate_is_computed_in_exactly_one_place():
    """Regression: Codec.packet_bits once omitted the stats side channel,
    under-reporting a stats_cond model's cheapest rung by 83%. Both the
    generic and the release-side accounting must agree, because they are
    now the same arithmetic."""
    from georvq.model.codec import tuple_bits as codec_tuple_bits
    side = SIDE
    for depths in ([20, 0, 0], [20, 20, 0], [20, 20, 20]):
        assert rel.tuple_bits(SPECS, depths, side) == \
            codec_tuple_bits(SPECS, depths, side)
    # and the side channel is charged at EVERY rung, cheapest included
    bare = codec_tuple_bits(SPECS, [20, 0, 0], 0.0)
    assert rel.tuple_bits(SPECS, [20, 0, 0], side) == bare + side


def test_the_transmitted_codes_are_recoverable_from_the_packet_mean():
    """Why the packet carries only the dequantized mean.

    The mean it carries is already on the quantizer's reconstruction
    grid, so re-encoding it returns the codes that were transmitted.
    Storing both would be two fields for one fact."""
    g = np.random.default_rng(3)
    mu = g.standard_normal((256, 48)).astype(np.float32) * 2 - 1
    ch = MeanChannel.fit(mu, bits=6)
    sent = ch.encode(mu)
    on_the_packet = ch.decode(sent)
    assert np.array_equal(ch.encode(on_the_packet), sent)


def test_config_rejects_overlapping_batch_fractions():
    """full_frac and passthrough_frac are shares of one batch but live in
    different config sections, so nothing else notices them overrunning
    it -- past 1.0 the path fraction goes negative and every row is
    silently sampled at full depth."""
    from georvq.config import Config
    import pydantic
    base = {"data": {"source": "ae", "store_dir": "/tmp/x"},
            "model": {"window": 64, "bands": 64,
                      "tiers": ["1:32:64:4", "4:32:64:4"],
                      "full_frac": 0.7,
                      "vq": {"passthrough_frac": 0.5}}}
    with pytest.raises(pydantic.ValidationError, match="same batch"):
        Config(**base)
    base["model"]["full_frac"] = 0.4          # 0.4 + 0.5 <= 1.0
    Config(**base)


def test_refinement_spec_survives_a_json_round_trip():
    """A bundle records the spec its table was fitted for, and `load`
    reads it back rather than trusting today's module constant -- a table
    decoded under a different (tier, bits, rotation) is silent garbage."""
    from georvq.inference.refine import RefinementSpec
    for spec in (rel.REFINEMENT,
                 RefinementSpec(tier=0, bits=2, quantizer="uniform"),
                 RefinementSpec(tier=1, bits=3, quantizer="lloyd",
                                rotate_seed=11)):
        assert RefinementSpec.from_json(spec.to_json()) == spec


def _tiny_bundle(tmp_path):
    from georvq.config import load_config
    from georvq.factory import build_model
    cfg = load_config("configs/smoke.yaml", [
        "model.stats_cond={subtract_mean: true, cond_encoder: true, "
        "cond_decoder: true, bits_per_band: 6}"])
    model = build_model(cfg, "cpu")
    means = np.random.default_rng(0).normal(
        size=(64, cfg.model.bands)).astype(np.float32)
    out = tmp_path / "bundle"
    rel.build_bundle(str(out), model, cfg, means)
    return out


def test_load_accepts_its_own_bundle(tmp_path):
    out = _tiny_bundle(tmp_path)
    R = rel.load(str(out))
    assert R.bundle["bitstream_id"]


def test_load_refuses_a_bundle_whose_files_disagree_with_its_id(tmp_path):
    """A mean channel from another bundle decodes to plausible garbage,
    so load must refuse it rather than return a working-looking codec."""
    out = _tiny_bundle(tmp_path)
    p = out / "mean_channel.npz"
    d = dict(np.load(p))
    d["scale"] = d["scale"] * 1.5
    np.savez(p, **d)
    with pytest.raises(ValueError, match="bitstream"):
        rel.load(str(out))


# --------------------------------------------------------------------------
# encode / decode round trip on a tiny untrained release
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_release(tmp_path_factory):
    import torch
    from georvq.config import load_config
    from georvq.factory import build_model
    from georvq.inference.api import Codec
    torch.manual_seed(0)
    cfg = load_config("configs/smoke.yaml", [
        "data.source=tessera", "model.bands=8",
        "model.tiers=['1:16:32:3','2:8:32:2','4:8:16:2']",
        "model.min_depths=[1,0,0]",
        "model.stats_cond={subtract_mean: true, cond_encoder: true, "
        "cond_decoder: true, bits_per_band: 6}"])
    model = build_model(cfg, "cpu").eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, 8, 16, 16, generator=g) \
        + 2 * torch.randn(6, 8, 1, 1, generator=g)
    valid = torch.ones(6, 16, 16, dtype=torch.bool)
    valid[1, :5, :7] = False
    x = x * valid[:, None]
    _, res, mu = Codec(model).encode_batch(x, refine_tiers=(2,), valid=valid)
    rng = np.random.default_rng(0)
    means = (np.repeat(mu.numpy(), 8, 0)
             + rng.normal(size=(48, 8)).astype(np.float32) * 0.3)
    out = tmp_path_factory.mktemp("tessera_tiny") / "bundle"
    rel.build_bundle(str(out), model, cfg, means,
                     refine_residuals=np.concatenate([res[2]] * 8))
    return rel.load(str(out)), x, valid


def test_release_encodes_against_the_transmitted_mean(tiny_release):
    import torch
    from georvq.inference.api import Codec
    R, x, valid = tiny_release
    pk = R.encode(x, valid)
    mu = np.stack([p.mean for p in pk])
    codes = R.mean_channel.encode(mu)
    assert np.array_equal(R.mean_channel.decode(codes), mu)
    want = Codec(R.model).encode(x, valid=valid, mean=torch.from_numpy(mu))
    for a, b in zip(pk, want):
        for ia, ib in zip(a.indices, b.indices):
            assert np.array_equal(ia, ib)
    assert all(p.refinements is None for p in pk)        # off by default


def test_release_decode_is_monotone_prefix_and_restores_nodata(tiny_release):
    import torch
    R, x, valid = tiny_release
    pk = R.encode(x, valid)
    full = R.decode(pk, valid=valid)
    assert full.shape == x.shape
    assert torch.equal(R.decode(pk, valid=valid), full)
    assert not full[1][:, ~valid[1]].any()
    low = R.decode(pk, depths=[1, 0, 0], valid=valid)
    assert not torch.equal(low, full)


def test_release_refinement_applies_only_at_full_fine_depth(tiny_release):
    import torch
    R, x, valid = tiny_release
    plain = R.encode(x, valid)
    refined = R.encode(x, valid, refine=True)
    assert R.bits(refine=True) - R.bits() == pytest.approx(
        R.refine.cost_bits(R.specs[2].cells, R.specs[2].dim))
    for a, b in zip(plain, refined):
        for ia, ib in zip(a.indices, b.indices):
            assert np.array_equal(ia, ib)
        assert set(b.refinements) == {2}
    part = [3, 2, 1]
    assert torch.equal(R.decode(plain, depths=part, valid=valid),
                       R.decode(refined, depths=part, valid=valid))
    assert not torch.equal(R.decode(plain, valid=valid),
                           R.decode(refined, valid=valid))
    # the refinement shrinks the fine tier's residual
    from georvq.inference.api import Codec
    mu = torch.from_numpy(np.stack([p.mean for p in plain]))
    _, res, _ = Codec(R.model).encode_batch(x, refine_tiers=(2,),
                                            valid=valid, mean=mu)
    q = np.stack([p.refinements[2] for p in refined])
    assert ((res[2] - q) ** 2).sum() < (res[2] ** 2).sum()


def test_tile_helpers_match_per_window_coding(tiny_release):
    import torch
    from georvq.tessera import decode_tile, encode_tile, to_windows, prep
    R, _, _ = tiny_release
    emb = np.random.default_rng(2).normal(size=(8, 32, 48)).astype(np.float32)
    emb *= 32.0
    packets, grid = encode_tile(R, emb, batch=4)
    assert grid == (1, 2, 3) and len(packets) == 6
    tile = decode_tile(R, packets, grid, depths=[3, 2, 2], batch=4)
    assert tile.shape == (1, 8, 32, 48)
    x, v = prep(emb)
    xw, vw, _ = to_windows(x, v, R.window)
    want = R.decode(R.encode(xw, vw), depths=[3, 2, 2])
    assert torch.equal(to_windows(tile, v, R.window)[0], want)
