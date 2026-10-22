"""Sentinel-2 adapter: multi-group store, pool, native-res stems and
multi-res heads through the pyramid, toy optimization."""
import numpy as np
import pytest
import torch

from georvq.config import Config
from georvq.data.build import ShardWriter
from georvq.data.pool import ChipPool
from georvq.data.store import ChipStore
from georvq.factory import build_model

GROUPS = {"10m": (4, 48), "20m": (6, 24), "60m": (3, 8)}


@pytest.fixture
def s2_store(tmp_path):
    rng = np.random.default_rng(0)
    w = ShardWriter(tmp_path / "s2", shard_size=4, dtype=np.uint16,
                    groups=GROUPS)
    for i in range(6):
        chip = {g: rng.integers(0, 10000, size=(b, px, px), dtype=np.uint16)
                for g, (b, px) in GROUPS.items()}
        w.add(chip, {"scene_id": f"s{i//4}", "patch_row": i, "patch_col": 0,
                     "crs": "EPSG:32601", "lon": float(i), "lat": 0.0,
                     "year": 0}, log_every=0)
    w.finalize("s2", nodata=0, grid_period=6)
    return ChipStore(tmp_path / "s2")


def _cfg():
    return Config.model_validate({
        "data": {"source": "s2", "store_dir": "/tmp/unused"},
        "model": {
            "window": 24, "bands": 32,
            "tiers": ["1:16:32:3", "4:8:32:3"],
            "ff_ratio": 2.0,
            "tokenizer": {"dim": 32, "depth": 1},
            "decoder": {"dim": 32, "depth": 1, "heads": 4, "head_feat": 8},
        },
        "log": {"run_tag": "s2test", "wandb": False},
    })


def test_multigroup_store(s2_store):
    assert [g["name"] for g in s2_store.groups] == ["10m", "20m", "60m"]
    assert s2_store.chip_px == 48 and s2_store.grid_period == 6
    assert s2_store.open_shard(0, "20m").shape == (4, 6, 24, 24)
    assert len(s2_store.index) == 6


def _resident(store):
    """Every chip resident, no refresh threads: what the S2 tests used to
    get from the in-RAM group loader."""
    return ChipPool(store, store.index, pool_size=len(store.index),
                    device=torch.device("cpu"), workers=0, seed=0)


def test_group_loader(s2_store):
    loader = _resident(s2_store)
    batch = loader.sample_batch(5, token=24)
    assert batch["10m"].shape == (5, 4, 24, 24)
    assert batch["20m"].shape == (5, 6, 12, 12)
    assert batch["60m"].shape == (5, 3, 4, 4)
    b2, rows = loader.sample_batch(3, token=24, return_rows=True)
    assert rows.shape == (3,)


def test_s2_model_forward(s2_store):
    cfg = _cfg()
    torch.manual_seed(0)
    model = build_model(cfg)
    model.eval()
    source = model.source
    loader = _resident(s2_store)
    raw = loader.sample_batch(4, token=24)
    x, valid = source.prep(raw, "cpu")
    assert valid.shape == (4, 24, 24)
    zs = model.encode_levels(x)
    assert [tuple(z.shape) for z in zs] == [(4, 1, 16), (4, 16, 8)]
    Ls = torch.tensor([[3, 2]]).expand(4, -1)
    out, commit, _ = model.forward_train(x, Ls)
    assert out["10m"].shape == (4, 4, 24, 24)
    assert out["20m"].shape == (4, 6, 12, 12)
    assert out["60m"].shape == (4, 3, 4, 4)
    loss, m = source.recon_stats(out, x, valid)
    assert torch.isfinite(loss) and "mse_60m" in m
    # normalizer round-trip restores the raw scale
    norm = source.normalizer(cfg.model)
    den = norm.denormalize(x, source.last_stats())
    raw_f = {k: v.float() / 10000.0 for k, v in raw.items()}
    for g in GROUPS:
        assert torch.allclose(den[g], raw_f[g], atol=1e-4)
    assert norm.stats_vector(source.last_stats()).shape == (4, 26)


def test_s2_toy_training(s2_store):
    """A short optimization drives the multi-res reconstruction loss down."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = build_model(cfg)
    source = model.source
    loader = _resident(s2_store)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    losses = []
    from georvq.model.codec import sample_monotone_depths
    for step in range(60):
        raw = loader.sample_batch(8, token=24)
        x, valid = source.prep(raw, "cpu")
        Ls = sample_monotone_depths(8, model.specs, "cpu",
                                    mins=cfg.model.min_depths)
        zs = model.encode_levels(x)
        zqs, commit, _ = model.quantize(zs, Ls)
        out = model.decode(zqs, Ls)
        loss, _ = source.recon_stats(out, x, valid)
        loss = loss + 0.25 * commit
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


# --------------------------------------------------- streaming multi-group

def _pool(store, **kw):
    from georvq.data.pool import ChipPool
    return ChipPool(store, store.index, pool_size=kw.pop("pool_size", 6),
                    device=torch.device("cpu"), workers=1,
                    refresh_chips_per_s=1000.0, seed=0, **kw)


def test_chippool_streams_a_multigroup_store(s2_store):
    """The blocker this branch exists to remove: the streaming pool was
    single-plane, so an S2 run could not be trained at all."""
    pool = _pool(s2_store)
    try:
        assert set(pool.ratio) == {"10m", "20m", "60m"}
        assert pool.ratio == {"10m": 1, "20m": 2, "60m": 6}
        assert pool.period == 6          # forced up by the 60m grid
        b = pool.sample_batch(5, token=24)
        assert b["10m"].shape == (5, 4, 24, 24)
        assert b["20m"].shape == (5, 6, 12, 12)
        assert b["60m"].shape == (5, 3, 4, 4)
        assert b.shape == (5,)           # GroupBatch behaves like a batch
        assert b.device == torch.device("cpu")
    finally:
        pool.shutdown()


def test_pool_crops_are_registered_across_groups(tmp_path):
    """A shared crop origin is the whole point: the 20m crop must cover the
    same ground as the 10m crop, at half the sampling."""
    from georvq.data.store import ChipStore

    # each pixel's value IS its row index, per group
    w = ShardWriter(tmp_path / "reg", shard_size=2, dtype=np.uint16,
                    groups={"10m": (1, 48), "20m": (1, 24)})
    rows10 = np.arange(48, dtype=np.uint16)[:, None].repeat(48, 1)
    rows20 = np.arange(24, dtype=np.uint16)[:, None].repeat(24, 1)
    for i in range(2):
        w.add({"10m": rows10[None], "20m": rows20[None]},
              {"scene_id": "s", "patch_row": i, "patch_col": 0,
               "crs": "EPSG:32601", "lon": 0.0, "lat": 0.0, "year": 0},
              log_every=0)
    w.finalize("s2", nodata=None, grid_period=2)
    store = ChipStore(tmp_path / "reg")

    pool = _pool(store, pool_size=2)
    try:
        b = pool.sample_batch(8, token=8)
        r10 = b["10m"][:, 0, 0, 0].to(torch.int64)
        r20 = b["20m"][:, 0, 0, 0].to(torch.int64)
        assert torch.equal(r20, r10 // 2)
        assert (r10 % 2 == 0).all()      # origins land on the 20m grid
    finally:
        pool.shutdown()


def test_fixed_val_tokens_multigroup(s2_store):
    from georvq.data.pool import fixed_val_tokens

    v = fixed_val_tokens(s2_store, s2_store.index, token=24, per_chip=4,
                         nodata=None)
    assert v["10m"].shape[1:] == (4, 24, 24)
    assert v["20m"].shape[1:] == (6, 12, 12)
    assert v["60m"].shape[1:] == (3, 4, 4)
    n = v.shape[0]
    assert v["10m"].shape[0] == v["60m"].shape[0] == n
    # sliceable like a tensor batch, which is how validate() chunks it
    assert v[:2]["20m"].shape == (2, 6, 12, 12)


def test_singleplane_pool_is_unchanged(tmp_path):
    """The refactor must not move a single-plane pool's sampling: every
    existing preset's training data depends on it."""
    from georvq.data.build import build_synthetic_store
    from georvq.data.pool import ChipPool
    from georvq.data.store import ChipStore

    build_synthetic_store(tmp_path / "flat", n_chips=4, bands=2, chip_px=32)
    store = ChipStore(tmp_path / "flat")
    pool = ChipPool(store, store.index, 4, torch.device("cpu"), workers=1,
                    refresh_chips_per_s=1000.0, seed=0)
    try:
        b = pool.sample_batch(3, token=16)
        assert isinstance(b, torch.Tensor)      # NOT a GroupBatch
        assert b.shape == (3, 2, 16, 16)
        assert pool.ratio == {None: 1}
    finally:
        pool.shutdown()


def test_s2_round_trips_through_the_inference_codec():
    """encode -> Packet -> decode, for a multi-resolution source.

    NOT yet charged: S2Normalizer removes a per-tile per-band mean/std (26
    floats) that no packet carries, so `packet_bits` here is the index cost
    only. docs/ADDING_A_SOURCE.md is blunt about why that matters: an
    uncharged stats channel once reached 34% of a bitstream unnoticed.
    """
    from georvq.inference.api import Codec

    cfg = _cfg()
    torch.manual_seed(0)
    model = build_model(cfg)
    model.eval()
    src = model.source
    raw = {"10m": torch.randint(1, 9000, (2, 4, 24, 24), dtype=torch.int32),
           "20m": torch.randint(1, 9000, (2, 6, 12, 12), dtype=torch.int32),
           "60m": torch.randint(1, 9000, (2, 3, 4, 4), dtype=torch.int32)}
    x, valid = src.prep(raw, "cpu")

    codec = Codec(model)
    packets = codec.encode(x, valid=valid)
    assert len(packets) == 2
    assert [tuple(i.shape) for i in packets[0].indices] == [(1, 3), (16, 3)]
    out = codec.decode(packets)
    assert out["10m"].shape == (2, 4, 24, 24)
    assert out["20m"].shape == (2, 6, 12, 12)
    assert out["60m"].shape == (2, 3, 4, 4)
    assert codec.packet_bits() > 0


def test_the_normalizer_channel_is_charged_to_the_rate():
    """Regression for a silently understated rate.

    `side_bits` used to charge only `model.stats_cond`, so a source that
    normalizes instead -- S2 strips 26 per-band mean/std floats per window
    -- had that channel counted nowhere. On this pyramid the omission is
    larger than the entire coarse-only operating point, which is how a
    stats channel reaches a third of a bitstream unnoticed.
    """
    from georvq.model.codec import tuple_bits

    cfg = _cfg()
    model = build_model(cfg)
    sb = model.side_bits()
    # no basis configured, so the cheapest available encoding is raw f16
    # (1-byte code + 26 values); the point of the regression is that it is
    # charged at all, not which encoding wins
    assert sb >= 26 * 16.0
    # and it actually reaches the rate figures
    mins = [1, 0]
    assert tuple_bits(cfg.model.tiers, mins, sb) == \
        tuple_bits(cfg.model.tiers, mins) + sb
    # the point of the regression: the channel is not a rounding error
    # next to the cheapest operating point
    assert sb > tuple_bits(cfg.model.tiers, mins)


def test_sources_without_a_normalizer_charge_nothing():
    """AE and Tessera must be untouched by the fix."""
    from georvq.sources.ae import AESource
    from georvq.sources.tessera import TesseraSource

    cfg = _cfg()
    assert AESource(cfg.model).side_bits(cfg.model) == 0.0
    assert TesseraSource(cfg.model).side_bits(cfg.model) == 0.0


def test_stats_pca_basis_shrinks_the_charged_channel(tmp_path):
    """The fitted basis must actually reduce the charged rate, and the
    charge must come from the artifact rather than a hardcoded guess."""
    import numpy as np

    from georvq.inference.stats_codec import StatsPacker, fit_stats_pca

    rng = np.random.default_rng(0)
    # stats vectors are strongly correlated across bands, which is the
    # structure the basis exists to exploit
    latent = rng.normal(size=(512, 4))
    mixing = rng.normal(size=(4, 26))
    stats = latent @ mixing * 0.05 + 0.2

    basis = dict(fit_stats_pca(stats))
    packer = StatsPacker(data=basis, tolerance=0.01)
    sizes = [len(packer.pack(s)) for s in stats]
    assert np.mean(sizes) < 26 * 2          # beats raw f16
    for s in stats[:32]:                    # and stays inside tolerance
        assert np.abs(packer.unpack(packer.pack(s)) - s).max() < 0.011

    basis["mean_bytes"] = np.float32(np.mean(sizes))
    path = tmp_path / "pca.npz"
    np.savez(path, **basis)

    raw_cfg = _cfg()
    model_raw = build_model(raw_cfg)

    cfg = _cfg()
    cfg.model.stats_pca = path
    model = build_model(cfg)
    # at a rung too cheap to afford a raw escape, the fitted basis is what
    # gets charged -- and its cost comes from the artifact's measurement
    cheap = [1, 0]
    assert model.side_bits(cheap) == pytest.approx(np.mean(sizes) * 8)
    assert model.side_bits(cheap) < model_raw.side_bits(cheap)


def test_a_basis_without_a_measured_size_is_rejected(tmp_path):
    """A basis fitted by hand, with no recorded packed size, must fail
    loudly rather than have the rate silently invented."""
    import numpy as np

    from georvq.inference.stats_codec import fit_stats_pca

    rng = np.random.default_rng(0)
    basis = fit_stats_pca(rng.normal(size=(64, 26)))
    path = tmp_path / "nosize.npz"
    np.savez(path, **basis)
    cfg = _cfg()
    cfg.model.stats_pca = path
    model = build_model(cfg)
    with pytest.raises(KeyError, match="mean_bytes"):
        model.side_bits()


def test_stats_precision_follows_the_rate():
    """The right stats encoding depends on the rung, because the two costs
    move in opposite directions: tolerance-coded PCA is ~78 bits but
    leaves a reflectance floor, while sending the vector verbatim removes
    the floor and costs 424 (f16) or 840 (f32) bits. At the cheapest rungs
    78 bits is already most of the bitstream; at full depth the floor
    equals the entire model error while f32 is ~1.6% of the rate.
    """
    from georvq.model.codec import tuple_bits

    from georvq.config import load_config

    cfg = load_config("configs/s2/s2_pyr96.yaml")
    model = build_model(cfg)
    src, mc = model.source, cfg.model

    cheap, full = [4, 0, 0], [sp.depth for sp in mc.tiers]
    assert src.side_bits(mc, cheap) < src.side_bits(mc, full)
    # never more than the budget of a rung's index bits, once affordable
    for ls in (cheap, full):
        sb = src.side_bits(mc, ls)
        idx = tuple_bits(mc.tiers, ls)
        cheapest = src._stats_encodings(mc)[0][1]
        assert sb <= max(idx * src.STATS_BUDGET, cheapest)
    # depths=None means "a whole packet", i.e. full depth
    assert model.side_bits() == src.side_bits(mc, full)
    # at full depth the rate is large enough to send the vector verbatim
    # (f16, which measurement shows reproduces exact stats), while the
    # cheapest rung still packs
    assert src.side_bits(mc, full) == 26 * 16 + 8
    assert src.stats_tolerance(mc, full) < 1e-3
    assert src.stats_tolerance(mc, cheap) == 0.01
    # no f32 tier: measured to buy nothing over f16
    assert all(b <= 26 * 16 + 8 for _, b in src._stats_encodings(mc))


def test_raw_stats_escape_round_trips_exactly_at_f32():
    """f32 is the point of the escape: each entry is a mean or std over a
    whole window, known far more precisely than any single uint16 DN, so
    rounding it to the source quantum would add error the data lacks."""
    import numpy as np

    from georvq.inference.stats_codec import StatsPacker, fit_stats_pca

    rng = np.random.default_rng(0)
    stats = rng.normal(size=(64, 26)) * 0.05 + 0.2
    packer = StatsPacker(data=fit_stats_pca(stats), tolerance=0.01)

    s = stats[0].astype(np.float32)
    exact = packer.pack(s, tolerance=0.0)
    assert exact[0] == 254 and len(exact) == 1 + 4 * 26
    assert np.array_equal(packer.unpack(exact), s)

    # the f16 escape is lossy but far below the PCA tolerance
    f16 = packer.pack_raw(s, np.float16)
    assert f16[0] == 255 and len(f16) == 1 + 2 * 26
    assert np.abs(packer.unpack(f16) - s).max() < 1e-3

    # and the default tolerance path is still the cheap one
    assert len(packer.pack(s)) < len(f16)


# ------------------------------------------------------ nodata masking

def _raw_windows(rng, B=3, W=24):
    return {g: torch.from_numpy(rng.integers(500, 9000, size=(B, b, W // r, W // r),
                                             dtype=np.uint16))
            for g, (b, r) in {"10m": (4, 1), "20m": (6, 2),
                              "60m": (3, 6)}.items()}


def _s2_source():
    from georvq.sources.s2 import S2Source
    return S2Source(_cfg().model)


def test_s2_stats_unchanged_on_fully_valid_windows():
    """With no nodata the masked stats equal the unmasked mean and the
    unbiased std, so existing checkpoints see the same inputs."""
    src = _s2_source()
    raw = _raw_windows(np.random.default_rng(1))
    src.prep(raw, "cpu")
    st = src.last_stats()
    for g, t in raw.items():
        f = t.float() / 10000.0
        assert torch.allclose(st[g][0], f.mean((2, 3), keepdim=True), atol=1e-6)
        assert torch.allclose(st[g][1], f.std((2, 3), keepdim=True), atol=1e-6)


def test_s2_stats_exclude_nodata():
    """A window straddling a swath edge: the stats are those of its valid
    pixels, the fill is 0 in normalized space, and the mask covers it."""
    src = _s2_source()
    raw = _raw_windows(np.random.default_rng(2))
    # left half of window 0 is nodata in every band, at every resolution
    for g, r in (("10m", 1), ("20m", 2), ("60m", 6)):
        raw[g][0, :, :, :12 // r] = 0
    # one 20m band has a slightly wider nodata edge than the others
    raw["20m"][0, 2, :, 6] = 0
    x, valid = src.prep(raw, "cpu")
    st = src.last_stats()
    for g, r in (("10m", 1), ("20m", 2), ("60m", 6)):
        f = raw[g][0].float() / 10000.0
        for b in range(f.shape[0]):
            v = f[b][raw[g][0, b] != 0]
            assert torch.allclose(st[g][0][0, b, 0, 0], v.mean(), atol=1e-6)
            assert torch.allclose(st[g][1][0, b, 0, 0], v.std(), atol=1e-6)
        assert (x[g][0][raw[g][0] == 0] == 0).all()
    assert not valid[0, :, :12].any()
    assert not valid[0, :, 12:14].any()      # the wider 20m edge
    assert valid[0, :, 14:].all() and valid[1:].all()


def test_s2_loss_ignores_nodata_pixels():
    src = _s2_source()
    raw = _raw_windows(np.random.default_rng(3))
    for g, r in (("10m", 1), ("20m", 2), ("60m", 6)):
        raw[g][0, :, :, :12 // r] = 0
    x, valid = src.prep(raw, "cpu")
    out = {g: t + 0.1 for g, t in x.items()}
    loss_a, _ = src.recon_stats(out, x, valid)
    out_b = {g: t.clone() for g, t in out.items()}
    for g, r in (("10m", 1), ("20m", 2), ("60m", 6)):
        out_b[g][0, :, :, :12 // r] = 50.0   # garbage where there is no data
    loss_b, _ = src.recon_stats(out_b, x, valid)
    assert torch.allclose(loss_a, loss_b)
    assert torch.allclose(loss_a, torch.tensor(0.01), atol=1e-5)
    from georvq.sources.s2 import _s2_r2
    assert torch.allclose(_s2_r2(out, x, valid), _s2_r2(out_b, x, valid))


def test_s2_all_nodata_window_is_finite():
    src = _s2_source()
    raw = {g: torch.zeros_like(t) for g, t in
           _raw_windows(np.random.default_rng(4)).items()}
    x, valid = src.prep(raw, "cpu")
    assert not valid.any()
    for g in x:
        assert torch.isfinite(x[g]).all()
    loss, m = src.recon_stats(x, x, valid)
    assert torch.isfinite(loss) and torch.isfinite(m["one_minus_mse_10m"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_groupbatch_tensor_index_on_cuda_uint16():
    """The reconstruction figure gathers val windows with a tensor index;
    CUDA has no index kernel for uint16, the dtype S2 val tokens hold."""
    from georvq.data.pool import GroupBatch
    g = GroupBatch({"10m": torch.arange(60000, 60024, dtype=torch.int32)
                    .to(torch.uint16).view(6, 4).cuda()})
    sel = torch.tensor([0, 5, 2])
    out = g[sel]["10m"].cpu()
    assert out.dtype == torch.uint16
    assert torch.equal(out, g["10m"].cpu()[sel])


def test_groupbatch_tensor_index_on_cpu_uint16():
    from georvq.data.pool import GroupBatch
    t = torch.arange(60000, 60024, dtype=torch.int32).to(torch.uint16).view(6, 4)
    g = GroupBatch({"10m": t})
    sel = torch.tensor([0, 5, 2])
    assert torch.equal(g[sel]["10m"], t[sel])


# ------------------------------------------------------ the release preset

def test_s2_release_preset_tiles_l1c_and_keeps_its_sample_budget():
    """configs/s2/beta_arch_w36.yaml: a 36 px window divides a 10980 px L1C tile
    exactly and its fine cell is one 60 m pixel; both tiers keep at least
    ~450 vectors per code at their deepest level under path sampling,
    with the coarse tier always sent in full. The coarse book gets 437 at
    batch 7168 and K = 8192, which the preset accepts as close enough, so
    the bound here is 430."""
    from georvq.config import load_config
    from georvq.model.codec import path_tuples
    cfg = load_config("configs/s2/beta_arch_w36.yaml")
    m = cfg.model
    assert 10980 % m.window == 0 and m.window % 6 == 0
    assert [t.grid for t in m.tiers] == [1, 6]
    assert m.window // m.tiers[-1].grid == 6
    assert m.min_depths[0] == m.tiers[0].depth
    assert m.stats_pca is None
    assert m.vq.init_mode == "tree" and m.vq.reset_mode == "split"
    path = path_tuples(m.tiers, m.min_depths)
    p_full = m.full_frac / (m.full_frac + 0.70)
    window = 1.0 / (1.0 - m.vq.ema_decay)
    for ti, sp in enumerate(m.tiers):
        reach = np.mean([t[ti] >= sp.depth for t in path])
        share = p_full + (1 - p_full) * reach
        per_code = cfg.train.batch_windows * sp.grid ** 2 * window * share
        assert per_code / sp.codes >= 430, (ti, per_code / sp.codes)
    torch.manual_seed(0)
    model = build_model(cfg)
    assert model.side_bits() == 16 * 26 + 8


def test_error_histogram_has_a_row_per_native_grid(s2_store):
    """S2's decoder emits three grids, so the signed-error histogram is
    split by grid, in standardized units and in DN."""
    import matplotlib
    matplotlib.use("Agg")
    from georvq.data.pool import fixed_val_tokens
    from georvq.train.figures import error_hist_harvest, error_hist_render

    cfg = _cfg()
    model = build_model(cfg)
    assert model.source.report(cfg.model).has_figure("error_hist")
    val = fixed_val_tokens(s2_store, s2_store.index, token=24, per_chip=2,
                           nodata=None)
    h = error_hist_harvest(model, model.source, val, tuples=[[3, 0], [3, 3]],
                           chunk=4)
    assert set(h["rel"]) == {f"{t}|{g}" for t in ("3-0", "3-3")
                             for g in ("10m", "20m", "60m")}
    assert set(h["absu"]) == set(h["rel"])
    # a 60 m error in DN is the standardized error times the band's std
    # in DN, so the two spaces differ by far more than rounding
    assert h["absu"]["3-0|60m"].abs().mean() > 10 * h["rel"]["3-0|60m"].abs().mean()
    fig = error_hist_render(h, step=0, unit="DN")
    assert len(fig.axes) == 6
