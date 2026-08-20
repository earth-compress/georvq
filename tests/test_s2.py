"""Sentinel-2 adapter: multi-group store, loader, native-res stems and
multi-res heads through the pyramid, toy optimization."""
import numpy as np
import pytest
import torch

from georvq.config import Config
from georvq.data.build import ShardWriter
from georvq.data.pool import GroupChipLoader
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


def test_group_loader(s2_store):
    loader = GroupChipLoader(s2_store, s2_store.index,
                             torch.device("cpu"), seed=0)
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
    loader = GroupChipLoader(s2_store, s2_store.index, torch.device("cpu"))
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


@pytest.mark.slow
def test_s2_toy_training(s2_store):
    """A short optimization drives the multi-res reconstruction loss down."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = build_model(cfg)
    source = model.source
    loader = GroupChipLoader(s2_store, s2_store.index, torch.device("cpu"))
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
