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
