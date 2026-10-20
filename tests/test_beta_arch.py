"""The ring-space beta_v4 architecture in georvq (configs/s2/beta_arch.yaml):
the source-built conv encoder and the reference curves drawn under a
run's validation."""
import json
import os

import numpy as np
import pytest
import torch

from georvq.config import Config, deep_merge, load_config
from georvq.data.build import ShardWriter
from georvq.factory import build_model

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


def test_preset_has_beta_v4_shapes():
    """beta_v2's encoder and books (41.8M with its 1.20M conv decoder)
    and beta_v4's 22.5M transformer decoder: 63.06M in all."""
    m = build_model(load_config("configs/s2/beta_arch.yaml"))
    n = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
    assert abs(n(m) - 63.06) < 0.01
    assert abs(n(m.decoder) - 22.47) < 0.01
    assert abs(n(m.rvqs) - 33.55) < 0.01
    x = {"10m": torch.randn(2, 4, 42, 42), "20m": torch.randn(2, 6, 21, 21),
         "60m": torch.randn(2, 3, 7, 7)}
    zs = m.encode_levels(x)
    assert [tuple(z.shape) for z in zs] == [(2, 1, 128), (2, 49, 128)]


def test_conv_encoder_needs_one_cell_per_60m_pixel():
    with pytest.raises(ValueError, match="window/6"):
        build_model(Config.model_validate({
            "data": {"source": "s2", "store_dir": "/tmp/unused"},
            "model": {"window": 24, "bands": 13,
                      "tiers": ["1:8:16:2", "2:8:16:2"],
                      "tokenizer": {"kind": "source", "dim": 8},
                      "decoder": {"dim": 16, "depth": 1, "heads": 2}}}))


def test_sources_without_an_encoder_say_so():
    with pytest.raises(NotImplementedError, match="tokenizer.kind: vit"):
        build_model(Config.model_validate({
            "data": {"source": "ae", "store_dir": "/tmp/unused"},
            "model": {"window": 16, "bands": 8,
                      "tiers": ["1:8:16:2", "4:8:16:2"],
                      "tokenizer": {"kind": "source", "dim": 8},
                      "decoder": {"dim": 16, "depth": 1, "heads": 2}}}))


def test_ref_curve_round_trip(tmp_path):
    from georvq.train.validate import load_ref_curve, write_ref_curve
    p = tmp_path / "r.json"
    write_ref_curve(p, "beta_v4 (ring-space, no deblock)", "r2",
                    {"8-1": 0.8, "8-8": 0.94}, scalars={"r2_ceiling": 0.95})
    rc = load_ref_curve(p)
    assert rc["slug"] == "beta_v4_ring_space_no_deblock"
    assert rc["quality_by_tuple"]["8-8"] == 0.94
    assert load_ref_curve(p, "other")["label"] == "other"
    # the preset's reference resolves from the repository root
    assert load_ref_curve("artifacts/s2_ref_beta_v4.json")["quality_key"] \
        == "r2"


def test_training_draws_the_reference(s2_store, tmp_path):
    """A run with a reference curve scores the reference's tuples at
    every validation and writes the history figure."""
    from georvq.train.loop import train
    from georvq.train.validate import write_ref_curve
    ref = tmp_path / "ref.json"
    write_ref_curve(ref, "beta (test)", "r2", {"3-1": 0.5, "3-2": 0.7},
                    scalars={"r2_ceiling": 0.8})
    cfg = Config.model_validate({
        "data": {"source": "s2", "store_dir": str(s2_store), "val_frac": 0.0,
                 "pool": {"size": 6, "workers": 0, "device": "cpu"}},
        "model": {"window": 24, "bands": 13,
                  "tiers": ["1:16:32:3", "4:16:32:3"],
                  "min_depths": [3, 0], "depth_sampling": "path",
                  "tokenizer": {"kind": "source", "dim": 16},
                  "decoder": {"dim": 32, "depth": 1, "heads": 4,
                              "head_feat": 8, "cls_bridge": False},
                  "vq": {"warmup_steps": 3, "init_mode": "tree",
                         "init_per_code": 4,
                         "reset_mode": "split"}},
        "train": {"steps": 8, "batch_windows": 8, "amp": False,
                  "val_interval": 4, "fig_interval": 4, "save_interval": 4},
        "eval": {"ref_curves": [{"path": str(ref)}]},
        "log": {"run_tag": "b", "wandb": False, "out_dir": str(tmp_path)},
    })
    run = train(cfg, gpu=-1)
    assert os.path.exists(os.path.join(run, "val_vs_ref.png"))
    blob = json.load(open(os.path.join(run, "final_val.json")))
    assert {"3-1", "3-2"} <= set(blob["quality_by_tuple"])


def test_figures_draw_reference_curves():
    import matplotlib.pyplot as plt
    from georvq.sources import get_source
    from georvq.sources.report import resolve_report
    from georvq.train.figures import ref_history_figure, rd_figure
    cfg = load_config("configs/s2/beta_arch.yaml")
    spec = resolve_report(get_source("s2", cfg.model), cfg)
    rc = {"label": "ref", "slug": "ref", "quality_key": "r2",
          "quality_by_tuple": {"8-1": 0.8, "8-8": 0.94},
          "scalars": {"r2_ceiling": 0.95}, "bpp_by_tuple": None}
    hist = {"step": [1, 2], "ceiling": [0.9, 0.93],
            "8-1": [0.7, 0.75], "8-8": [0.9, 0.92]}
    f = ref_history_figure(hist, [rc], spec.quality)
    assert len([a for a in f.axes if a.get_title()]) == 3
    plt.close(f)
    ts = [[8, 1], [8, 8]]
    f = rd_figure({"8-1": 0.75, "8-8": 0.92}, {"r2_ceiling": 0.93},
                  cfg.model.tiers, ts, 2, spec, ref_curves=[rc], px=42 ** 2,
                  side_bits=424.0)
    labels = [l.get_label() for l in f.axes[0].get_lines()]
    assert "ref" in labels
    plt.close(f)


def _three_tier_cfg(tiers, window=24, **model):
    return Config.model_validate({
        "data": {"source": "s2", "store_dir": "/tmp/unused"},
        "model": {"window": window, "bands": 13, "tiers": tiers,
                  "tokenizer": {"kind": "source", "dim": 8},
                  "decoder": {"dim": 16, "depth": 1, "heads": 2,
                              "head_feat": 8, "cls_bridge": False},
                  **model}})


def test_w60_preset_has_three_nested_tiers():
    """s2_w60: one whole-window cell, 5 x 5 cells of 12 px and 10 x 10
    cells of 6 px, all 128 wide."""
    m = build_model(load_config("configs/s2/s2_w60.yaml"))
    x = {"10m": torch.randn(2, 4, 60, 60), "20m": torch.randn(2, 6, 30, 30),
         "60m": torch.randn(2, 3, 10, 10)}
    zs = m.encode_levels(x)
    assert [tuple(z.shape) for z in zs] == \
        [(2, 1, 128), (2, 25, 128), (2, 100, 128)]


def test_two_tier_encoder_has_only_beta_parameters():
    """The middle branch exists only when there is a middle tier, so
    beta_v2's weights still load strictly into a two-tier encoder."""
    m = build_model(load_config("configs/s2/beta_arch.yaml"))
    assert not any("mid_" in k for k in m.state_dict())
    m3 = build_model(_three_tier_cfg(["1:8:16:2", "2:8:16:2", "4:8:16:2"]))
    assert any(k.startswith("tokenizer.mid_down") for k in m3.state_dict())


@pytest.mark.parametrize("tiers,window", [
    (["1:8:16:2", "3:8:16:2", "10:8:16:2"], 60),    # 3 does not divide 10
    (["1:8:16:2", "2:8:16:2", "5:8:16:2", "10:8:16:2"], 60),   # four tiers
    (["1:8:16:2", "2:8:16:2", "5:8:16:2"], 60),     # fine cell is not 6 px
])
def test_conv_encoder_rejects_bad_three_tier_layouts(tiers, window):
    with pytest.raises(ValueError, match="g_mid dividing window/6"):
        build_model(_three_tier_cfg(tiers, window=window))


def test_mid_token_reads_only_its_own_block():
    """Each middle token is computed from its own 2 x 2 block of the 60 m
    feature map (12 x 12 px at 10 m), so changing one fine cell's features
    moves the middle token over it and no other, whilst the pooled coarse
    vector sees every cell. The conv trunk before the feature map reaches
    across cell edges as beta_v2's does, so this is a property of the
    branch, not of the whole encoder."""
    m = build_model(_three_tier_cfg(["1:8:16:2", "2:8:16:2", "4:8:16:2"]))
    enc = m.tokenizer.eval()
    torch.manual_seed(0)
    h = torch.randn(3, 32, 4, 4)
    with torch.no_grad():
        base = enc.mid(h)
        assert base.shape == (3, 4, 8)
        for r in range(4):
            for c in range(4):
                h2 = h.clone()
                h2[:, :, r, c] += 1.0
                moved = (enc.mid(h2) - base).abs().amax((0, 2)) > 0
                own = (r // 2) * 2 + c // 2
                assert moved.nonzero().flatten().tolist() == [own]
    x = {"10m": torch.randn(1, 4, 24, 24), "20m": torch.randn(1, 6, 12, 12),
         "60m": torch.randn(1, 3, 4, 4)}
    with torch.no_grad():
        h = enc.trunk(x)
        coarse, mid, fine = enc(x)
        assert torch.equal(mid, enc.mid(h))
        assert torch.equal(fine, enc.hq_proj(h).flatten(2).transpose(1, 2))
        x2 = {k: v.clone() for k, v in x.items()}
        x2["60m"][:, :, 3, 3] += 1.0       # one corner 60 m pixel
        c2, m2, _ = enc(x2)
        assert not torch.equal(c2, coarse)
        assert torch.equal(m2[:, 0], mid[:, 0])    # opposite corner block


def test_three_tier_conv_encoder_trains(s2_store, tmp_path):
    """Warmup with a quarter batch, tree init at full batch, then path
    sampled quantised steps over all three tiers."""
    from georvq.train.loop import train
    cfg = Config.model_validate({
        "data": {"source": "s2", "store_dir": str(s2_store), "val_frac": 0.0,
                 "pool": {"size": 6, "workers": 0, "device": "cpu"}},
        "model": {"window": 24, "bands": 13,
                  "tiers": ["1:16:16:3", "2:16:32:3", "4:16:32:3"],
                  "min_depths": [3, 0, 0], "depth_sampling": "path",
                  "full_frac": 0.05,
                  "tokenizer": {"kind": "source", "dim": 16},
                  "decoder": {"dim": 32, "depth": 1, "heads": 4,
                              "head_feat": 8, "cls_bridge": False},
                  "vq": {"warmup_steps": 4, "init_mode": "tree",
                         "init_per_code": 4, "reset_mode": "split",
                         "clip_norm": 0.25}},
        "train": {"steps": 8, "batch_windows": 8,
                  "batch_ramp": {"until": 2, "divisor": 4}, "amp": False,
                  "val_interval": 8, "fig_interval": 8, "save_interval": 8},
        "eval": {"recon_tuples": ["3-0-0", "3-3-1", "3-3-3"]},
        "log": {"run_tag": "w3", "wandb": False, "out_dir": str(tmp_path)},
    })
    run = train(cfg, gpu=-1)
    blob = json.load(open(os.path.join(run, "final_val.json")))
    assert {"3-0-0", "3-3-1", "3-3-3"} <= set(blob["quality_by_tuple"])
