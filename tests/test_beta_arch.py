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
