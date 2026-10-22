"""End-to-end training smokes on a synthetic store (CPU)."""
import os

import pytest
import torch

from georvq.config import load_config
from georvq.data.build import build_synthetic_store


@pytest.fixture
def smoke_cfg(tmp_path):
    build_synthetic_store(tmp_path / "store", n_chips=8, bands=8, chip_px=64,
                          shard_size=4)
    return [f"data.store_dir={tmp_path / 'store'}",
            f"log.out_dir={tmp_path / 'ckpt'}"]


def test_pyramid_smoke(smoke_cfg, tmp_path):
    from georvq import from_checkpoint
    from georvq.train.loop import train
    cfg = load_config("configs/smoke.yaml", smoke_cfg)
    run_dir = train(cfg, gpu=-1)
    assert os.path.exists(os.path.join(run_dir, "final_val.json"))
    model = from_checkpoint(os.path.join(run_dir, "model.pt"))
    assert model.full_config.train.steps == 30
    x = torch.nn.functional.normalize(torch.randn(2, 8, 16, 16), dim=1)
    with torch.no_grad():
        _, cums = model.level_cumsums(x)
        out = model.eval_tuple(cums, [3, 2], x)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_pyramid_smoke_rate_and_warmup(smoke_cfg):
    from georvq.train.loop import train
    cfg = load_config(
        "configs/smoke.yaml",
        smoke_cfg + ["model.rate={model_dim: 32, blocks: 1, ctx_dim: 8, batch: 4}",
                     "model.vq.warmup_steps=10",
                     "model.vq.grad_codebook=true",
                     "model.vq.kmeans_sample=32",
                     "train.steps=20"])
    run_dir = train(cfg, gpu=-1)
    assert os.path.exists(os.path.join(run_dir, "model.pt"))


def test_stage1_smoke(smoke_cfg):
    from georvq.train.stage1 import train
    cfg = load_config(
        "configs/smoke.yaml",
        smoke_cfg + ["model.vq=null", "model.sigma={max: 2.0, freqs: 4}",
                     "train.steps=20", "train.val_interval=10",
                     "log.run_tag=smoke_s1"])
    run_dir = train(cfg, gpu=-1)
    assert os.path.exists(os.path.join(run_dir, "model.pt"))


def test_decoder_only_changes_only_the_decoder(smoke_cfg):
    """train.decoder_only: after a few steps, with a validation in between
    (which returns the model to train mode), every encoder and codebook
    tensor, buffers included, is bit-identical to the checkpoint it
    started from, and the decoder has moved."""
    from georvq.repro import load_checkpoint
    from georvq.train.loop import train
    src_dir = train(load_config("configs/smoke.yaml", smoke_cfg), gpu=-1)
    src = os.path.join(src_dir, "model.pt")
    cfg = load_config("configs/smoke.yaml", smoke_cfg + [
        f"train.init_from={src}", "train.decoder_only=true",
        "model.vq.warmup_steps=0", "train.steps=6", "train.val_interval=3",
        "train.save_interval=6", "log.run_tag=smoke_dec"])
    dec_dir = train(cfg, gpu=-1)
    a = load_checkpoint(src)["model"]
    b = load_checkpoint(os.path.join(dec_dir, "model.pt"))["model"]
    assert a.keys() == b.keys()
    moved = [k for k in a if not torch.equal(a[k], b[k])]
    assert moved and all(k.startswith("decoder.") for k in moved), moved
    assert any(k.startswith("rvqs.") for k in a)
