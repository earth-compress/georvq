"""Stop signals, resume, atomic checkpoints and the non-finite step skip."""
import os
import signal

import pytest
import torch

from georvq.config import load_config
from georvq.data.build import build_synthetic_store
from georvq.train.signals import STOP_SIGNALS, stop_on_signals


@pytest.mark.parametrize("sig", [signal.SIGUSR1, signal.SIGTERM])
def test_stop_signal_sets_the_flag_and_restores_handlers(sig):
    before = signal.getsignal(sig)
    with stop_on_signals() as stop:
        assert not stop
        os.kill(os.getpid(), sig)
        assert stop and stop.signum == sig
    assert signal.getsignal(sig) == before
    assert set(STOP_SIGNALS) == {signal.SIGUSR1, signal.SIGTERM}


def test_checkpoint_write_is_atomic(tiny_raw_cfg, tmp_path, monkeypatch):
    """A write that dies part-way leaves the previous checkpoint intact."""
    from georvq.config import Config
    from georvq.factory import build_model
    from georvq.repro import save_checkpoint
    cfg = Config.model_validate(tiny_raw_cfg)
    m = build_model(cfg, torch.device("cpu"))
    path = tmp_path / "model.pt"
    save_checkpoint(path, m, cfg, step=1)
    real_save = torch.save

    def dying_save(obj, f, *a, **kw):
        with open(f, "wb") as fh:
            fh.write(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", dying_save)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(path, m, cfg, step=2)
    monkeypatch.setattr(torch, "save", real_save)
    assert torch.load(path, weights_only=False)["step"] == 1


@pytest.fixture
def smoke(tmp_path):
    build_synthetic_store(tmp_path / "store", n_chips=8, bands=8, chip_px=64,
                          shard_size=4)
    return lambda out: [f"data.store_dir={tmp_path / 'store'}",
                        f"log.out_dir={tmp_path / out}"]


def _interrupt_at(monkeypatch, at, sig=signal.SIGUSR1):
    """Send `sig` to this process during training step `at`."""
    from georvq.train import loop
    real = loop.mark_resets

    def mark(model, step, first_reset):
        if step == at:
            os.kill(os.getpid(), sig)
        return real(model, step, first_reset)
    monkeypatch.setattr(loop, "mark_resets", mark)
    return real


@pytest.mark.slow
def test_resume_continues_the_run(smoke, monkeypatch):
    """Stopped by SIGUSR1 at step 20 and resumed, the run reaches the
    same step-30 state as one that ran straight through."""
    from georvq.train import loop
    full = loop.train(load_config("configs/smoke.yaml", smoke("a")), gpu=-1)

    real = _interrupt_at(monkeypatch, 20)
    with pytest.raises(SystemExit) as e:
        loop.train(load_config("configs/smoke.yaml", smoke("b")), gpu=-1)
    assert e.value.code == 0
    monkeypatch.setattr(loop, "mark_resets", real)
    run_b = os.path.join(smoke("b")[1].split("=", 1)[1], "smoke")
    mid = torch.load(os.path.join(run_b, "model.pt"), weights_only=False)
    assert mid["step"] == 20 and "opt" in mid and "train_state" in mid
    assert not os.path.exists(os.path.join(run_b, "final_val.json"))

    cfg = load_config("configs/smoke.yaml",
                      smoke("b") + [f"train.resume={run_b}/model.pt"])
    resumed = loop.train(cfg, gpu=-1)
    a = torch.load(os.path.join(full, "model.pt"), weights_only=False)
    b = torch.load(os.path.join(resumed, "model.pt"), weights_only=False)
    assert a["step"] == b["step"] == 30
    assert a["sched"]["last_epoch"] == b["sched"]["last_epoch"] == 30
    assert a["sched"]["_last_lr"] == pytest.approx(b["sched"]["_last_lr"])
    assert a["metrics"] == pytest.approx(b["metrics"])
    assert a["train_state"]["best_q"] == pytest.approx(
        b["train_state"]["best_q"])
    for k, t in a["model"].items():
        torch.testing.assert_close(b["model"][k], t, rtol=1e-5, atol=1e-6,
                                   msg=k)


@pytest.mark.slow
def test_sigterm_saves_and_exits_zero(smoke, monkeypatch):
    from georvq.train import loop
    _interrupt_at(monkeypatch, 12, signal.SIGTERM)
    with pytest.raises(SystemExit) as e:
        loop.train(load_config("configs/smoke.yaml",
                               smoke("t") + ["train.save_optim=false"]),
                   gpu=-1)
    assert e.value.code == 0
    run = os.path.join(smoke("t")[1].split("=", 1)[1], "smoke")
    ck = torch.load(os.path.join(run, "model.pt"), weights_only=False)
    # the stop save carries optimizer state even with save_optim off
    assert ck["step"] == 12 and "opt" in ck and "sched" in ck


@pytest.mark.slow
def test_resume_refuses_a_weights_only_checkpoint(smoke):
    from georvq.train import loop
    run = loop.train(load_config("configs/smoke.yaml", smoke("w")
                                 + ["train.steps=10",
                                    "train.save_interval=10"]), gpu=-1)
    cfg = load_config("configs/smoke.yaml",
                      smoke("w") + [f"train.resume={run}/model_best.pt"])
    with pytest.raises(ValueError, match="resume"):
        loop.train(cfg, gpu=-1)


@pytest.mark.slow
def test_nonfinite_final_step_still_validates_and_saves(smoke, monkeypatch):
    """A skipped update on the last step must not skip the final
    validation, the final model.pt or final_val.json."""
    from georvq.train import loop
    real = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def clip(params, max_norm, *a, **kw):
        calls["n"] += 1
        out = real(params, max_norm, *a, **kw)
        return torch.tensor(float("inf")) if calls["n"] == 30 else out
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    run = loop.train(load_config("configs/smoke.yaml", smoke("n")
                                 + ["model.vq.clip_norm=1.0"]), gpu=-1)
    ck = torch.load(os.path.join(run, "model.pt"), weights_only=False)
    assert ck["step"] == 30
    assert ck["train_state"]["nonfinite_steps"] == 1
    assert ck["sched"]["last_epoch"] == 30
    import json
    with open(os.path.join(run, "final_val.json")) as f:
        blob = json.load(f)
    assert blob["quality_by_tuple"]


@pytest.mark.slow
def test_stage1_signal_saves_and_exits_zero(smoke, monkeypatch):
    from georvq.train import stage1
    real = stage1.sample_sigmas
    calls = {"n": 0}

    def sig(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 7:
            os.kill(os.getpid(), signal.SIGUSR1)
        return real(*a, **kw)
    monkeypatch.setattr(stage1, "sample_sigmas", sig)
    cfg = load_config("configs/smoke.yaml", smoke("s1") + [
        "model.vq=null", "model.sigma={max: 2.0, freqs: 4}",
        "train.steps=20", "train.val_interval=10", "train.optim.warmup=0",
        "log.run_tag=smoke_s1"])
    with pytest.raises(SystemExit) as e:
        stage1.train(cfg, gpu=-1)
    assert e.value.code == 0
    run = os.path.join(smoke("s1")[1].split("=", 1)[1], "smoke_s1")
    ck = torch.load(os.path.join(run, "model.pt"), weights_only=False)
    assert ck["step"] == 7 and "opt" in ck


def test_stage1_presets_pass_and_unsupported_knobs_are_refused():
    import glob

    from georvq.train.loop import make_lr_fn
    from georvq.train.stage1 import check_stage1_config
    for f in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
        cfg = load_config(f)
        if cfg.model.sigma is not None:
            check_stage1_config(cfg)
    cfg = load_config("configs/ae/stage1.yaml",
                      ["train.batch_ramp={until: 10, divisor: 2}"])
    with pytest.raises(ValueError, match="batch_ramp"):
        check_stage1_config(cfg)
    # the shared schedule takes warmup 0 and the constant schedule
    cfg = load_config("configs/ae/stage1.yaml",
                      ["train.optim.warmup=0", "train.optim.schedule=constant"])
    f, s0 = make_lr_fn(cfg), cfg.train.optim.lr0_steps
    assert f(s0) == 1.0 and f(cfg.train.steps - 1) == 1.0


@pytest.mark.slow
def test_loop_checkpoints_load_weights_only(smoke, monkeypatch):
    """Every checkpoint the loop writes, including the resume state,
    loads under torch.load(weights_only=True) with only TorchVersion
    allowlisted (a subset of georvq.repro.load_checkpoint's list)."""
    from torch.torch_version import TorchVersion

    from georvq.train import loop
    _interrupt_at(monkeypatch, 15)
    with pytest.raises(SystemExit):
        loop.train(load_config("configs/smoke.yaml", smoke("wo")), gpu=-1)
    run = os.path.join(smoke("wo")[1].split("=", 1)[1], "smoke")
    for name in ("model.pt", "model_best.pt"):
        with torch.serialization.safe_globals([TorchVersion]):
            ck = torch.load(os.path.join(run, name), weights_only=True)
        assert ck["step"] >= 10
    with torch.serialization.safe_globals([TorchVersion]):
        ck = torch.load(os.path.join(run, "model.pt"), weights_only=True)
    assert {"opt", "sched", "train_state"} <= set(ck)
    ts = ck["train_state"]
    assert set(ts) == {"quality_key", "best_q", "best_is_quantized",
                       "best_ceil_guard",
                       "nonfinite_steps", "last_full_q", "vq", "rng"}
    # the restored RNG reproduces the saved stream
    loop.set_rng_state(ts["rng"])
    import random

    import numpy as np
    a = (torch.rand(3), np.random.rand(3), random.random())
    loop.set_rng_state(ts["rng"])
    b = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(a[0], b[0]) and (a[1] == b[1]).all() and a[2] == b[2]
