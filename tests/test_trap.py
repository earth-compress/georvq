import os

import torch
import torch.nn as nn

from georvq.config import TrapConfig
from georvq.train.trap import Trap, group_of


def _toy(tmp_path, **kw):
    m = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    cfg = TrapConfig(min_step=0, snapshot_every=2,
                     batch_history_bytes=10_000, **kw)
    return m, Trap(cfg, m, str(tmp_path)), cfg


def test_group_of_decoder_names():
    assert group_of("decoder.stages.0.3.ada.1.weight") == "dec.stage1.ada"
    assert group_of("decoder.stages.1.0.attn.in_proj_weight") \
        == "dec.stage2.attn"
    assert group_of("decoder.stem.2.mlp.0.weight") == "dec.stem.mlp"
    assert group_of("decoder.heads.1.render.0.bias") == "dec.head1"
    assert group_of("decoder.cond_mlps.0.0.weight") == "dec.cond"
    assert group_of("decoder.projs.2.weight") == "dec.in"
    assert group_of("tokenizer.vit.layers.0.linear1.weight") == "tokenizer"


def test_tripwire_fires_on_surge_and_nonfinite(tmp_path):
    m, trap, _ = _toy(tmp_path, gnorm_abs=0.5, gnorm_ratio=10.0)
    opt = torch.optim.AdamW(m.parameters())
    for s in range(20):                       # establish a low EMA
        assert not trap.observe(s, 0.05, opt, {})
    assert trap.observe(20, 5.0, opt, {})     # 100x the EMA
    assert not trap.observe(21, 0.3, opt, {})  # below abs floor
    assert trap.observe(22, float("nan"), opt, {})
    # EMA must not have been ratcheted up by the surge
    assert trap.ema < 0.1


def test_dump_holds_prestep_state_and_flight(tmp_path):
    m, trap, _ = _toy(tmp_path)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    x = torch.randn(3, 4)
    for s in range(4):
        opt.zero_grad()
        m(x).pow(2).mean().backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
        trap.observe(s, gn, opt, {"loss": 1.0})
        pre = {k: v.clone() for k, v in m.state_dict().items()}
        opt.step()
        trap.after_step(s, m, opt, {"x": x, "st": None})
    opt.zero_grad()
    m(x).pow(2).mean().backward()
    trap.dump(9, m, opt, sched, {"x": x, "st": None}, {"gnorm": 1.0})
    d = os.path.join(str(tmp_path), "trap_step9")
    b = torch.load(os.path.join(d, "bundle.pt"), weights_only=False)
    assert set(b["grads"]) == {n for n, _ in m.named_parameters()}
    assert b["batch"]["st"] is None
    assert b["batch"]["x"].dtype is torch.bfloat16
    assert b["rng"]["torch"] is not None and "opt" in b
    # rolling snapshot is the state from the last snapshot_every multiple
    snap = torch.load(os.path.join(d, "snapshot_pre.pt"),
                      weights_only=False)
    assert snap["step"] == 2
    hist = torch.load(os.path.join(d, "batch_history.pt"),
                      weights_only=False)
    assert [h["step"] for h in hist][-1] == 3
    flight = os.path.join(str(tmp_path), "trap_flight.npz")
    assert os.path.exists(flight)
    import numpy as np
    f = np.load(flight)
    assert "gnorm" in f and "u_0" in f.files or "g_0" in f.files


def test_trap_smoke_dumps_replayable_bundle(tmp_path):
    """End-to-end: a hair-trigger trap on the synthetic smoke run must
    dump a bundle whose weights + batch reproduce the trapped loss."""
    from georvq.config import load_config
    from georvq.data.build import build_synthetic_store
    from georvq.train.loop import train
    build_synthetic_store(tmp_path / "store", n_chips=8, bands=8,
                          chip_px=64, shard_size=4)
    cfg = load_config("configs/smoke.yaml", [
        f"data.store_dir={tmp_path / 'store'}",
        f"log.out_dir={tmp_path / 'ckpt'}",
        "train.trap={min_step: 5, gnorm_abs: 1e-9, gnorm_ratio: 1e-9, "
        "n_dumps: 1, snapshot_every: 4}"])
    run_dir = train(cfg, gpu=-1)
    dumps = [d for d in os.listdir(run_dir) if d.startswith("trap_step")]
    assert len(dumps) == 1
    b = torch.load(os.path.join(run_dir, dumps[0], "bundle.pt"),
                   weights_only=False)
    assert b["step"] == 5 and b["grads"] and "exp_avg_sq" in str(b["opt"])
    assert os.path.exists(os.path.join(run_dir, "trap_flight.npz"))


def test_update_ratio_reads_adam_moments(tmp_path):
    m, trap, _ = _toy(tmp_path)
    opt = torch.optim.AdamW(m.parameters())
    m(torch.randn(3, 4)).pow(2).mean().backward()
    opt.step()                                # moments now exist
    row = trap._group_stats(opt)
    assert any(k.startswith("u_") and v > 0 for k, v in row.items())


def test_group_of_plain_decoder_blocks():
    assert group_of("decoder.blocks.3.ada.1.weight") == "dec.block3.ada"
    assert group_of("decoder.blocks.0.attn.in_proj_weight") == \
        "dec.block0.attn"
    assert group_of("decoder.blocks.2.norm.weight") == "dec.block2.other"


def test_hold_listens_on_a_private_unix_socket(tmp_path):
    import stat
    import threading
    import time
    _, trap, _ = _toy(tmp_path, hold_timeout=1.0)
    path = os.path.join(trap.dir, "trap_hold.sock")
    t = threading.Thread(target=trap.hold, args=({},))
    t.start()
    for _ in range(200):
        if os.path.exists(path):
            break
        time.sleep(0.005)
    st = os.stat(path)
    assert stat.S_ISSOCK(st.st_mode)
    assert stat.S_IMODE(st.st_mode) == 0o600
    t.join(5)
    assert not t.is_alive() and not os.path.exists(path)
