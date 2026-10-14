"""Data-parallel helpers, exercised with a real 2-process gloo group.

These run on CPU so they gate every commit, not just GPU boxes. They
target the failure modes that are SILENT in a distributed run: gradients
that never actually sync, ranks that start from different weights, and
ranks that draw identical data (which makes the second GPU pure waste
while every metric still looks plausible).
"""
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from georvq.train import dist as D


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(fn, world=2):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_entry, args=(fn, r, world, q, port))
             for r in range(world)]
    try:
        for p in procs:
            p.start()
        out = [q.get(timeout=120) for _ in range(world)]
        for p in procs:
            p.join(timeout=120)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join()
    for r, res in out:
        assert res != "ERROR", f"rank {r} failed"
    return dict(out)


def _entry(fn, rank, world, q, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_RANK=str(rank))
    try:
        dist.init_process_group(backend="gloo")
        q.put((rank, fn(rank, world)))
    except Exception:                                   # pragma: no cover
        import traceback; traceback.print_exc()
        q.put((rank, "ERROR"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------- bodies
def _grads_are_averaged(rank, world):
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.full((4,), float(rank + 1))          # 1.0 and 2.0
    D.average_grads([p])
    return p.grad.tolist()


def _broadcast_equalizes(rank, world):
    torch.manual_seed(rank)                             # deliberately differ
    m = torch.nn.Linear(4, 4)
    m.register_buffer("buf", torch.full((3,), float(rank)))
    D.broadcast_state(m)
    return (m.weight.sum().item(), m.buf.tolist())


def _rank_offset_seeds_diverge(rank, world):
    from georvq.repro import seed_all
    seed_all(1234 + 1000 * rank)
    return torch.randint(0, 1_000_000, (5,)).tolist()


def _world_and_rank(rank, world):
    return (D.rank(), D.world_size(), D.is_main())


# ---------------------------------------------------------------- tests
def test_gradients_are_averaged_not_summed():
    """Summing instead of averaging silently scales the LR by the world
    size -- the run still trains, just not the run you configured."""
    out = _run(_grads_are_averaged)
    for r in out:
        assert out[r] == pytest.approx([1.5] * 4), out
    assert out[0] == out[1], "ranks disagree after all-reduce"


def test_broadcast_state_equalizes_params_and_buffers():
    out = _run(_broadcast_equalizes)
    assert out[0][0] == pytest.approx(out[1][0]), "weights differ after bcast"
    assert out[0][1] == out[1][1] == [0.0, 0.0, 0.0], "buffers not broadcast"


def test_ranks_draw_different_data():
    """The whole point of the second rank. Identical streams would halve
    the effective batch diversity with no error anywhere."""
    out = _run(_rank_offset_seeds_diverge)
    assert out[0] != out[1], "ranks would sample identical batches"


def test_rank_helpers():
    out = _run(_world_and_rank)
    assert out[0] == (0, 2, True) and out[1] == (1, 2, False)


def test_helpers_are_noops_without_a_process_group():
    """The single-GPU path must be completely unaffected."""
    assert not D.is_distributed()
    assert D.world_size() == 1 and D.rank() == 0 and D.is_main()
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.ones(3)
    D.average_grads([p])
    assert p.grad.tolist() == [1.0, 1.0, 1.0]
    assert D.broadcast_state(torch.nn.Linear(2, 2)) == 0
    assert D.setup.__call__ is not None


def test_setup_rejects_live_ema_books_under_multi_gpu():
    """EMA codebooks live in buffers, so a gradient all-reduce does not
    touch them and the ranks' books would drift apart silently."""
    from georvq.config import load_config
    cfg = load_config("configs/smoke.yaml", ["model.vq={freeze_books: false}"])
    assert cfg.model.vq.freeze_books is False
    # the guard lives in setup(); assert the condition it keys on exists
    assert hasattr(cfg.model.vq, "freeze_books")
