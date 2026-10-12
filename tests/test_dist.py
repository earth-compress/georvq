"""Data-parallel helpers, exercised with a real 2-process gloo group.

These run on CPU so they gate every commit, not just GPU boxes. They
target the failure modes that are SILENT in a distributed run: gradients
that never actually sync, ranks that start from different weights, and
ranks that draw identical data (which makes the second GPU pure waste
while every metric still looks plausible).
"""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from georvq.train import dist as D


def _run(fn, world=2):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_entry, args=(fn, r, world, q))
             for r in range(world)]
    for p in procs:
        p.start()
    out = [q.get(timeout=120) for _ in range(world)]
    for p in procs:
        p.join(timeout=120)
    for r, res in out:
        assert res != "ERROR", f"rank {r} failed"
    return dict(out)


def _entry(fn, rank, world, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29517",
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


# ------------------------------------------------------ live EMA books
def _books(seed=0, reset_interval=3):
    from georvq.model.vq_quant import QuantizedResidualVQ
    torch.manual_seed(seed)
    rvq = QuantizedResidualVQ(num_levels=2, num_codes=16, dim=4,
                              decay=0.9, reset_interval=reset_interval,
                              dead_threshold=0.5, reset_mode="split")
    m = torch.nn.Module()
    m.rvqs = torch.nn.ModuleList([rvq])
    m.train()
    m.rvqs[0](_batch(-1))      # lazy data init on a batch every rank shares
    return m


def _batch(step):
    g = torch.Generator().manual_seed(100 + step)
    return torch.randn(64, 4, generator=g)


def _ema_state(m):
    return [torch.cat([lev.ema_count, lev.ema_sum.flatten(),
                       lev.codebook.weight.data.flatten()]).tolist()
            for lev in m.rvqs[0].levels]


def _synced_books(rank, world, steps=2, reset_interval=100):
    m = _books(reset_interval=reset_interval)
    D.broadcast_state(m)
    sync = D.BookSync(m)
    n_bcast = 0
    for step in range(steps):
        x = _batch(step).chunk(world)[rank]
        sync.snapshot()
        m.rvqs[0](x)
        n_bcast += sync.sync()
    return _ema_state(m), D.state_checksum(m), n_bcast


def test_book_sync_matches_one_process_on_the_global_batch():
    """Two ranks on half a batch each must end with the books one process
    gets from the whole batch. Summed, not averaged: dead_threshold is an
    absolute count."""
    ref = _books(reset_interval=100)
    for step in range(2):
        ref.rvqs[0](_batch(step))
    want = _ema_state(ref)
    out = _run(_synced_books)
    for r in out:
        got, drift, _ = out[r]
        assert drift == 0.0
        for g, w in zip(got, want):
            assert g == pytest.approx(w, rel=1e-5, abs=1e-6)
    assert out[0][0] == out[1][0], "ranks disagree after sync"


def _synced_through_reset(rank, world):
    # level step 3 (init batch + 2) is a reset step on every level
    return _synced_books(rank, world, steps=4, reset_interval=3)


def test_book_sync_ranks_agree_through_a_reset():
    """Resets draw replacements from rank-local data; the ranks must still
    hold identical books afterwards."""
    out = _run(_synced_through_reset)
    assert out[0][2] == 2, "no reset step was exercised"
    assert out[0][1] == 0.0 and out[1][1] == 0.0
    assert out[0][0] == out[1][0]
