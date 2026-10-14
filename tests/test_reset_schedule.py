"""Dead-code resets follow one trainer-owned schedule.

The trainer flags every level at first_reset + k*R (`mark_resets`); a
flagged level resets at its next training forward that sees data and
clears the flag. Levels do not schedule themselves: under skip_masked a
level sits out some batches, so a private per-level counter drifted
(observed on era5_t64: resets at 3000 / 3002 / 3003) and a rarely-used
level reset far less often than configured.
"""
import torch

from georvq.model.vq import VectorQuantizer
from georvq.train.loop import mark_resets

INTERVAL, FIRST, STEPS = 100, 50, 700


def _model(*levels):
    holder = torch.nn.Module()
    holder.levels = torch.nn.ModuleList(levels)
    m = torch.nn.Module()
    m.rvqs = torch.nn.ModuleList([holder])
    return m


def _vq():
    vq = VectorQuantizer(num_codes=8, dim=4, decay=0.9, kmeans_init=False,
                         reset_interval=INTERVAL, dead_threshold=0.01)
    vq.train()
    return vq


def _drive():
    """A dense level (active every step) and a sparse one (every 3rd);
    returns the global steps each reset at."""
    torch.manual_seed(0)
    dense, sparse = _vq(), _vq()
    m = _model(dense, sparse)
    fired = {"dense": [], "sparse": []}
    for name, vq in (("dense", dense), ("sparse", sparse)):
        vq._reset_dead = (lambda n: lambda *a, **k:
                          fired[n].append(step))(name)
    x = torch.randn(64, 4)
    for step in range(1, STEPS + 1):
        mark_resets(m, step, FIRST)
        dense(x)
        if step % 3 == 0:
            sparse(x)
    return fired


def test_levels_hold_the_global_schedule():
    fired = _drive()
    d, s = fired["dense"], fired["sparse"]
    assert d == list(range(FIRST, STEPS + 1, INTERVAL))
    # the sparse level resets at its first active batch after each point:
    # at most its activity gap late, never drifting, never doubled
    assert len(s) == len(d)
    assert all(0 <= b - a < 3 for a, b in zip(d, s))


def test_a_level_never_resets_itself():
    """Without the trainer's flag nothing fires, whatever reset_interval
    says: the interval is the trainer's period, not a private clock."""
    vq = _vq()
    vq._reset_dead = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("reset without a schedule"))
    for _ in range(3 * INTERVAL):
        vq(torch.randn(64, 4))


def test_disabled_levels_are_not_flagged():
    a, b = _vq(), _vq()
    b.reset_interval = 0
    m = _model(a, b)
    mark_resets(m, FIRST, FIRST)
    assert a.reset_pending and not b.reset_pending
    a.reset_pending = False
    mark_resets(m, FIRST - 1, FIRST)          # before the first point
    mark_resets(m, FIRST + 1, FIRST)          # between points
    assert not a.reset_pending
