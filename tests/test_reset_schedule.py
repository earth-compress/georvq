"""Dead-code resets fire on the trainer's global-step schedule.

Each level's private `step` counter advances only on batches where depth
sampling makes it active, so counters drift apart across levels
(observed on era5_t64: resets at 3000 / 3002 / 3003) and a
rarely-sampled level resets far less often than the interval. The
trainer now syncs every counter to its global step and anchors
`next_reset`; these tests drive two levels with different activity
rates and check the schedule holds for both.
"""
import torch

from georvq.model.vq import VectorQuantizer

INTERVAL, ANCHOR, STEPS = 100, 50, 700


def _vq():
    return VectorQuantizer(num_codes=8, dim=4, decay=0.9, kmeans_init=False,
                           reset_interval=INTERVAL, dead_threshold=0.01)


def _drive(sync):
    """Run a dense (every step) and a sparse (every 3rd step) level under
    the trainer protocol; returns the global steps each fired at."""
    torch.manual_seed(0)
    dense, sparse = _vq(), _vq()
    fired = {"dense": [], "sparse": []}
    for name, vq in (("dense", dense), ("sparse", sparse)):
        if sync:
            vq.next_reset = ANCHOR
        vq.train()
        vq._reset_dead = (lambda n: lambda *a, **k:
                          fired[n].append(step))(name)
    x = torch.randn(64, 4)
    for step in range(1, STEPS + 1):
        if sync:
            dense.step.fill_(step)
            sparse.step.fill_(step)
        dense(x)
        if step % 3 == 0:
            sparse(x)
    return fired


def test_synced_levels_hold_the_global_schedule():
    fired = _drive(sync=True)
    d, s = fired["dense"], fired["sparse"]
    # one firing per schedule point, at a fixed cadence
    assert len(d) == len(s) == (STEPS - ANCHOR) // INTERVAL + 1
    assert all(b - a == INTERVAL for a, b in zip(d, d[1:]))
    # the sparse level fires at its first active batch after each point:
    # at most its activity gap behind, and never drifting further
    assert all(abs(a - b) <= 3 for a, b in zip(d, s))


def test_unsynced_counters_reproduce_the_drift():
    """The bug this guards against: private activity counters make a
    1-in-3 level fire at one third of the intended cadence."""
    fired = _drive(sync=False)
    d, s = fired["dense"], fired["sparse"]
    assert all(b - a == INTERVAL for a, b in zip(d, d[1:]))
    assert len(s) < len(d) / 2
    assert s and s[0] > 3 * d[0] - 10
