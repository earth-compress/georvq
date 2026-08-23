"""The lr schedule's edge cases, which only bite at run time.

`warmup: 0` is the natural way to write "no warmup" and used to raise
ZeroDivisionError at the FIRST optimiser step -- so a run died minutes
in, after the resident pool had prefilled and the warm start had loaded.
Config validation cannot catch it: 0 is a legal int.
"""
import numpy as np
import pytest


def lr_fn(s, warmup, lr0, const, steps=1000):
    """Mirror of the schedule in georvq/train/loop.py."""
    if s < lr0:
        return 0.0
    s -= lr0
    ramp = min(1.0, (s + 1) / max(warmup, 1))
    if const:
        return ramp
    total = max(steps - lr0, 1)
    return ramp * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total)))


@pytest.mark.parametrize("lr0", [0, 500])
@pytest.mark.parametrize("const", [True, False])
def test_zero_warmup_is_finite_everywhere(lr0, const):
    for s in (0, lr0, lr0 + 1, 999):
        v = lr_fn(s, warmup=0, lr0=lr0, const=const)
        assert np.isfinite(v) and v >= 0.0


def test_zero_warmup_means_full_lr_immediately():
    """No warmup should mean exactly that, not a one-step ramp."""
    assert lr_fn(500, warmup=0, lr0=500, const=True) == 1.0
    assert lr_fn(501, warmup=0, lr0=500, const=True) == 1.0


def test_lr0_phase_is_exactly_zero():
    """The lr=0 phase is an assertion mechanism -- if it leaked a little
    lr, a 'nothing moved' check would quietly stop being one."""
    for s in range(500):
        assert lr_fn(s, warmup=1, lr0=500, const=True) == 0.0
    assert lr_fn(500, warmup=1, lr0=500, const=True) == 1.0


def test_warmup_still_ramps_when_asked_for():
    vals = [lr_fn(s, warmup=100, lr0=0, const=True) for s in range(100)]
    assert vals[0] == pytest.approx(0.01)
    assert vals[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(vals, vals[1:]))
