"""Stop requests from Slurm, turned into a flag the training loop polls.

Slurm sends SIGUSR1 ahead of the time limit when the job asks for it
(`--signal=B:USR1@300`) and SIGTERM on `scancel`. The handlers only set
a flag; the loop checks it once per step, writes a resumable checkpoint
and exits 0, so a run is never killed in the middle of a step or of a
checkpoint write.
"""
from __future__ import annotations

import signal
import threading
from contextlib import contextmanager

STOP_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)


class StopFlag:
    """Truthy once a stop signal has arrived; `signame` says which."""

    def __init__(self):
        self.signum = None

    def __bool__(self):
        return self.signum is not None

    @property
    def signame(self):
        return signal.Signals(self.signum).name if self.signum else None

    def _handler(self, signum, frame):
        if self.signum is None:
            print(f"received {signal.Signals(signum).name}: saving at the "
                  f"end of this step, then exiting", flush=True)
        self.signum = signum


@contextmanager
def stop_on_signals():
    """Install SIGUSR1/SIGTERM handlers for the duration of the block and
    yield the flag they set. Off the main thread Python cannot install
    handlers, so the flag is returned unarmed; the previous handlers are
    restored on exit."""
    flag = StopFlag()
    if threading.current_thread() is not threading.main_thread():
        yield flag
        return
    prev = {s: signal.signal(s, flag._handler) for s in STOP_SIGNALS}
    try:
        yield flag
    finally:
        for s, h in prev.items():
            signal.signal(s, h)
