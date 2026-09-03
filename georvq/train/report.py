"""Deferred reporting: render figures and log them off the training path.

The reporting steps split in two. The HARVEST runs on the training
thread while the model is at hand: every decode the figures need
happens there, and the results land in plain CPU tensors (subsampled at
source, so a payload is tens of MB, not GB). The RENDER — percentiles,
matplotlib, wandb image upload — needs nothing but those payloads, and
runs here on a single background worker while the GPU goes back to
training. Measured on era5_t64 fig-steps, the render tail was over two
of the three minutes the GPU sat idle.

One worker, a bounded queue: reports are rendered in submission order,
and if rendering ever falls behind by more than one full report the
training thread blocks on submit rather than stacking up payloads. A
render error is printed and counted, never raised into training.

wandb ordering: the worker logs after the training thread has moved on,
so figure logs use their own step metric (`figs/step`, declared via
define_metric at setup) instead of the global step, which wandb
requires to be monotone.
"""
from __future__ import annotations

import queue
import threading
import traceback


class AsyncReporter:
    """Run zero-argument render jobs on one background thread, in order."""

    _SENTINEL = object()

    def __init__(self, maxsize=2):
        self.q = queue.Queue(maxsize=maxsize)
        self.errors = 0
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="report-worker")
        self._thread.start()

    def submit(self, job):
        """Queue a job (blocks if the worker is a full report behind)."""
        self.q.put(job)

    def _run(self):
        while True:
            job = self.q.get()
            if job is self._SENTINEL:
                return
            try:
                job()
            except Exception:
                self.errors += 1
                traceback.print_exc()

    def close(self, timeout=600):
        """Flush remaining jobs and stop the worker."""
        self.q.put(self._SENTINEL)
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()
