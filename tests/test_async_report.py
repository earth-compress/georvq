"""Deferred reporting: worker ordering, error isolation, and the
harvest/render splits matching their one-call forms."""
import threading
import time

import matplotlib
matplotlib.use("Agg")
import torch

from georvq.train.report import AsyncReporter


def test_worker_runs_jobs_in_order_and_survives_errors():
    r = AsyncReporter()
    done = []
    lock = threading.Lock()

    def job(i):
        def _run():
            if i == 1:
                raise RuntimeError("boom")
            with lock:
                done.append(i)
        return _run

    for i in range(4):
        r.submit(job(i))
    assert r.close(timeout=10)
    assert done == [0, 2, 3]
    assert r.errors == 1


def test_close_flushes_pending_jobs():
    r = AsyncReporter()
    done = []
    r.submit(lambda: (time.sleep(0.2), done.append("slow"))[-1])
    r.submit(lambda: done.append("after"))
    assert r.close(timeout=10)
    assert done == ["slow", "after"]


def test_rd_err_split_matches_one_call(  ):
    from georvq.train.figures import rd_err_harvest, rd_err_render

    class _Stub:
        def __init__(self, specs):
            self.specs, self.training = specs, False

        def eval(self):
            return self

        def train(self):
            return self

        def level_cumsums(self, x, loc=None):
            return None, None

        def eval_tuple(self, cums, t, ref, valid=None):
            torch.manual_seed(sum(t))
            return ref + 0.1 * torch.randn_like(ref)

    class _Norm:
        def denormalize(self, x, stats):
            return x * 2.0

    class _Src:
        def prep(self, raw, device):
            x = raw.float()
            return x, torch.ones(x.shape[0], *x.shape[2:],
                                 dtype=torch.bool)

        def normalizer(self, cfg):
            return _Norm()

        def last_stats(self):
            return None
        cfg = None

    from georvq.config import Config
    import json
    from georvq.config import load_config
    cfg = load_config("configs/smoke.yaml",
                      ["model.tiers=['1:16:32:3','4:8:32:3']"])
    from georvq.factory import build_model
    m = build_model(cfg, "cpu")
    stub = _Stub(m.specs)
    val = torch.randn(32, 8, 16, 16)
    p = rd_err_harvest(stub, _Src(), val, [[3, 0], [3, 3]], px=256,
                       max_tokens=32, chunk=16, cap=1000)
    # subsample cap respected, payload is CPU tensors only
    assert all(v.numel() <= 1000 for v in p["errs"].values())
    fig = rd_err_render(p, step=1, unit="m")
    labels = [l.get_label() for l in fig.axes[0].lines]
    assert {"MAE", "RMSE", "p90", "p99"} <= set(labels)
