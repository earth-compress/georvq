"""The physical-unit R-D figure survives a full-size validation pool.

Regression: torch.quantile refuses tensors over 2^24 elements, and
pooling every valid pixel of the val subset (512 windows x 37 levels x
4096 px on era5) is far past that. The first draw at step 2000 raised
and took down both A/B runs. The figure now uses kthvalue, which has no
size limit; this test drives it with a stub codec whose pooled error
tensor crosses the boundary.
"""
import matplotlib
matplotlib.use("Agg")
import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.train.figures import rd_err_figure


class _StubCodec:
    def __init__(self, specs):
        self.specs, self.training = specs, False

    def eval(self):
        return self

    def train(self):
        return self

    def level_cumsums(self, x, loc=None):
        return None, None

    def eval_tuple(self, cums, t, ref, valid=None):
        return ref + 0.1 * torch.randn_like(ref)


class _Norm:
    def denormalize(self, x, stats):
        return x * 2.0


def test_rd_err_figure_past_the_quantile_size_limit():
    cfg = load_config("configs/smoke.yaml",
                      ["model.tiers=['1:16:32:3','4:8:32:3']",
                       "model.window=64", "model.bands=8"])
    m = build_model(cfg, "cpu")
    stub = _StubCodec(m.specs)

    class Src:
        def prep(self, raw, device):
            x = raw.float()
            return x, torch.ones(x.shape[0], *x.shape[2:], dtype=torch.bool)

        def normalizer(self, model_cfg):
            return _Norm()

        def last_stats(self):
            return None

    Src.cfg = cfg.model

    # 512 windows x 8 bands x 64x64 = 16.8M pooled elements > 2^24
    val = torch.randn(512, 8, 64, 64)
    fig = rd_err_figure(stub, Src(), val, [[3, 0], [3, 3]], step=1,
                        px=4096, unit="m", max_tokens=512, chunk=128)
    labels = [t.get_text() for t in fig.legends[0].texts] if fig.legends \
        else [l.get_label() for l in fig.axes[0].lines]
    assert {"MAE", "RMSE", "p90", "p99"} <= set(labels)
