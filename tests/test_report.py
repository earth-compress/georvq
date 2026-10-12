"""Per-source reporting: what a run measures, plots and logs.

The bug these pin: reporting used to be inherited from whichever source
came before. A single-band elevation model logged "cosine" (degenerate on
one band) because `validate()` read `recon_stats(...)["cos"]` by name, and
a 3-component PCA-RGB preview of one band cannot be drawn at all.
"""
import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.sources import Source
from georvq.sources.report import (GRAY, PCA_RGB, R2, RMSE,
                                   ReportConfigError, ReportSpec,
                                   band_rgb_preview, embedding_field_spec,
                                   resolve_report, scalar_field_spec)


def _cfg(source="ae", bands=8, **log_report):
    return Config.model_validate({
        "data": {"source": source, "store_dir": "/tmp/unused"},
        "model": {
            "window": 16, "bands": bands,
            "tiers": ["1:8:16:2", "4:8:16:2"],
            "tokenizer": {"dim": 16, "depth": 1},
            "decoder": {"dim": 16, "depth": 1, "heads": 2, "head_feat": 4},
        },
        "log": {"run_tag": "t", "wandb": False, "report": log_report},
    })


class _FakeSource(Source):
    """A source whose declaration we control, to test resolution alone."""
    name = "fake"

    def __init__(self, spec):
        self._spec = spec

    def report(self, model_cfg):
        return self._spec


# ------------------------------------------------------- what sources say

def test_each_shipped_source_declares_its_own_surface():
    """The three shipped sources must not share a reporting surface by
    accident -- that sharing is the bug."""
    from georvq.sources.ae import AESource
    from georvq.sources.s2 import S2Source
    from georvq.sources.tessera import TesseraSource

    cfg = _cfg()
    ae = AESource(cfg.model).report(cfg.model)
    tes = TesseraSource(cfg.model).report(cfg.model)
    assert ae.quality.key == "cos" and tes.quality.key == "cos"
    # Tessera adds the magnitude diagnostic its non-unit-norm field needs
    assert "vec_rel_l2" in [m.key for m in tes.extra]
    assert "vec_rel_l2" not in [m.key for m in ae.extra]

    s2cfg = Config.model_validate({
        "data": {"source": "s2", "store_dir": "/tmp/unused"},
        "model": {"window": 24, "bands": 32, "tiers": ["1:16:32:3",
                                                       "4:8:32:3"],
                  "tokenizer": {"dim": 32, "depth": 1},
                  "decoder": {"dim": 32, "depth": 1, "heads": 4,
                              "head_feat": 8}},
        "log": {"run_tag": "t", "wandb": False},
    })
    s2 = S2Source(s2cfg.model).report(s2cfg.model)
    # Reflectance, not an embedding field: no cosine, no block-mean floor.
    # The headline is ABSOLUTE (R2), because S2's magnitude spread is 6.6x
    # -- the widest in the five-source table -- and because the loss is
    # MSE, so selecting checkpoints on a direction metric would select on
    # something the run is not optimising. The angle stays as a
    # diagnostic: band ratios can degrade while MSE looks fine.
    assert s2.quality.key == "r2"
    assert [m.key for m in s2.extra] == ["rmse", "sam"]
    assert s2.baselines == ()
    assert s2.preview.key == "band_rgb"


def test_a_source_that_declares_nothing_fails_loudly():
    """No silent default -- that is how the inheritance bug happened."""
    class Bare(Source):
        name = "bare"

    with pytest.raises(NotImplementedError, match="declares no ReportSpec"):
        Bare(None).report(None)


# ------------------------------------------------------ the band-count gate

def test_cosine_is_refused_on_a_single_band_field():
    """The DEM case: cosine over one band collapses to sign agreement, so
    declaring it must fail rather than publish a meaningless 0.99."""
    src = _FakeSource(embedding_field_spec())
    with pytest.raises(ReportConfigError, match="sign agreement"):
        resolve_report(src, _cfg(bands=1))


def test_pca_preview_is_refused_below_three_bands():
    src = _FakeSource(ReportSpec(quality=R2, preview=PCA_RGB))
    with pytest.raises(ReportConfigError, match="preview"):
        resolve_report(src, _cfg(bands=2))


def test_scalar_field_spec_is_fine_on_one_band():
    """R2 + RMSE + grayscale is exactly what a 1-band source should get."""
    spec = resolve_report(_FakeSource(scalar_field_spec()), _cfg(bands=1))
    assert spec.quality.key == "r2"
    assert [m.key for m in spec.extra] == ["rmse"]
    assert spec.baselines == ()          # no unit-norm block-mean floor
    assert spec.preview.key == "gray"


# ------------------------------------------------------------ config gating

def test_config_can_narrow_the_metric_set_and_pick_the_headline():
    spec = resolve_report(_FakeSource(embedding_field_spec()),
                          _cfg(metrics=["ccos", "cos"]))
    assert spec.quality.key == "ccos"    # first listed becomes the headline
    assert [m.key for m in spec.extra] == ["cos"]


def test_config_cannot_widen_beyond_what_the_source_declares():
    with pytest.raises(ReportConfigError, match="not declared by this source"):
        resolve_report(_FakeSource(embedding_field_spec()),
                       _cfg(metrics=["cos", "sam"]))


def test_config_can_switch_figures_off_entirely():
    spec = resolve_report(_FakeSource(embedding_field_spec()),
                          _cfg(figures=[]))
    assert spec.figures == ()


def test_unknown_figure_name_is_a_config_error():
    with pytest.raises(ReportConfigError, match="unknown"):
        resolve_report(_FakeSource(embedding_field_spec()),
                       _cfg(figures=["rd", "nope"]))


def test_preview_can_be_disabled_which_also_drops_the_recon_figure():
    spec = resolve_report(_FakeSource(embedding_field_spec()),
                          _cfg(preview="none"))
    assert spec.preview is None
    assert not spec.has_figure("recon")


def test_baselines_can_be_dropped():
    spec = resolve_report(_FakeSource(embedding_field_spec()),
                          _cfg(baselines=[]))
    assert spec.baselines == ()


def test_an_unbounded_metric_drops_the_log_distortion_axis():
    """1 - m on a log scale is meaningless for an RMSE or an angle."""
    spec = resolve_report(_FakeSource(scalar_field_spec(quality=RMSE)),
                          _cfg(bands=1))
    assert not spec.has_figure("rd_log")
    assert spec.has_figure("rd")


def test_defaults_change_nothing_for_an_embedding_field():
    """The shipped AE/Tessera surface must survive untouched, so existing
    runs stay comparable."""
    spec = resolve_report(_FakeSource(embedding_field_spec()), _cfg())
    assert spec.quality.key == "cos"
    assert [m.key for m in spec.extra] == ["ccos"]
    assert [b.key for b in spec.baselines] == ["blockmean"]
    assert spec.preview.key == "pca_rgb"
    assert spec.has_figure("rd") and spec.has_figure("rd_log")


# ------------------------------------------------------------- the metrics

def test_r2_is_one_for_a_perfect_reconstruction_and_zero_for_the_mean():
    x = torch.randn(4, 1, 8, 8)
    valid = torch.ones(4, 8, 8, dtype=torch.bool)
    assert float(R2.fn(x, x, valid)) == pytest.approx(1.0, abs=1e-5)
    flat = x.mean().expand_as(x)
    assert float(R2.fn(flat, x, valid)) == pytest.approx(0.0, abs=1e-5)


def test_spectral_angle_is_zero_for_a_pure_gain_change():
    """SAM reads band RATIOS: scaling every band of a pixel must not move
    it, which is exactly why S2 uses it over a per-band MSE."""
    from georvq.metrics import spectral_angle
    torch.manual_seed(0)
    x = torch.rand(2, 5, 4, 4) + 0.1
    valid = torch.ones(2, 4, 4, dtype=torch.bool)
    # arccos is ill-conditioned at 0: cos = 1 - eps maps to angle ~ sqrt(2 eps),
    # so fp32 rounding alone puts a noise floor near 4e-4 radians here.
    assert float(spectral_angle(x * 2.5, x, valid)) == pytest.approx(0, abs=1e-3)


def test_masked_metrics_ignore_invalid_pixels():
    from georvq.metrics import masked_rmse
    x = torch.zeros(1, 2, 4, 4)
    out = x.clone()
    out[:, :, 0, 0] = 99.0                    # a wild error...
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[:, 0, 0] = False                    # ...on an invalid pixel
    assert float(masked_rmse(out, x, valid)) == pytest.approx(0.0)


# ------------------------------------------------------------- the previews

def test_gray_preview_renders_a_single_band_to_rgb():
    """PCA-RGB cannot do this at all -- it needs 3 components."""
    x = torch.randn(3, 1, 8, 8)
    valid = torch.ones(3, 8, 8, dtype=torch.bool)
    rgb = GRAY.render(x, GRAY.basis(x, valid))
    assert rgb.shape == (3, 8, 8, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_band_rgb_preview_can_select_a_group_from_a_dict():
    """S2's model emits {'10m': ..., '20m': ...}; the preview renders the
    10m group's true-colour triplet."""
    pv = band_rgb_preview(2, 1, 0, select=lambda t: t["10m"])
    x = {"10m": torch.rand(2, 4, 8, 8), "20m": torch.rand(2, 6, 4, 4)}
    valid = torch.ones(2, 8, 8, dtype=torch.bool)
    rgb = pv.render(x, pv.basis(x, valid))
    assert rgb.shape == (2, 8, 8, 3)


def test_pca_preview_still_renders_an_embedding_field():
    x = torch.randn(2, 8, 8, 8)
    valid = torch.ones(2, 8, 8, dtype=torch.bool)
    rgb = PCA_RGB.render(x, PCA_RGB.basis(x, valid))
    assert rgb.shape == (2, 8, 8, 3)


# ------------------------------------------------------------ end to end

def test_validate_reports_under_the_declared_key_not_cos():
    """The whole point: a scalar source's val scalars say r2, and no
    cosine or block-mean panel appears anywhere."""
    from georvq.train.validate import validate

    cfg = _cfg(bands=1)
    torch.manual_seed(0)
    model = build_model(cfg)

    class _Scalar(Source):
        name = "scalar"

        def prep(self, raw, device):
            x = raw.float()
            return x, torch.ones(x.shape[0], *x.shape[2:], dtype=torch.bool)

        def recon_stats(self, out, x, valid):
            from georvq.metrics import masked_r2, masked_rmse
            return masked_rmse(out, x, valid), {
                "r2": masked_r2(out, x, valid)}

        def report(self, model_cfg):
            return scalar_field_spec()

    src = _Scalar(cfg.model)
    spec = resolve_report(src, cfg)
    val_s = torch.randn(8, 1, 16, 16)
    vs, by_tuple = validate(model, src, val_s, [[1, 1], [2, 2]],
                            canon_names={"2-2"}, spec=spec)
    assert "r2_ceiling" in vs and "rmse_ceiling" in vs
    assert not any(k.startswith("cos") for k in vs)
    assert "cos_blockmean" not in vs and "r2_blockmean" not in vs
    assert set(by_tuple) == {"1-1", "2-2"}
    assert "rmse" in vs["_extra"] and "2-2" in vs["_extra"]["rmse"]


def test_rd_figure_labels_the_axis_with_the_declared_metric():
    import matplotlib.pyplot as plt

    from georvq.train.figures import rd_figure

    cfg = _cfg(bands=1)
    spec = resolve_report(_FakeSource(scalar_field_spec()), cfg)
    specs = cfg.model.tiers
    tuples = [[1, 1], [2, 2]]
    vs = {"r2_ceiling": 0.99}
    fig = rd_figure({"1-1": 0.5, "2-2": 0.9}, vs, specs, tuples, 100, spec,
                    px=256)
    assert fig.axes[0].get_ylabel() == "R² (variance explained)"
    plt.close(fig)


def test_rd_figure_pareto_front_follows_the_metric_direction():
    """For a lower-is-better metric the frontier is the running MIN; the
    old code hardcoded a running max and would have drawn a flat line."""
    import matplotlib.pyplot as plt

    from georvq.train.figures import rd_figure

    cfg = _cfg(bands=1)
    spec = resolve_report(_FakeSource(scalar_field_spec(quality=RMSE)), cfg)
    fig = rd_figure({"1-1": 5.0, "2-2": 2.0}, {"rmse_ceiling": 1.0},
                    cfg.model.tiers, [[1, 1], [2, 2]], 100, spec, px=256)
    front = [ln for ln in fig.axes[0].lines
             if ln.get_label() == "Pareto front"][0]
    assert list(front.get_ydata()) == [5.0, 2.0]     # improving downward
    plt.close(fig)
