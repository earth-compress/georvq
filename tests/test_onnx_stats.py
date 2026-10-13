"""ONNX export of a stats_cond model (georvq.inference.onnx_stats): the
onnxruntime graphs against torch fp32 on CPU, and the runtime path
running with torch unavailable."""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")

from georvq.config import load_config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference.api import Codec  # noqa: E402
from georvq.inference.onnx_stats import (OnnxStatsCodec,  # noqa: E402
                                         export_stats_onnx)

WIN = 16
TIERS = "model.tiers=['1:16:32:3','2:8:32:2','4:8:16:2']"
STATS = ("model.stats_cond={subtract_mean: true, cond_encoder: true, "
         "cond_decoder: true, bits_per_band: 6}")
TUPLES = [(3, 2, 2), (3, 0, 0), (2, 1, 0), (1, 2, 1)]

# blocks every torch import in the child process
NO_TORCH = textwrap.dedent("""
    import sys
    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is blocked in this test")
    sys.meta_path.insert(0, _Block())
""")


def _inputs(model, n=24, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.integers(-127, 128, size=(n, 64, WIN, WIN)).astype(np.int8)
    # a per-window direction, so the mean carries most of the energy
    raw = np.clip(raw // 4 + rng.integers(-90, 90, size=(n, 64, 1, 1)),
                  -127, 127).astype(np.int8)
    raw[0, :, :5, :7] = -128
    raw[1, :, 8:, :] = -128
    x, v = model.source.prep(torch.from_numpy(raw), "cpu")
    return x, v, model.patch_stats(x, v)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    cfg = load_config("configs/smoke.yaml", [
        "model.bands=64", f"model.window={WIN}", TIERS, STATS])
    model = build_model(cfg, "cpu")
    with torch.no_grad():
        # the stats embeddings are zero-initialised; give them weight so
        # the conditioning paths are exercised
        for emb in (model.stats_enc, model.stats_dec):
            torch.nn.init.normal_(emb.mlp[-1].weight, std=0.2)
        for rvq in model.rvqs:
            for lev in rvq.levels:
                lev.codebook.weight.normal_(std=0.3)
    model.eval()
    out = str(tmp_path_factory.mktemp("onnx_stats"))
    meta = export_stats_onnx(model, out)
    return model, out, meta


def test_meta_records_the_stats_input(exported):
    model, out, meta = exported
    assert meta["format"] == "georvq-onnx-stats/1"
    assert meta["stats"] == {"width": 64, "use_std": False,
                             "subtract_mean": True}
    assert [t["depth"] for t in meta["tiers"]] == [3, 2, 2]
    for name in ("encoder.onnx", "decoder.onnx", "codebooks.npz"):
        assert os.path.exists(os.path.join(out, name))


def test_encoder_matches_torch(exported):
    model, out, _ = exported
    ort = OnnxStatsCodec(out)
    x, v, st = _inputs(model)
    with torch.no_grad():
        zt = model.encode_levels(x, st, v)
    zo = ort.latents(x.numpy(), st.numpy(), v.numpy())
    for a, b in zip(zt, zo):
        assert np.abs(a.numpy() - b).max() < 1e-4
    with torch.no_grad():
        want, _, _ = Codec(model).encode_batch(x, valid=v, mean=st)
    got = ort.encode(x.numpy(), st.numpy(), v.numpy())
    assert min(np.mean(a == b) for a, b in zip(want, got)) > 0.99


def test_decoder_matches_torch(exported):
    model, out, _ = exported
    ort = OnnxStatsCodec(out)
    x, v, st = _inputs(model, seed=1)
    idx = ort.encode(x.numpy(), st.numpy(), v.numpy())
    for depths in TUPLES:
        zqs = []
        Ls = torch.tensor([depths] * len(x))
        with torch.no_grad():
            for rvq, sp, i, L in zip(model.rvqs, model.specs, idx, depths):
                z = torch.zeros(len(x), sp.cells, sp.dim)
                for lev in range(L):
                    z = z + rvq.levels[lev].codebook(
                        torch.from_numpy(i[..., lev]))
                zqs.append(z)
            want = model.decode(zqs, Ls, st, v).numpy()
        got = ort.decode([i if L else None for i, L in zip(idx, depths)],
                         depths, st.numpy(), v.numpy())
        assert got.shape == want.shape
        assert np.abs(got - want).max() < 1e-4, depths
        # invalid pixels are zero, valid ones unit-norm (AE finalisation)
        assert (got[~v.numpy()[:, None].repeat(64, 1)] == 0).all()
        n = np.linalg.norm(got, axis=1)[v.numpy()]
        assert np.allclose(n, 1.0, atol=1e-4)


def test_runtime_imports_no_torch(exported, tmp_path):
    model, out, _ = exported
    ort = OnnxStatsCodec(out)
    x, v, st = _inputs(model, seed=2)
    idx = ort.encode(x.numpy(), st.numpy(), v.numpy())
    want = ort.decode(idx, (3, 2, 1), st.numpy(), v.numpy())
    np.savez(tmp_path / "case.npz", x=x.numpy(), v=v.numpy(), st=st.numpy(),
             want=want, **{f"i{t}": a for t, a in enumerate(idx)})
    code = NO_TORCH + textwrap.dedent(f"""
        import numpy as np
        from georvq.inference.onnx_stats import OnnxStatsCodec
        c = np.load({str(tmp_path / "case.npz")!r})
        ort = OnnxStatsCodec({out!r})
        idx = ort.encode(c["x"], c["st"], c["v"])
        for t, a in enumerate(idx):
            assert np.array_equal(a, c[f"i{{t}}"])
        got = ort.decode(idx, (3, 2, 1), c["st"], c["v"])
        assert np.array_equal(got, c["want"])
        assert "torch" not in sys.modules
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=os.getcwd())
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_refuses_a_model_without_stats(tmp_path):
    cfg = load_config("configs/smoke.yaml", [
        "model.bands=64", f"model.window={WIN}", TIERS])
    with pytest.raises(ValueError, match="stats_cond"):
        export_stats_onnx(build_model(cfg, "cpu"), str(tmp_path))


def test_meta_records_the_written_opsets(exported):
    _, _, meta = exported
    ops = meta["graph_opsets"]
    assert set(ops) == {"encoder", "decoder"}
    assert all(isinstance(v, int) and v >= meta["opset"] for v in ops.values())
