"""The base install is the decode path. These tests run the decode-path
modules in a subprocess where the [train] packages cannot be imported,
and check that georvq.train is never loaded."""
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_ONLY = ("pandas", "matplotlib", "sklearn", "wandb")
# packages and modules a base install must import; subpackages are walked
DECODE = ("georvq.inference", "georvq.ae", "georvq.tessera", "georvq.sources",
          "georvq.model", "georvq.factory", "georvq.compat", "georvq.config",
          "georvq.metrics", "georvq.repro")

_PRELUDE = f"""
import importlib, pkgutil, sys
BLOCKED = {TRAIN_ONLY!r}

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{{name}} is not in the base install")
        return None

sys.meta_path.insert(0, _Block())

def check_boundary():
    bad = sorted(m for m in sys.modules
                 if m == "georvq.train" or m.startswith("georvq.train.")
                 or m.split(".")[0] in BLOCKED)
    assert not bad, bad
"""


def _run(body):
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    code = _PRELUDE + textwrap.dedent(body)
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_decode_modules_import_without_training_packages():
    out = _run(f"""
        for blocked in ("pandas", "georvq.data.store"):
            try:
                importlib.import_module(blocked)
            except ImportError:
                pass
            else:
                raise AssertionError(f"{{blocked}} imported past the block")
        names = []
        for top in {DECODE!r}:
            m = importlib.import_module(top)
            names.append(top)
            if hasattr(m, "__path__"):
                for info in pkgutil.walk_packages(m.__path__, top + "."):
                    importlib.import_module(info.name)
                    names.append(info.name)
        assert "georvq.ae.v2" in names and "georvq.ae.index" in names
        from georvq.sources import get_source
        import georvq.sources.s2
        check_boundary()
        print(len(names))
    """)
    assert int(out.split()[-1]) > 30


def test_ae_v2_encode_decode_and_index_run_without_training_packages(tmp_path):
    pytest.importorskip("rasterio")
    _run(f"""
        import os
        import numpy as np
        from tests.test_ae_index import (TILES, source_index, tiny_codec,
                                         write_tile)
        from georvq.ae import v2
        from georvq.ae.index import build_index
        tmp = {str(tmp_path)!r}
        codec = tiny_codec()
        f = os.path.join(tmp, TILES[0][2] + ".georvq.zarr")
        write_tile(f, codec, 40, 48)
        with v2.open_dataset(f, codec=codec) as ds:
            a = ds.read(quality="3-2-0")
            assert a.shape == (64, 40, 48) and a.dtype == np.int8
            v2.reconstruct(ds, os.path.join(tmp, "out.tif"))
        source_index(os.path.join(tmp, "src.parquet"), TILES[:1])
        t = build_index(tmp, os.path.join(tmp, "src.parquet"))
        assert t.num_rows == 1
        check_boundary()
    """)
