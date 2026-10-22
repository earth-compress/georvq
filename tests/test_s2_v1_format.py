"""The S2 v1 store format, held fixed while the release code changes.

As tests/test_tessera_v2_format.py does for Tessera v2:

- A store written by `georvq.s2.v1.encode_safe` from a small SAFE
  product (tests/s2_safe_fixture.py) with a tiny S2-shaped codec is
  compared with golden/s2_v1_store_snapshot.json: every node's
  attributes and array metadata as JSON, the stored keys, and the sha256
  of every chunk's decoded bytes, the `safe_meta` asset included. The
  codec's records are an integer function of the source DN, and its ids
  are fixed strings, so the snapshot does not depend on float
  arithmetic; the model's numerics are pinned by test_s2_v1_bundle. The
  product has nodata in a corner (one chunk with no valid pixel) and is
  stored on three output grids. Values that depend on the GDAL, PROJ or
  rasterio version (WKT strings, PROJJSON) are left out, and so is the
  JPEG 2000 coding of the fixture's band images (the snapshot holds
  their parsed parameters, not their bytes).
- With the released bundle present, the golden tile is encoded again on
  CPU and the store is compared file by file with
  golden/s2_v1_tile.georvq.zarr.

Re-record with

    python -m tests.test_s2_v1_format record
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

pytest.importorskip("rasterio")
pytest.importorskip("zarr")

from georvq.s2 import v1  # noqa: E402
from tests.s2_safe_fixture import make_safe, synthetic_bands  # noqa: E402
from tests.test_ae_v2_format import _files, snapshot  # noqa: E402
from tests.test_s2_v1 import T10, tiny_codec  # noqa: E402

HERE = os.path.dirname(__file__)
SNAPSHOT = os.path.join(HERE, "golden/s2_v1_store_snapshot.json")
N = 216
CHUNK_PX = {"base": 108, "tier1": 72}
BITSTREAM_ID, DECODER_ID = "0123456789abcdef", "fedcba9876543210"


def _codec():
    codec = tiny_codec()
    codec.bitstream_id, codec.decoder_id = BITSTREAM_ID, DECODER_ID
    lay = codec.layout(quality=True)
    rng = np.random.default_rng(1)
    W = codec.window
    proj = [rng.integers(1, 1 << 20, size=(n * (W // f) ** 2, 64))
            for (_, f, _), n in zip(v1.GRIDS, (4, 6, 3))]

    def encode_windows(raw, stats=True):
        """Records as an integer function of each window's DN."""
        B = raw[0].shape[0]
        h = sum(a.reshape(B, -1).astype(np.int64) @ p
                for a, p in zip(raw, proj))

        def values(col, salt):
            w = np.array(lay.columns[col].widths, np.int64)
            k = np.arange(len(w))
            v = h[:, (k + salt) % 64] + 7919 * k
            return v % (1 << w)

        rec = {"base": values("base", 0)[None]}
        for t in lay.tiers:
            if not t.in_base:
                rec[t.name] = np.stack([values(c, 3 * i + 1) for i, c in
                                        enumerate(t.columns)])
        if stats:
            rec["quality"] = values("quality", 5)[None]
        return rec, None, None

    codec.encode_windows = encode_windows
    return codec


def _bands():
    a = synthetic_bands(N, seed=4)
    for x, f in zip(a, v1.FACTORS):
        x[:, :108 // f, 108 // f:] = 0          # one empty base chunk
    return a


def _write_store(tmp):
    safe = make_safe(os.path.join(tmp, "src"), _bands(), T10)
    out = os.path.join(tmp, "snap.georvq.zarr")
    v1.encode_safe(safe, out, _codec(), chunk_px=CHUNK_PX, strip_px=72,
                   batch=5, revision="r1", source="snap/product.SAFE")
    return out


def _snapshot(out):
    snap = snapshot(out)
    # the zip holds the fixture's JPEG 2000 files (TCI, PVI, a mask),
    # whose bytes depend on the OpenJPEG version: its size and hash are
    # left out, and test_s2_v1 checks its contents
    snap["chunks"].pop("assets/safe_meta")
    node = snap["nodes"]["assets/safe_meta"]
    node.pop("shape")
    node["chunk_grid"].pop("configuration")
    for k in ("size", "sha256"):
        node["attributes"]["georvq_asset"].pop(k)
    return snap


def test_store_format_matches_the_snapshot(tmp_path):
    torch.set_grad_enabled(False)
    try:
        out = _write_store(str(tmp_path))
    finally:
        torch.set_grad_enabled(True)
    got = _snapshot(out)
    want = json.load(open(SNAPSHOT))
    assert got["keys"] == want["keys"]
    assert set(got["nodes"]) == set(want["nodes"])
    for node in want["nodes"]:
        assert got["nodes"][node] == want["nodes"][node], node
    assert got["chunks"] == want["chunks"]
    aux = json.load(open(os.path.join(out, "zarr.json")))["attributes"][
        "georvq_aux"]["safe"]
    assert list(aux["band_files"]) == list(v1.S2_ORDER)
    assert aux["jp2"]["B01"]["precision"] == 15


# --------------------------------------------------------------------------
# the golden tile, encoded again with the released bundle
# --------------------------------------------------------------------------

def test_golden_tile_encodes_to_the_same_store(tmp_path):
    from tests.test_s2_v1_bundle import TILE, _bundle_codec, _inputs, write_tile
    if not os.path.exists(TILE):
        pytest.skip("tile fixture not recorded")
    codec = _bundle_codec()
    out = str(tmp_path / "golden_tile.georvq.zarr")
    torch.set_grad_enabled(False)
    try:
        write_tile(codec, out, *_inputs())
    finally:
        torch.set_grad_enabled(True)
    want, got = _files(TILE), _files(out)
    assert sorted(got) == sorted(want)
    for k in want:
        assert got[k] == want[k], k


def record():
    torch.set_grad_enabled(False)
    with tempfile.TemporaryDirectory() as tmp:
        snap = _snapshot(_write_store(tmp))
    with open(SNAPSHOT, "w") as f:
        json.dump(snap, f, indent=1, sort_keys=True)
        f.write("\n")
    print(SNAPSHOT)


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        record()
