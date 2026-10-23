"""The Tessera v2 store format, held fixed while the release code changes.

Two checks, as tests/test_ae_v2_format.py makes them for AE v2:

- A store written by `georvq.tessera.v2.encode_tile` for a tiny
  Tessera-shaped codec is compared with
  golden/tessera_v2_store_snapshot.json: every node's attributes and
  array metadata as JSON, the stored keys, and the sha256 of every
  chunk's decoded bytes. The codec's records are an integer function of
  the source codes and scale bits, and its ids are fixed strings, so the
  snapshot does not depend on float arithmetic and holds on any CPU and
  torch version; the model's numerics are pinned by
  test_tessera_v2_bundle. The tile is not a multiple of the window, has
  nodata in several conventions and one fine chunk with no valid pixel.
  Values that depend on the GDAL, PROJ or rasterio version (WKT strings,
  PROJJSON) are left out.
- With the released bundle present, the golden tile is encoded again on
  CPU and the store is compared file by file with
  golden/tessera_v2_tile.georvq.zarr.

Re-record both with

    python -m tests.test_tessera_v2_format record
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")
pytest.importorskip("zarr")

from georvq.config import load_config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference.mean_codec import RVQPCAMeanChannel  # noqa: E402
from georvq.tessera import v2  # noqa: E402
from tests.test_ae_v2_format import _files, snapshot  # noqa: E402

HERE = os.path.dirname(__file__)
SNAPSHOT = os.path.join(HERE, "golden/tessera_v2_store_snapshot.json")
H, WD, WIN = 75, 90, 16
YEAR, LON, LAT = 2024, 17.25, 46.75
CHUNK_PX = {"base": 64, "mid": 48, "fine": 32}
BITSTREAM_ID, DECODER_ID = "0123456789abcdef", "fedcba9876543210"
AUX = {"geotessera": {
    "year": YEAR, "lon": LON, "lat": LAT,
    "embedding": "snap/global_0.1_degree_representation/2024/"
                 "grid_17.25_46.75/grid_17.25_46.75.npy",
    "scales": "snap/global_0.1_degree_representation/2024/"
              "grid_17.25_46.75/grid_17.25_46.75_scales.npy",
    "georef": "landmask", "rule_agrees": None}}


def _channel(bands=128):
    """A fixed RVQ-PCA mean channel: 2 x 16 codes and 40 PCA bits."""
    rng = np.random.default_rng(0)
    books = rng.integers(-8, 9, size=(2, 16, bands)).astype(np.float32) / 64
    bits = np.zeros(bands, np.int64)
    bits[:8], bits[8:16] = 3, 2
    return RVQPCAMeanChannel(books, np.zeros(bands), np.eye(bands),
                             np.full(bands, 0.1), bits, np.full(bands, 2.0))


def _codec():
    cfg = load_config("configs/smoke.yaml", [
        "data.source=tessera", "model.bands=128", f"model.window={WIN}",
        "model.tiers=['1:16:32:3','2:8:32:2','4:8:16:2']",
        "model.min_depths=[2,0,0]",
        "model.stats_cond={subtract_mean: true, cond_encoder: true, "
        "cond_decoder: true, bits_per_band: 6}"])
    torch.manual_seed(0)
    codec = v2.TesseraV2Codec(build_model(cfg, "cpu"), _channel())
    codec.bitstream_id, codec.decoder_id = BITSTREAM_ID, DECODER_ID
    lay = codec.layout(quality=True)
    rng = np.random.default_rng(1)
    proj_q = rng.integers(1, 1 << 20, size=(128 * WIN * WIN, 64))
    proj_s = rng.integers(1, 1 << 20, size=(WIN * WIN, 64))

    def encode_windows(raw, stats=True):
        """Records as an integer function of each window's codes and the
        bits of its scales."""
        q, s = raw
        B = q.shape[0]
        sb = np.ascontiguousarray(s, np.float32).view(np.int32) \
            .reshape(B, -1).astype(np.int64) % 997
        h = q.reshape(B, -1).astype(np.int64) @ proj_q + sb @ proj_s

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


def _source(root):
    """A geotessera tile in a local copy of the bucket's layout, with a
    landmask of its size: nodata as all-zero vectors, as NaN, infinite and
    zero scales, and in all of the top-right fine chunk."""
    rng = np.random.default_rng(2)
    q = rng.integers(-127, 128, size=(H, WD, 128), dtype=np.int8)
    s = rng.uniform(0.001, 0.05, size=(H, WD)).astype(np.float32)
    q[:6, :10] = 0
    s[50:52, 20:40] = np.nan
    s[60, 3] = np.inf
    s[30:35, 30:34] = 0.0
    q[:32, 64:] = 0
    name = v2.tile_name(LON, LAT)
    d = os.path.join(root, v2.EMBEDDINGS_DIR, str(YEAR), name)
    os.makedirs(d)
    np.save(os.path.join(d, f"{name}.npy"), q)
    np.save(os.path.join(d, f"{name}_scales.npy"), s)
    lm = os.path.join(root, v2.LANDMASK_DIR)
    os.makedirs(lm)
    epsg, t, _ = v2.rule_georef(LON, LAT)
    with rasterio.open(os.path.join(lm, f"{name}.tiff"), "w", driver="GTiff",
                       height=H, width=WD, count=1, dtype="uint8",
                       crs=f"EPSG:{epsg}", transform=rasterio.Affine(*t)) as ds:
        ds.write(np.ones((1, H, WD), np.uint8))
    return v2.GeotesseraTile(YEAR, LON, LAT, root=root, check_rule=False,
                             source_root="snap")


def _write_store(tmp):
    out = os.path.join(tmp, "snap.georvq.zarr")
    tile = _source(os.path.join(tmp, "bucket"))
    v2.encode_product(tile, out, _codec(), stats=True, chunk_px=CHUNK_PX,
                      strip_px=32, batch=5, release=v2.RELEASE_NAME,
                      uri=v2.RELEASE_URI, revision="r1")
    return out


def test_store_format_matches_the_snapshot(tmp_path):
    out = _write_store(str(tmp_path))
    got = snapshot(out)
    want = json.load(open(SNAPSHOT))
    assert got["keys"] == want["keys"]
    assert set(got["nodes"]) == set(want["nodes"])
    for node in want["nodes"]:
        assert got["nodes"][node] == want["nodes"][node], node
    assert got["chunks"] == want["chunks"]
    meta = json.load(open(os.path.join(out, "zarr.json")))
    assert meta["attributes"]["georvq_aux"] == AUX


# --------------------------------------------------------------------------
# the golden tile, encoded again with the released bundle
# --------------------------------------------------------------------------

def _encode_golden(out):
    from tests.test_tessera_v2_bundle import _bundle_codec, _inputs, write_tile
    codec = _bundle_codec()
    q, s, meta = _inputs()
    write_tile(codec, out, q, s, meta)


def test_golden_tile_encodes_to_the_same_store(tmp_path):
    from tests.test_tessera_v2_bundle import TILE
    if not os.path.exists(TILE):
        pytest.skip("tile fixture not recorded")
    out = str(tmp_path / "golden_tile.georvq.zarr")
    torch.set_grad_enabled(False)
    try:
        _encode_golden(out)
    finally:
        torch.set_grad_enabled(True)
    want, got = _files(TILE), _files(out)
    assert sorted(got) == sorted(want)
    for k in want:
        assert got[k] == want[k], k


def record():
    with tempfile.TemporaryDirectory() as tmp:
        snap = snapshot(_write_store(tmp))
    with open(SNAPSHOT, "w") as f:
        json.dump(snap, f, indent=1, sort_keys=True)
        f.write("\n")
    print(SNAPSHOT)


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        record()
