"""The AE v2 store format, held fixed while the release code changes.

Two checks:

- A store written by `georvq.ae.v2.encode_tile` for a tiny AE-shaped
  codec is compared with golden/ae_v2_store_snapshot.json: every node's
  attributes and array metadata as JSON, the stored keys, and the sha256
  of every chunk's decoded bytes. The codec's records come from an
  integer function of the source pixels and its ids are fixed strings,
  so the snapshot does not depend on float arithmetic and holds on any
  CPU and torch version; the model's numerics are pinned by
  test_ae_v2_bundle. Values that depend on the GDAL, PROJ or rasterio
  version (WKT strings, PROJJSON, the source profile under `georvq_aux`)
  are left out, and `georvq_aux` is checked for its keys only.
- With the released bundle present, the golden tile's windows are
  encoded again on CPU and the store is compared file by file with
  golden/ae_v2_tile.georvq.zarr, the header's `product.source` aside.

Re-record both with

    python -m tests.test_ae_v2_format record
"""
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
zarr = pytest.importorskip("zarr")
from rasterio.transform import Affine  # noqa: E402

from georvq.ae import v2  # noqa: E402
from georvq.config import load_config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference.mean_codec import RVQPCAMeanChannel  # noqa: E402

HERE = os.path.dirname(__file__)
SNAPSHOT = os.path.join(HERE, "golden/ae_v2_store_snapshot.json")
H, WD, WIN = 200, 168, 16
CHUNK_PX = {"base": 128, "mid": 64, "fine": 32}
BITSTREAM_ID, DECODER_ID = "0123456789abcdef", "fedcba9876543210"
# attributes whose text comes from GDAL, PROJ or rasterio
ENV_KEYS = ("proj:wkt2", "proj:projjson", "crs_wkt")
AUX_KEYS = ["band_tags", "block_shapes", "descriptions", "image_structure",
            "orientation", "overviews", "profile", "tags"]


def _channel(bands=64):
    """A fixed RVQ-PCA mean channel: 2 x 16 codes and 40 PCA bits."""
    rng = np.random.default_rng(0)
    books = rng.integers(-8, 9, size=(2, 16, bands)).astype(np.float32) / 64
    bits = np.zeros(bands, np.int64)
    bits[:8], bits[8:16] = 3, 2
    return RVQPCAMeanChannel(books, np.zeros(bands), np.eye(bands),
                             np.full(bands, 0.1), bits, np.full(bands, 2.0))


def _codec():
    cfg = load_config("configs/smoke.yaml", [
        "model.bands=64", f"model.window={WIN}",
        "model.tiers=['1:16:32:3','2:8:32:2','4:8:16:2']",
        "model.min_depths=[1,0,0]",
        "model.stats_cond={subtract_mean: true, cond_encoder: true, "
        "cond_decoder: true, bits_per_band: 6}"])
    torch.manual_seed(0)
    codec = v2.AEV2Codec(build_model(cfg, "cpu"), _channel())
    codec.bitstream_id, codec.decoder_id = BITSTREAM_ID, DECODER_ID
    lay = codec.layout(quality=True)
    proj = np.random.default_rng(1).integers(1, 1 << 20, size=(64 * WIN * WIN, 64))

    def encode_windows(raw, stats=True):
        """Records as an integer function of each window's pixels."""
        raw = raw[0] if isinstance(raw, tuple) else raw
        B = raw.shape[0]
        h = raw.reshape(B, -1).astype(np.int64) @ proj        # (B, 64)

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


def _source(path):
    """A bottom-up AE-like tile with nodata in two blocks and in all of
    the top-left fine chunk (an empty chunk)."""
    rng = np.random.default_rng(2)
    north = rng.integers(-127, 128, size=(64, H, WD), dtype=np.int8)
    north[:, :30, :50] = v2.NODATA
    north[:, 120:140, 100:] = v2.NODATA
    north[:, :32, :32] = v2.NODATA
    t = Affine(10.0, 0.0, 500000.0, 0.0, 10.0, 4000000.0 - 10.0 * H)
    with rasterio.open(path, "w", driver="GTiff", width=WD, height=H,
                       count=64, dtype="int8", nodata=v2.NODATA,
                       crs="EPSG:32633", transform=t, tiled=True,
                       blockxsize=64, blockysize=64, compress="deflate",
                       interleave="band") as ds:
        ds.write(north[:, ::-1])
        for b in range(1, 65):
            ds.set_band_description(b, f"A{b - 1:02d}")


def _strip(d):
    """`d` without the attributes whose text depends on the environment."""
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k not in ENV_KEYS}
    if isinstance(d, list):
        return [_strip(v) for v in d]
    return d


def snapshot(path) -> dict:
    """Attributes and metadata of every node, the stored keys, and the
    sha256 of the decoded bytes of every chunk of every array."""
    nodes, chunks, keys = {}, {}, []
    for r, _, files in os.walk(path):
        for f in files:
            rel = os.path.relpath(os.path.join(r, f), path)
            keys.append(rel.replace(os.sep, "/"))
            if f != "zarr.json":
                continue
            meta = json.load(open(os.path.join(r, f)))
            node = os.path.relpath(r, path).replace(os.sep, "/")
            attrs = meta.get("attributes", {})
            aux = attrs.pop("georvq_aux", None)
            if aux is not None:
                attrs["georvq_aux"] = sorted(aux)
            meta.pop("consolidated_metadata", None)
            nodes[node] = _strip(meta)
    g = zarr.open_group(path, mode="r")

    def walk(grp, prefix):
        for name, a in grp.arrays():
            full = prefix + name
            c = a.chunks
            grid = [range(0, s, k) for s, k in zip(a.shape, c)]
            out = {}
            for idx in np.ndindex(*[len(x) for x in grid]):
                sl = tuple(slice(x[i], x[i] + k) for x, i, k in zip(grid, idx, c))
                out[".".join(map(str, idx))] = hashlib.sha256(
                    np.ascontiguousarray(a[sl]).tobytes()).hexdigest()
            chunks[full] = out
        for name, sub in grp.groups():
            walk(sub, prefix + name + "/")
    walk(g, "")
    return {"nodes": nodes, "keys": sorted(keys), "chunks": chunks}


def _write_store(tmp):
    src = os.path.join(tmp, "snap.tif")
    out = os.path.join(tmp, "snap.georvq.zarr")
    _source(src)
    v2.encode_tile(src, out, _codec(), stats=True, chunk_px=CHUNK_PX,
                   strip_px=48, batch=37, source="snap.tif", revision="r1")
    return out


def test_store_format_matches_the_snapshot(tmp_path):
    got = snapshot(_write_store(str(tmp_path)))
    want = json.load(open(SNAPSHOT))
    assert got["keys"] == want["keys"]
    assert set(got["nodes"]) == set(want["nodes"])
    for node in want["nodes"]:
        assert got["nodes"][node] == want["nodes"][node], node
    assert got["nodes"]["."]["attributes"]["georvq_aux"] == AUX_KEYS
    assert got["chunks"] == want["chunks"]


# --------------------------------------------------------------------------
# the golden tile, encoded again with the released bundle
# --------------------------------------------------------------------------

def _files(d) -> dict:
    out = {}
    for r, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(r, f)
            out[os.path.relpath(p, d).replace(os.sep, "/")] = open(p, "rb").read()
    return out


def _without_source(raw: bytes) -> dict:
    meta = json.loads(raw)
    meta["attributes"]["georvq"]["product"].pop("source")
    return meta


def _encode_golden(out):
    from tests.test_ae_v2_bundle import (TILE_CHUNK_PX, _bundle_codec,
                                         _windows, _write_tile)
    codec = _bundle_codec()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "golden_tile.tif")
        _write_tile(_windows(), src, codec.window)
        v2.encode_tile(src, out, codec, stats=True, chunk_px=TILE_CHUNK_PX,
                       release="golden", source="golden_tile.tif")


def test_golden_tile_encodes_to_the_same_store(tmp_path):
    from tests.test_ae_v2_bundle import TILE
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
        if k == "zarr.json":
            assert _without_source(got[k]) == _without_source(want[k])
        else:
            assert got[k] == want[k], k


def record():
    import shutil
    from tests.test_ae_v2_bundle import TILE
    with tempfile.TemporaryDirectory() as tmp:
        snap = snapshot(_write_store(tmp))
    with open(SNAPSHOT, "w") as f:
        json.dump(snap, f, indent=1, sort_keys=True)
        f.write("\n")
    torch.set_grad_enabled(False)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "golden_tile.georvq.zarr")
        _encode_golden(out)
        if os.path.exists(TILE):
            shutil.move(TILE, os.path.join(tmp, "old.georvq.zarr"))
        shutil.copytree(out, TILE)
    print(SNAPSHOT, TILE)


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        record()
