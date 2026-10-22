"""S2 v1 release bundle: a golden encode and decode gate on the released
bundle, a golden store read back exactly, and a SAFE round trip.

The tests need the released bundle and skip when it is absent. They look
in GEORVQ_S2_V1_BUNDLE, GEORVQ_S2_BUNDLE, the bundle cache entry for the
golden bitstream id (georvq.inference.bundle_resolve.bundle_path,
~/.cache/georvq/bundles/<bitstream_id> by default), then the
maintainer's copy.

Inputs. golden/s2_v1_tile_inputs.npz is a 144 x 144 px crop (at 10 m;
72 px at 20 m, 24 px at 60 m) of the 13 bands of the L1C product
S2C_MSIL1C_20260503T084641_N0512_R107_T37UDB_20260503T104912, cut where
the swath edge crosses it, so part of it is nodata (DN 0). Its DN keep
the product's -1000 radiometric offset convention (processing baseline
05.12).

golden/s2_v1_golden.npz holds the records the bundle's encoder produced
for the crop on CPU, and sha256 hashes of the float32 DN field decoded
from them and of the uint16 output, per grid, at four rungs.
golden/s2_v1_tile.georvq.zarr is the crop written as an S2 v1 store with
small chunks, read back through georvq.open.

Everything is compared exactly. The hashes were recorded on CPU in fp32
(cirrus, x86-64) with torch using THREADS threads, which the tests set,
because the float bits of the 60 m head's output were found to depend on
the thread count (the uint16 output did not). A CPU with other vector
kernels could produce different float bits, in which case the test
reports the DN RMSE next to the recorded one. Re-record with

    python -m tests.test_s2_v1_bundle inputs   # needs the SAFE product
    python -m tests.test_s2_v1_bundle record
"""
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

pytest.importorskip("rasterio")
pytest.importorskip("zarr")

import georvq  # noqa: E402
from georvq.s2 import safe as S  # noqa: E402
from georvq.s2 import v1  # noqa: E402

HERE = os.path.dirname(__file__)
INPUTS = os.path.join(HERE, "golden/s2_v1_tile_inputs.npz")
GOLDEN = os.path.join(HERE, "golden/s2_v1_golden.npz")
TILE = os.path.join(HERE, "golden/s2_v1_tile.georvq.zarr")
LOCAL_BUNDLE = "/data/georvq-releases/s2-v1/bundle"
CKPT = os.environ.get(
    "GEORVQ_S2_V1_CKPT", "/home/ali/projects/georvq-release-format/"
    "checkpoints/beta_arch_w36/frozen_42500.pt")
CHANNEL = os.environ.get("GEORVQ_S2_V1_CHANNEL",
                         "/data/georvq-releases/s2-v1/stats_channel.npz")
GOLDEN_RUNGS = ((8, 0), (8, 1), (8, 4), (8, 8))
THREADS = 8
N = 144
TILE_CHUNK_PX = {"base": 72, "tier1": 72}
SAFE = ("/mnt/raid/data/s2get/Sentinel-2/MSI/L1C/2026/05/03/"
        "S2C_MSIL1C_20260503T084641_N0512_R107_T37UDB_20260503T104912.SAFE")


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _golden_bid():
    if os.path.exists(GOLDEN):
        with np.load(GOLDEN) as g:
            return str(g["bitstream_id"])
    return None


def _find_bundle():
    """GEORVQ_S2_V1_BUNDLE as given; the other places only when their
    bundle.json names the golden bitstream id (any S2 v1 bundle when the
    golden file is not recorded yet)."""
    from georvq.inference.bundle_resolve import bundle_path
    if os.environ.get("GEORVQ_S2_V1_BUNDLE"):
        return os.environ["GEORVQ_S2_V1_BUNDLE"]
    bid = _golden_bid()
    for d in (os.environ.get(v1.ENV_BUNDLE),
              bundle_path(bid) if bid else None, LOCAL_BUNDLE):
        meta = os.path.join(d or "", "bundle.json")
        if d and os.path.exists(meta) and (
                bid is None or json.load(open(meta)).get("bitstream_id") == bid):
            return d
    return None


BUNDLE = _find_bundle()


def _inputs():
    z = np.load(INPUTS)
    return (tuple(z[f"b{g}"] for g, _, _ in S.GRIDS),
            json.loads(str(z["meta"])))


def _bundle_codec():
    if BUNDLE is None or not os.path.exists(os.path.join(BUNDLE, "bundle.json")):
        pytest.skip("S2 v1 bundle not present (set GEORVQ_S2_V1_BUNDLE or "
                    "GEORVQ_S2_BUNDLE, or place it in the bundle cache)")
    from georvq.inference.store_codec import quiet_build
    with quiet_build():
        return v1.load_bundle(BUNDLE, "cpu")


def _window_valid(blk, W):
    v = np.zeros((blk.grid[0] * W, blk.grid[1] * W), bool)
    v[:blk.valid.shape[0], :blk.valid.shape[1]] = blk.valid
    return np.stack([v[r * W:(r + 1) * W, c * W:(c + 1) * W]
                     for r, c in zip(blk.rows, blk.cols)])


def _assemble(windows, blk, f, W):
    n, C, w, _ = windows.shape
    gh, gw = blk.grid
    full = np.zeros((C, gh * w, gw * w), windows.dtype)
    for k, (r, c) in enumerate(zip(blk.rows, blk.cols)):
        full[:, r * w:(r + 1) * w, c * w:(c + 1) * w] = windows[k]
    return full[:, :N // f, :N // f]


def _decode_all(codec, blk):
    """{rung: ({grid: float32 DN field (n, C, w, w)}, {grid: uint16
    output (C, h, w)})} on CPU from the encoder's records."""
    rec = blk.records
    tiers = [rec[n].transpose(1, 0, 2) for n in codec.names[1:]]
    vw = _window_valid(blk, codec.window)
    out = {}
    for r in GOLDEN_RUNGS:
        f, _ = codec.decode_field(rec["base"][0], tiers, r, vw)
        o = codec.decode_records(rec["base"][0], tiers, r, vw)
        out[r] = ({g: f[g].cpu().numpy() for g in f},
                  {g: _assemble(o[g], blk, ff, codec.window)
                   for g, ff, _ in S.GRIDS})
    return out


def _rmse(out, arrays):
    """DN RMSE per grid over pixels valid in both."""
    res = {}
    for (g, _, _), a in zip(S.GRIDS, arrays):
        m = (out[g] != 0) & (a != 0)
        res[g] = float(np.sqrt(((out[g].astype(float) - a) ** 2)[m].mean()))
    return res


@pytest.fixture(scope="module", autouse=True)
def _threads():
    n = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def released():
    codec = _bundle_codec()
    if not os.path.exists(GOLDEN):
        pytest.skip("golden codes not recorded")
    arrays, meta = _inputs()
    torch.set_grad_enabled(False)
    try:
        blk = codec.encode_block(arrays, stats=True)
    finally:
        torch.set_grad_enabled(True)
    return codec, arrays, meta, blk


def test_released_bundle_decodes_like_its_checkpoint(released):
    codec, arrays, _, blk = released
    if not (os.path.exists(CKPT) and os.path.exists(CHANNEL)):
        pytest.skip("source checkpoint not present (set GEORVQ_S2_V1_CKPT "
                    "and GEORVQ_S2_V1_CHANNEL)")
    from georvq.inference.store_codec import quiet_build
    with quiet_build():
        ref = v1.S2V1Codec.load(CKPT, CHANNEL, "cpu")
    assert (ref.bitstream_id, ref.decoder_id) == (codec.bitstream_id,
                                                  codec.decoder_id)
    torch.set_grad_enabled(False)
    try:
        got = ref.encode_block(arrays, stats=True)
        for k in blk.records:
            assert np.array_equal(got.records[k], blk.records[k]), k
    finally:
        torch.set_grad_enabled(True)


def test_golden_codes(released):
    codec, _, _, blk = released
    g = np.load(GOLDEN)
    assert codec.bitstream_id == str(g["bitstream_id"])
    assert codec.decoder_id == str(g["decoder_id"])
    assert np.array_equal(blk.rows, g["rows"])
    assert np.array_equal(blk.cols, g["cols"])
    assert np.array_equal(blk.valid, g["valid"])
    assert sorted(blk.records) == sorted(k[4:] for k in g.files
                                         if k.startswith("rec_"))
    for k in blk.records:
        assert np.array_equal(blk.records[k], g[f"rec_{k}"]), k


def test_golden_decode(released):
    codec, arrays, _, blk = released
    g = np.load(GOLDEN)
    for k in blk.records:
        blk.records[k] = g[f"rec_{k}"]
    torch.set_grad_enabled(False)
    try:
        dec = _decode_all(codec, blk)
    finally:
        torch.set_grad_enabled(True)
    rec_rmse = json.loads(str(g["rmse"]))
    for i, r in enumerate(GOLDEN_RUNGS):
        f, o = dec[r]
        name = "-".join(map(str, r))
        for j, (gn, _, _) in enumerate(S.GRIDS):
            assert _sha(f[gn]) == str(g["sha_field"][i, j]), (
                f"{name} {gn}: float field differs; DN RMSE {_rmse(o, arrays)}"
                f" vs recorded {rec_rmse[name]}")
            assert _sha(o[gn]) == str(g["sha_out"][i, j]), (name, gn)


def test_golden_tile_reads_back(released):
    codec, arrays, _, blk = released
    if not os.path.exists(TILE):
        pytest.skip("tile fixture not recorded")
    g = np.load(GOLDEN)
    with georvq.open(TILE, codec=codec) as ds:
        h = ds.header
        assert h["codec"]["bitstream_id"] == codec.bitstream_id
        assert h["licence"]["attribution"] == \
            "Contains modified Copernicus Sentinel data 2026"
        assert (ds.product_height, ds.product_width) == (N, N)
        for i, r in enumerate(GOLDEN_RUNGS):
            got = ds.read_grids(quality=r)
            for j, (gn, _, _) in enumerate(S.GRIDS):
                assert _sha(got[gn]) == str(g["sha_out"][i, j]), (r, gn)
                assert np.array_equal(ds.grid(gn).read(quality=r), got[gn])


def test_safe_round_trip_with_the_bundle(released, tmp_path):
    from tests.s2_safe_fixture import make_safe
    codec, arrays, meta, _ = released
    safe = make_safe(str(tmp_path / "src"), arrays, meta["transform"],
                     epsg=meta["epsg"])
    store = str(tmp_path / "p.georvq.zarr")
    torch.set_grad_enabled(False)
    try:
        v1.encode_safe(safe, store, codec, chunk_px=TILE_CHUNK_PX)
        root = v1.reconstruct_safe(store, str(tmp_path / "out"),
                                   quality="8-4", codec=codec)
        with georvq.open(store, codec=codec) as ds:
            want = ds.read_grids(quality=(8, 4))
    finally:
        torch.set_grad_enabled(True)
    import rasterio
    files = S.band_files(safe)
    for g, _, names in S.GRIDS:
        for i, b in enumerate(names):
            with rasterio.open(os.path.join(root, files[b])) as ds, \
                    rasterio.open(os.path.join(safe, files[b])) as d0:
                assert ds.transform == d0.transform and ds.crs == d0.crs
                assert np.array_equal(ds.read(1),
                                      np.minimum(want[g][i], 32767)), b
    man = open(os.path.join(root, "manifest.safe")).read()
    for rel, (size, name, h) in S.manifest_entries(man).items():
        p = os.path.join(root, rel)
        assert size == os.path.getsize(p) and h == S.file_checksum(p, name)


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

def write_tile(codec, out, arrays, meta):
    """Encode the crop to the store `out` as the golden tile is written."""
    from tests.s2_safe_fixture import make_safe
    with tempfile.TemporaryDirectory() as tmp:
        safe = make_safe(tmp, arrays, meta["transform"], epsg=meta["epsg"])
        v1.encode_safe(safe, out, codec, chunk_px=TILE_CHUNK_PX,
                       release="golden", source="golden/" +
                       os.path.basename(safe))


def inputs():
    """Cut a 144 px crop of the product across its swath edge: the first,
    in row-major order of a 144 px grid, whose 10 m pixels are between 30
    and 60 percent nodata."""
    import rasterio
    from rasterio.windows import Window
    files = S.band_files(SAFE)
    with rasterio.open(os.path.join(SAFE, files["B02"])) as ds:
        b02 = ds.read(1)
        t = ds.transform
        epsg = ds.crs.to_epsg()
    pick = None
    for y in range(0, b02.shape[0] - N + 1, N):
        for x in range(0, b02.shape[1] - N + 1, N):
            z = (b02[y:y + N, x:x + N] == 0).mean()
            if 0.3 <= z <= 0.6:
                pick = (y, x)
                break
        if pick:
            break
    y0, x0 = pick
    out = {}
    for g, f, names in S.GRIDS:
        a = []
        for b in names:
            with rasterio.open(os.path.join(SAFE, files[b])) as ds:
                a.append(ds.read(1, window=Window(x0 // f, y0 // f, N // f,
                                                  N // f)))
        out[f"b{g}"] = np.stack(a)
    tr = [t.a, t.b, t.c + x0 * t.a, t.d, t.e, t.f + y0 * t.e]
    meta = {"safe": os.path.basename(SAFE), "rows": [y0, y0 + N],
            "cols": [x0, x0 + N], "epsg": epsg, "transform": tr,
            "nodata_fraction_b02": float((out["b10m"][0] == 0).mean())}
    np.savez_compressed(INPUTS, **out, meta=np.array(json.dumps(meta)))
    print(INPUTS, os.path.getsize(INPUTS), meta)


def record():
    import shutil
    torch.set_grad_enabled(False)
    torch.set_num_threads(THREADS)
    codec = _bundle_codec()
    arrays, meta = _inputs()
    blk = codec.encode_block(arrays, stats=True)
    dec = _decode_all(codec, blk)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "tile.georvq.zarr")
        write_tile(codec, out, arrays, meta)
        with georvq.open(out, codec=codec) as ds:
            reads = {r: ds.read_grids(quality=r) for r in GOLDEN_RUNGS}
        if os.path.exists(TILE):
            shutil.move(TILE, os.path.join(tmp, "old.georvq.zarr"))
        shutil.copytree(out, TILE)
    grids = [g for g, _, _ in S.GRIDS]
    rec = {f"rec_{k}": v for k, v in blk.records.items()}
    rmse = {"-".join(map(str, r)): _rmse(dec[r][1], arrays)
            for r in GOLDEN_RUNGS}
    np.savez_compressed(
        GOLDEN, **rec, rows=blk.rows, cols=blk.cols, valid=blk.valid,
        bitstream_id=np.array(codec.bitstream_id),
        decoder_id=np.array(codec.decoder_id), rungs=np.array(GOLDEN_RUNGS),
        sha_field=np.array([[_sha(dec[r][0][g]) for g in grids]
                            for r in GOLDEN_RUNGS]),
        sha_out=np.array([[_sha(dec[r][1][g]) for g in grids]
                          for r in GOLDEN_RUNGS]),
        rmse=np.array(json.dumps(rmse)))
    for r in GOLDEN_RUNGS:
        same = all(np.array_equal(reads[r][g], dec[r][1][g]) for g in grids)
        print("-".join(map(str, r)), rmse["-".join(map(str, r))],
              "read == decode" if same else "read differs from decode")
    print(GOLDEN, os.path.getsize(GOLDEN), TILE)


if __name__ == "__main__":
    if sys.argv[1:] == ["inputs"]:
        inputs()
    elif sys.argv[1:] == ["record"]:
        record()
