"""Tessera v1.1 release bundle: a golden encode and decode gate on the
released bundle, and a golden store read back exactly.

The tests need the released bundle and skip when it is absent. They look
in GEORVQ_TESSERA_V1_1_BUNDLE, GEORVQ_TESSERA_BUNDLE, the bundle cache
entry for the golden bitstream id (georvq.inference.bundle_resolve.
bundle_path, ~/.cache/georvq/bundles/<bitstream_id> by default), then the
maintainer's copy.

Inputs. golden/tessera_v1_1_tile_inputs.npz is a 100 x 120 px crop of
held-out chip 752 of /data/tessera-samples-v11/chips_512 (validation
split of the training run, val_frac 0.04 seed 0; 2019, near 31.75 N,
108.35 E): int8 (128, H, W) codes and a float32 scale plane (the chip
store's float16 scales, widened), georeferenced on its zone's grid. It
is about 45% nodata.

golden/tessera_v1_1_golden.npz holds the records the bundle's encoder
produced for the crop on CPU, and sha256 hashes of the float32 field
decoded from them and of the float32 output pixels at four rungs.
golden/tessera_v1_1_tile.georvq.zarr is the crop written as a Tessera
v1.1 store with small chunks (a region of its zone's grid, the product
a shard store codes), read back through georvq.open.

Everything is compared exactly. The hashes were recorded on CPU in fp32
(cirrus, x86-64); a CPU with other vector kernels could produce
different float bits, in which case the test reports the relative L2
error against the source next to the recorded one so the size of the
difference is visible. Re-record with

    python -m tests.test_tessera_v1_1_bundle inputs   # needs chips_512
    python -m tests.test_tessera_v1_1_bundle record
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
pytest.importorskip("pyproj")

import georvq  # noqa: E402
from georvq.tessera import tiles, v1_1  # noqa: E402

HERE = os.path.dirname(__file__)
INPUTS = os.path.join(HERE, "golden/tessera_v1_1_tile_inputs.npz")
GOLDEN = os.path.join(HERE, "golden/tessera_v1_1_golden.npz")
TILE = os.path.join(HERE, "golden/tessera_v1_1_tile.georvq.zarr")
# maintainer's copy, tried after GEORVQ_TESSERA_V1_1_BUNDLE,
# GEORVQ_TESSERA_BUNDLE and the bundle cache entry
LOCAL_BUNDLE = "/data/georvq-releases/tessera-v11/bundle"
CKPT = os.environ.get(
    "GEORVQ_TESSERA_V1_1_CKPT",
    "/data/georvq-releases/tessera-v11/tessera_v11_step133000.pt")
MEAN_NPZ = os.environ.get(
    "GEORVQ_TESSERA_V1_1_MEAN",
    "/data/georvq-releases/tessera-v11/mean_channel.npz")
GOLDEN_RUNGS = ((4, 0, 0), (20, 0, 0), (20, 13, 0), (20, 13, 12))
# The release's 2048 px base chunks would make the fixture mostly
# padding. These keep it small and split the fine tier over four chunks.
TILE_CHUNK_PX = {"base": 128, "mid": 128, "fine": 64}
CHIP = {"store": "/data/tessera-samples-v11/chips_512", "chip_id": 752,
        "rows": (336, 436), "cols": (160, 280)}


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _golden_bid():
    if os.path.exists(GOLDEN):
        with np.load(GOLDEN) as g:
            return str(g["bitstream_id"])
    return None


def _find_bundle():
    """GEORVQ_TESSERA_V1_1_BUNDLE as given; the other places only when their
    bundle.json names the golden bitstream id (any Tessera v1.1 bundle when
    the golden file is not recorded yet)."""
    from georvq.inference.bundle_resolve import bundle_path
    if os.environ.get("GEORVQ_TESSERA_V1_1_BUNDLE"):
        return os.environ["GEORVQ_TESSERA_V1_1_BUNDLE"]
    bid = _golden_bid()
    for d in (os.environ.get(v1_1.ENV_BUNDLE),
              bundle_path(bid) if bid else None, LOCAL_BUNDLE):
        meta = os.path.join(d or "", "bundle.json")
        if not (d and os.path.exists(meta)):
            continue
        m = json.load(open(meta))
        # the tests encode, so a decoder-only bundle (as a reader's cache
        # holds) does not serve
        if (bid is None or m.get("bitstream_id") == bid) and \
                m.get("variant") != "decoder":
            return d
    return None


BUNDLE = _find_bundle()


def _inputs():
    z = np.load(INPUTS)
    return z["embedding"], z["scales"], json.loads(str(z["meta"]))


def _tile(q, s, meta):
    """The crop as a region of its zone's grid, the product a Tessera v1.1
    store codes (a shard, normally)."""
    R, C = meta["zone_window"][:2]
    tr = meta["transform"]
    return v1_1.TesseraShard.from_arrays(
        meta["year"], meta["zone"], q, s, meta["epsg"], tr[2] - C * tr[0],
        tr[5] - R * tr[4], tr[0], R, C, meta["source"])


def _rel_l2(out, q, s, valid):
    src = q.astype(np.float32) * s[None]
    e = np.sqrt(((out - src) ** 2).sum(0)) / np.maximum(
        np.sqrt((src ** 2).sum(0)), 1e-6)
    return float(e[valid].mean())


def _assemble(windows, rows, cols, grid, shape):
    """(n, C, W, W) coded windows -> (C, H, W) north-up, zero elsewhere."""
    n, C, W, _ = windows.shape
    gh, gw = grid
    full = np.zeros((C, gh * W, gw * W), np.float32)
    for k, (r, c) in enumerate(zip(rows, cols)):
        full[:, r * W:(r + 1) * W, c * W:(c + 1) * W] = windows[k]
    return full[:, :shape[0], :shape[1]]


def _bundle_codec():
    if BUNDLE is None or not os.path.exists(os.path.join(BUNDLE, "bundle.json")):
        pytest.skip("Tessera v1.1 bundle not present (set "
                    "GEORVQ_TESSERA_V1_1_BUNDLE or GEORVQ_TESSERA_BUNDLE, or "
                    "place it in the bundle cache)")
    from georvq.inference.store_codec import quiet_build
    with quiet_build():
        return v1_1.load_bundle(BUNDLE, "cpu")


def _encode(codec, q, s):
    """BlockRecords of the crop, as encode_shard computes them."""
    return codec.encode_block((np.ascontiguousarray(q), s), stats=False)


def _decode_all(codec, blk):
    """{rung: (float32 field (n, C, W, W), float32 output (C, H, W))} on
    CPU from the encoder's records."""
    rec = blk.records
    tiers = [rec[n].transpose(1, 0, 2) for n in codec.names[1:]]
    W = codec.window
    v = np.zeros((blk.grid[0] * W, blk.grid[1] * W), bool)
    v[:blk.valid.shape[0], :blk.valid.shape[1]] = blk.valid
    vw = np.stack([v[r * W:(r + 1) * W, c * W:(c + 1) * W]
                   for r, c in zip(blk.rows, blk.cols)])
    out = {}
    for r in GOLDEN_RUNGS:
        f, _ = codec.decode_field(rec["base"][0], tiers, r, vw)
        o = codec.decode_records(rec["base"][0], tiers, r, vw)
        out[r] = (f.cpu().numpy(),
                  _assemble(o, blk.rows, blk.cols, blk.grid, blk.valid.shape))
    return out


@pytest.fixture(scope="module")
def released():
    codec = _bundle_codec()
    if not os.path.exists(GOLDEN):
        pytest.skip("golden codes not recorded")
    q, s, meta = _inputs()
    torch.set_grad_enabled(False)
    try:
        blk = _encode(codec, q, s)
    finally:
        torch.set_grad_enabled(True)
    return codec, q, s, blk


def test_golden_bundle_is_found_in_the_cache(tmp_path, monkeypatch):
    bid = _golden_bid()
    if bid is None:
        pytest.skip("golden codes not recorded")
    monkeypatch.delenv("GEORVQ_TESSERA_V1_1_BUNDLE", raising=False)
    monkeypatch.setenv("GEORVQ_CACHE", str(tmp_path))
    other = tmp_path / "other"
    other.mkdir()
    json.dump({"bitstream_id": "0" * 16}, open(other / "bundle.json", "w"))
    monkeypatch.setenv(v1_1.ENV_BUNDLE, str(other))
    entry = tmp_path / "bundles" / bid
    entry.mkdir(parents=True)
    json.dump({"bitstream_id": bid}, open(entry / "bundle.json", "w"))
    assert _find_bundle() == str(entry)


def test_released_bundle_decodes_like_its_checkpoint(released):
    codec, q, s, blk = released
    if not (os.path.exists(CKPT) and os.path.exists(MEAN_NPZ)):
        pytest.skip("source checkpoint not present (set "
                    "GEORVQ_TESSERA_V1_1_CKPT)")
    from georvq.inference.store_codec import quiet_build
    with quiet_build():
        ref = v1_1.TesseraV11Codec.load(CKPT, MEAN_NPZ, "cpu")
    assert (ref.bitstream_id, ref.decoder_id) == (codec.bitstream_id,
                                                  codec.decoder_id)
    sb = codec.model.state_dict()
    for k, v in ref.model.state_dict().items():
        if k.rsplit(".", 1)[-1] in ("ema_count", "ema_sum", "ema_sse", "step",
                                    "initialized"):
            continue
        assert torch.equal(v, sb[k]), k
    torch.set_grad_enabled(False)
    try:
        got = _encode(ref, q, s)
        for k in blk.records:
            assert np.array_equal(got.records[k], blk.records[k]), k
        rec = blk.records
        tiers = [rec[n].transpose(1, 0, 2) for n in codec.names[1:]]
        W = codec.window
        vw = np.ones((len(blk.rows), W, W), bool)
        for r in ((4, 0, 0), (20, 13, 12)):
            fa, _ = ref.decode_field(rec["base"][0], tiers, r, vw)
            fb, _ = codec.decode_field(rec["base"][0], tiers, r, vw)
            assert torch.equal(fa.view(torch.int32), fb.view(torch.int32)), r
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
    codec, q, s, blk = released
    g = np.load(GOLDEN)
    for k in blk.records:
        blk.records[k] = g[f"rec_{k}"]
    torch.set_grad_enabled(False)
    try:
        dec = _decode_all(codec, blk)
    finally:
        torch.set_grad_enabled(True)
    for i, r in enumerate(GOLDEN_RUNGS):
        f, o = dec[r]
        name = "-".join(map(str, r))
        assert _sha(f) == str(g["sha_field"][i]), (
            f"{name}: float field differs, relative L2 "
            f"{_rel_l2(o, q, s, blk.valid):.9f} vs recorded "
            f"{float(g['rel_l2'][i]):.9f}")
        assert _sha(o) == str(g["sha_out"][i]), name


def test_golden_tile_reads_back(released):
    codec, q, s, blk = released
    if not os.path.exists(TILE):
        pytest.skip("tile fixture not recorded")
    g = np.load(GOLDEN)
    H, W = s.shape
    with georvq.open(TILE, codec=codec) as ds:
        h = ds.reader.header
        assert h["codec"]["bitstream_id"] == codec.bitstream_id
        assert h["licence"]["spdx"] == "CC-BY-SA-4.0"
        assert h["product"]["embeddings"] == tiles.EMBEDDINGS
        assert (ds.height, ds.width, ds.count) == (H, W, 128)
        for i, r in enumerate(GOLDEN_RUNGS):
            a = ds.read(quality=r)
            assert _sha(a) == str(g["sha_read"][i]), r
            assert (a[:, ~blk.valid] == 0).all()


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

def write_tile(codec, out, q, s, meta, stats=True):
    """Encode the crop to the store `out` as the golden tile is written."""
    v1_1.encode_shard(_tile(q, s, meta), out, codec, stats=stats,
                      chunk_px=TILE_CHUNK_PX, release="golden")


def inputs():
    """Cut the crop from the chip store and georeference it on its zone's
    grid from the chip's place in its tile (needs the v1.1 store's zone
    metadata)."""
    from pyproj import Transformer

    from georvq.data.store import open_store
    st = open_store(CHIP["store"])
    _, va = st.split(0.04, seed=0)
    r = va[va.chip_id == CHIP["chip_id"]].iloc[0]
    (y0, y1), (x0, x1) = CHIP["rows"], CHIP["cols"]
    q = np.load(f"{CHIP['store']}/shard_{int(r.shard):04d}.npy",
                mmap_mode="r")[int(r.slot)][:, y0:y1, x0:x1]
    s = np.load(f"{CHIP['store']}/scales_{int(r.shard):04d}.npy",
                mmap_mode="r")[int(r.slot)][y0:y1, x0:x1]
    lon, lat, year = float(r.tile_lon), float(r.tile_lat), int(r.year)
    z = tiles.utm_zone(lon)
    zg = tiles.zone_grid(tiles.open_store(), z)
    assert f"EPSG:{zg['epsg']}" == r.crs
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{zg['epsg']}",
                               always_xy=True)
    r0, c0, _, _ = tiles.tile_window(lon, lat, zg["west"], zg["top"],
                                     zg["px"], fwd.transform)
    R, C = r0 + int(r.row_off) + y0, c0 + int(r.col_off) + x0
    ti = zg["years"].index(year)
    meta = {"chip_id": int(r.chip_id), "year": year, "tile_lon": lon,
            "tile_lat": lat, "rows": [y0, y1], "cols": [x0, x1],
            "epsg": zg["epsg"], "zone": z,
            "transform": list(tiles.TesseraTile.zone_transform(zg, R, C)),
            "zone_window": [R, C, y1 - y0, x1 - x0],
            "source": tiles.source_url(tiles.V11_STORE, z, ti, R, C,
                                       y1 - y0, x1 - x0),
            "store": CHIP["store"],
            "split": "validation, val_frac 0.04 seed 0"}
    np.savez_compressed(INPUTS, embedding=np.ascontiguousarray(q),
                        scales=np.asarray(s, np.float32),
                        meta=np.array(json.dumps(meta)))
    print(INPUTS, os.path.getsize(INPUTS), meta)


def record():
    import shutil
    torch.set_grad_enabled(False)
    codec = _bundle_codec()
    q, s, meta = _inputs()
    blk = _encode(codec, q, s)
    dec = _decode_all(codec, blk)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "tile.georvq.zarr")
        write_tile(codec, out, q, s, meta)
        with georvq.open(out, codec=codec) as ds:
            reads = {r: ds.read(quality=r) for r in GOLDEN_RUNGS}
        if os.path.exists(TILE):
            shutil.move(TILE, os.path.join(tmp, "old.georvq.zarr"))
        shutil.copytree(out, TILE)
    rec = {f"rec_{k}": v for k, v in blk.records.items()}
    np.savez_compressed(
        GOLDEN, **rec, rows=blk.rows, cols=blk.cols, valid=blk.valid,
        bitstream_id=np.array(codec.bitstream_id),
        decoder_id=np.array(codec.decoder_id), rungs=np.array(GOLDEN_RUNGS),
        sha_field=np.array([_sha(dec[r][0]) for r in GOLDEN_RUNGS]),
        sha_out=np.array([_sha(dec[r][1]) for r in GOLDEN_RUNGS]),
        sha_read=np.array([_sha(reads[r]) for r in GOLDEN_RUNGS]),
        rel_l2=np.array([_rel_l2(dec[r][1], q, s, blk.valid)
                         for r in GOLDEN_RUNGS]))
    for r in GOLDEN_RUNGS:
        print("-".join(map(str, r)),
              f"relative L2 {_rel_l2(dec[r][1], q, s, blk.valid):.6f}",
              "read == decode" if np.array_equal(reads[r], dec[r][1]) else
              f"read differs from decode by {np.abs(reads[r] - dec[r][1]).max():.3g}")
    print(GOLDEN, os.path.getsize(GOLDEN), TILE)


if __name__ == "__main__":
    if sys.argv[1:] == ["inputs"]:
        inputs()
    elif sys.argv[1:] == ["record"]:
        record()
