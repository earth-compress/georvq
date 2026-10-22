"""AE v2 release bundle: int8 codebook storage, the bundle round trip on
a tiny synthetic model, and a golden decode gate on the released bundle.

The golden gate needs the released bundle and skips when it is absent.
It looks in GEORVQ_AE_V2_BUNDLE, GEORVQ_AE_BUNDLE, then the bundle
cache entry for the golden bitstream id (`georvq.ae.v2_resolve.
bundle_path`, ~/.cache/georvq/bundles/<bitstream_id> by default). Its inputs are the eight AE validation windows of
golden/divfull_val_tokens.npz. golden/ae_v2_golden.npz holds the codes
the bundle's encoder produced for them on CPU, and sha256 hashes of the
float32 field and the int8 tile decoded from those codes at three rungs.
golden/ae_v2_tile.georvq.zarr is those windows written as one v2
store with small chunks, read back through the store reader.

Everything is compared exactly. The hashes were recorded on CPU in fp32
(cirrus, x86-64); a CPU with other vector kernels could produce
different float bits, in which case the test reports the cosine against
the source next to the recorded one so the size of the difference is
visible. Re-record with

    python -m tests.test_ae_v2_bundle record
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

from georvq.ae import v2
from georvq.ae.v2_bundle import (build_bundle, book_from_int, book_to_int,
                                 load_bundle)
from georvq.config import load_config
from georvq.factory import build_model
from georvq.inference.mean_codec import RVQPCAMeanChannel
from georvq.model.vq_quant import QuantizedBookVQ
from georvq.repro import save_checkpoint

HERE = os.path.dirname(__file__)
INPUTS = os.path.join(HERE, "golden/divfull_val_tokens.npz")
GOLDEN = os.path.join(HERE, "golden/ae_v2_golden.npz")
# maintainer's copy, tried after GEORVQ_AE_V2_BUNDLE, GEORVQ_AE_BUNDLE
# and the bundle cache entry for the golden bitstream id
LOCAL_BUNDLE = "/home/ali/projects/georvq/scratch/ae-v2/bundle_v2"


def _find_bundle():
    """GEORVQ_AE_V2_BUNDLE as given; the other places only when their
    bundle.json names the golden bitstream id."""
    from georvq.ae.v2_resolve import ENV_BUNDLE, bundle_path
    if os.environ.get("GEORVQ_AE_V2_BUNDLE"):
        return os.environ["GEORVQ_AE_V2_BUNDLE"]
    with np.load(GOLDEN) as g:
        bid = str(g["bitstream_id"])
    for d in (os.environ.get(ENV_BUNDLE), bundle_path(bid), LOCAL_BUNDLE):
        meta = os.path.join(d or "", "bundle.json")
        if d and os.path.exists(meta) and \
                json.load(open(meta)).get("bitstream_id") == bid:
            return d
    return None


BUNDLE = _find_bundle()
CKPT = os.environ.get(
    "GEORVQ_AE_V2_CKPT",
    "/home/ali/projects/georvq/scratch/ae-v2/ownk_sc4_step80000.pt")
MEAN_NPZ = os.environ.get(
    "GEORVQ_AE_V2_MEAN",
    "/home/ali/projects/georvq/scratch/ae-v2/meanrvq/mean_channel_v2.npz")
TILE = os.path.join(HERE, "golden/ae_v2_tile.georvq.zarr")
GOLDEN_RUNGS = ((2, 0, 0), (8, 8, 2), (8, 8, 8))
TILE_GRID = (2, 4)           # the eight windows as a 2 x 4 window tile
# Records are dense over a chunk, so the default 8192 px base chunk would
# make the fixture 3.6 MB for eight windows. These sizes keep it small and
# still split the fine tier over two chunks.
TILE_CHUNK_PX = {"base": 256, "mid": 256, "fine": 128}


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------
# int8 books
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [8, 4])
def test_projected_book_round_trips_bit_for_bit(bits):
    torch.manual_seed(0)
    lev = QuantizedBookVQ(512, 24, book_bits=bits)
    lev.codebook.weight.data.mul_(torch.logspace(-3, 1, 512)[:, None])
    lev.project_book()
    W = lev.codebook.weight.detach().clone()
    q, s = book_to_int(W, bits)
    assert q.dtype == np.int8 and s.dtype == np.float32
    assert np.abs(q[q != -128].astype(int)).max() <= 2 ** (bits - 1) - 1
    assert (q == -128).any()            # some weights are -0.0
    assert torch.equal(book_from_int(q, s).view(torch.int32),
                       W.view(torch.int32))


def test_off_grid_book_is_refused():
    W = torch.randn(16, 8)
    with pytest.raises(ValueError, match="not on the 8-bit grid"):
        book_to_int(W, 8)


# --------------------------------------------------------------------------
# synthetic bundle round trip
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("aev2bundle")
    torch.manual_seed(0)
    cfg = load_config("configs/smoke.yaml", [
        "model.bands=64", "model.window=16",
        "model.tiers=['1:16:32:3','2:8:32:2','4:8:16:2']",
        "model.min_depths=[1,0,0]", "model.vq.book_bits=8",
        "model.stats_cond={subtract_mean: true, cond_encoder: true, "
        "cond_decoder: true, bits_per_band: 6}"])
    model = build_model(cfg, "cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    for rvq in model.rvqs:
        rvq.project_books()
    model.eval()
    ckpt = tmp / "tiny.pt"
    save_checkpoint(ckpt, model, cfg, 7)

    rng = np.random.default_rng(0)
    means = rng.normal(size=(256, 64)).astype(np.float32) * 0.05
    books = means[:8][None].copy()
    a = ((means[:, None] - books[0][None]) ** 2).sum(-1).argmin(1)
    channel = RVQPCAMeanChannel.fit_residual(books, means - books[0][a],
                                             total_bits=96)
    npz = tmp / "mean.npz"
    channel.save(npz)
    out = tmp / "bundle"
    build_bundle(ckpt, npz, out)

    raw = rng.integers(-127, 128, size=(6, 64, 16, 16)).astype(np.int8)
    raw[1, :, :5, :7] = v2.NODATA
    return {"ckpt": ckpt, "npz": npz, "dir": out, "raw": raw}


def test_bundle_holds_no_training_state_and_int8_books(tiny):
    with np.load(tiny["dir"] / "weights.npz", allow_pickle=False) as z:
        keys = z.files
    assert keys and not any(k.rsplit(".", 1)[-1] in
                            ("ema_count", "ema_sum", "step", "initialized")
                            for k in keys)
    assert not any(k.endswith("codebook.weight") for k in keys)
    with np.load(tiny["dir"] / "codebooks.npz", allow_pickle=False) as z:
        assert all(z[k].dtype == np.int8 for k in z.files if k.endswith(".q"))
        assert all(z[k].dtype == np.float32 for k in z.files
                   if k.endswith(".scale"))
    meta = json.load(open(tiny["dir"] / "bundle.json"))
    assert [tuple(r["tuple"]) for r in meta["rungs"]][0] == (1, 0, 0)


def test_bundle_decodes_like_the_checkpoint(tiny):
    a = v2.AEV2Codec.load(str(tiny["ckpt"]), str(tiny["npz"]))
    b = load_bundle(tiny["dir"])
    assert (a.bitstream_id, a.decoder_id) == (b.bitstream_id, b.decoder_id)
    sa, sb = a.model.state_dict(), b.model.state_dict()
    for k, v in sa.items():
        if k.rsplit(".", 1)[-1] in ("ema_count", "ema_sum", "step",
                                    "initialized"):
            continue
        assert torch.equal(v, sb[k]), k
    ra, _, va = a.encode_windows(tiny["raw"], stats=False)
    rb, _, _ = b.encode_windows(tiny["raw"], stats=False)
    assert ra.keys() == rb.keys()
    for k in ra:
        assert np.array_equal(ra[k], rb[k]), k
    tiers = [ra[n].transpose(1, 0, 2) for n in a.names[1:]]
    valid = va.numpy()
    for r in [(1, 0, 0), (2, 1, 0), (3, 2, 2)]:
        fa, _ = a.decode_field(ra["base"][0], tiers, r, valid)
        fb, _ = b.decode_field(ra["base"][0], tiers, r, valid)
        assert torch.equal(fa.view(torch.int32), fb.view(torch.int32))
        assert np.array_equal(a.decode_records(ra["base"][0], tiers, r, valid),
                              b.decode_records(ra["base"][0], tiers, r, valid))


def test_tampered_bundle_is_refused(tiny, tmp_path):
    import shutil
    d = tmp_path / "b"
    shutil.copytree(tiny["dir"], d)
    meta = json.load(open(d / "bundle.json"))
    meta["bitstream_id"] = "0" * 16
    json.dump(meta, open(d / "bundle.json", "w"))
    with pytest.raises(ValueError, match="bitstream_id"):
        load_bundle(d)
    with np.load(d / "codebooks.npz") as z:
        books = {k: z[k] for k in z.files}
    books["rvqs.0.levels.0.q"][0, 0] ^= 1
    np.savez(d / "codebooks.npz", **books)
    with pytest.raises(ValueError, match="sha256"):
        load_bundle(d)


def test_decoder_only_bundle(tiny, tmp_path):
    from georvq.ae.v2_bundle import write_decoder_bundle
    d = tmp_path / "dec"
    meta = write_decoder_bundle(tiny["dir"], d)
    full = json.load(open(tiny["dir"] / "bundle.json"))
    assert meta["variant"] == "decoder-fp16" and full["variant"] == "full"
    assert meta["bitstream_id"] == full["bitstream_id"]
    assert meta["parent_decoder_id"] == full["decoder_id"]
    assert meta["decoder_id"] != full["decoder_id"]
    with np.load(d / "weights.npz", allow_pickle=False) as z:
        assert not any(k.startswith(("tokenizer.", "stats_enc.", "stem."))
                       for k in z.files)
        assert all(z[k].dtype != np.float32 for k in z.files)
    for n in ("codebooks.npz", "mean_channel.npz", "config.json"):
        assert open(d / n, "rb").read() == open(tiny["dir"] / n, "rb").read()
    a, b = load_bundle(tiny["dir"]), load_bundle(d)
    assert b.decode_only and not a.decode_only
    assert b.decoder_id == meta["decoder_id"]
    with pytest.raises(RuntimeError, match="decoder-only"):
        b.encode_windows(tiny["raw"])
    ra, _, va = a.encode_windows(tiny["raw"], stats=False)
    tiers = [ra[n].transpose(1, 0, 2) for n in a.names[1:]]
    for r in [(1, 0, 0), (3, 2, 2)]:
        fa, _ = a.decode_field(ra["base"][0], tiers, r, va.numpy())
        fb, _ = b.decode_field(ra["base"][0], tiers, r, va.numpy())
        assert torch.isfinite(fb).all()
        assert (fa - fb).abs().max() < 1e-2 * fa.abs().max()
    # georvq.open takes it for a store written by its parent decoder
    import georvq
    from tests.test_ae_index import write_tile
    store = str(tmp_path / "t.georvq.zarr")
    write_tile(store, a, 32, 48)
    with georvq.open(store, codec=b) as ds:
        assert ds.read(quality="3-2-2").shape == (64, 32, 48)
        assert ds.reader.quality_decoder is None


# --------------------------------------------------------------------------
# golden decode of the released bundle
# --------------------------------------------------------------------------

def test_golden_bundle_is_found_in_the_cache(tmp_path, monkeypatch):
    with np.load(GOLDEN) as g:
        bid = str(g["bitstream_id"])
    monkeypatch.delenv("GEORVQ_AE_V2_BUNDLE", raising=False)
    monkeypatch.setenv("GEORVQ_CACHE", str(tmp_path))
    other = tmp_path / "other"
    other.mkdir()
    json.dump({"bitstream_id": "0" * 16}, open(other / "bundle.json", "w"))
    monkeypatch.setenv("GEORVQ_AE_BUNDLE", str(other))
    entry = tmp_path / "bundles" / bid
    entry.mkdir(parents=True)
    json.dump({"bitstream_id": bid}, open(entry / "bundle.json", "w"))
    assert _find_bundle() == str(entry)


def _windows():
    return np.load(INPUTS)["tokens"]


def _decode_all(codec, rec, valid):
    """{rung: (float32 field, int8 tile)} on CPU from the encoder's
    records."""
    tiers = [rec[n].transpose(1, 0, 2) for n in codec.names[1:]]
    out = {}
    for r in GOLDEN_RUNGS:
        f, v = codec.decode_field(rec["base"][0], tiers, r, valid)
        out[r] = (f.cpu().numpy(), v2.requant(f, v))
    return out


def _cos(field, x, valid):
    c = (field * x).sum(1)
    return float(c[valid].mean())


def _bundle_codec():
    if BUNDLE is None or not os.path.exists(os.path.join(BUNDLE,
                                                         "bundle.json")):
        pytest.skip("AE v2 bundle not present (set GEORVQ_AE_V2_BUNDLE or "
                    "GEORVQ_AE_BUNDLE, or place it in the bundle cache)")
    return load_bundle(BUNDLE, "cpu")


@pytest.fixture(scope="module")
def released():
    codec = _bundle_codec()
    raw = _windows()
    rec, x, valid = codec.encode_windows(raw, stats=False)
    return codec, raw, rec, x.numpy(), valid.numpy()


def test_released_bundle_decodes_like_its_checkpoint(released):
    codec, raw, rec, _, valid = released
    if not (os.path.exists(CKPT) and os.path.exists(MEAN_NPZ)):
        pytest.skip("source checkpoint not present (set GEORVQ_AE_V2_CKPT)")
    ref = v2.AEV2Codec.load(CKPT, MEAN_NPZ, "cpu")
    assert (ref.bitstream_id, ref.decoder_id) == (codec.bitstream_id,
                                                  codec.decoder_id)
    sb = codec.model.state_dict()
    for k, v in ref.model.state_dict().items():
        if k.rsplit(".", 1)[-1] in ("ema_count", "ema_sum", "step",
                                    "initialized"):
            continue
        assert torch.equal(v, sb[k]), k
    got = ref.encode_windows(raw[:2], stats=False)[0]
    want = codec.encode_windows(raw[:2], stats=False)[0]
    for k in want:
        assert np.array_equal(got[k], want[k]), k
    tiers = [rec[n].transpose(1, 0, 2) for n in codec.names[1:]]
    for r in ((2, 0, 0), (8, 8, 8)):
        fa, _ = ref.decode_field(rec["base"][0], tiers, r, valid)
        fb, _ = codec.decode_field(rec["base"][0], tiers, r, valid)
        assert torch.equal(fa.view(torch.int32), fb.view(torch.int32)), r


def test_golden_codes(released):
    codec, _, rec, _, _ = released
    g = np.load(GOLDEN)
    assert codec.bitstream_id == str(g["bitstream_id"])
    assert codec.decoder_id == str(g["decoder_id"])
    for k in rec:
        assert np.array_equal(rec[k], g[f"rec_{k}"]), k


def test_golden_decode(released):
    codec, _, _, x, valid = released
    g = np.load(GOLDEN)
    rec = {k[4:]: g[k] for k in g.files if k.startswith("rec_")}
    dec = _decode_all(codec, rec, valid)
    for i, r in enumerate(GOLDEN_RUNGS):
        f, q = dec[r]
        name = "-".join(map(str, r))
        assert _sha(f) == str(g["sha_field"][i]), (
            f"{name}: float field differs, cos {_cos(f, x, valid):.9f} vs "
            f"recorded {float(g['cos'][i]):.9f}")
        assert _sha(q) == str(g["sha_int8"][i]), name


def test_golden_tile_reads_back(released):
    codec, raw, _, _, _ = released
    if not os.path.exists(TILE):
        pytest.skip("tile fixture not recorded")
    g = np.load(GOLDEN)
    gh, gw = TILE_GRID
    W = codec.window
    with v2.open_dataset(TILE, codec=codec) as ds:
        assert ds.reader.header["codec"]["bitstream_id"] == codec.bitstream_id
        for i, r in enumerate(GOLDEN_RUNGS):
            a = ds.read(quality=r)[:, ::-1]            # stored bottom-up
            tiles = a.reshape(64, gh, W, gw, W).transpose(1, 3, 0, 2, 4) \
                .reshape(gh * gw, 64, W, W)
            assert _sha(tiles) == str(g["sha_int8"][i]), r


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

def _write_tile(raw, path, W):
    """The windows as one bottom-up AE-like GeoTIFF, row-major."""
    import rasterio
    from rasterio.transform import Affine
    gh, gw = TILE_GRID
    north = raw.reshape(gh, gw, 64, W, W).transpose(2, 0, 3, 1, 4) \
        .reshape(64, gh * W, gw * W)
    H, Wd = north.shape[1:]
    t = Affine(10.0, 0.0, 500000.0, 0.0, 10.0, 4000000.0 - 10.0 * H)
    with rasterio.open(path, "w", driver="GTiff", width=Wd, height=H,
                       count=64, dtype="int8", nodata=v2.NODATA,
                       crs="EPSG:32633", transform=t, tiled=True,
                       blockxsize=64, blockysize=64, compress="deflate",
                       interleave="band") as ds:
        ds.write(np.ascontiguousarray(north[:, ::-1]))
        for b in range(1, 65):
            ds.set_band_description(b, f"A{b - 1:02d}")


def record():
    import tempfile
    torch.set_grad_enabled(False)
    codec = load_bundle(BUNDLE, "cpu")
    raw = _windows()
    rec, x, valid = codec.encode_windows(raw, stats=False)
    dec = _decode_all(codec, rec, valid.numpy())
    out = {f"rec_{k}": v for k, v in rec.items()}
    out.update(
        bitstream_id=np.array(codec.bitstream_id),
        decoder_id=np.array(codec.decoder_id),
        rungs=np.array(GOLDEN_RUNGS),
        sha_field=np.array([_sha(dec[r][0]) for r in GOLDEN_RUNGS]),
        sha_int8=np.array([_sha(dec[r][1]) for r in GOLDEN_RUNGS]),
        cos=np.array([_cos(dec[r][0], x.numpy(), valid.numpy())
                      for r in GOLDEN_RUNGS]))
    np.savez_compressed(GOLDEN, **out)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "golden_tile.tif")
        _write_tile(raw, src, codec.window)
        v2.encode_tile(src, TILE, codec, stats=True, chunk_px=TILE_CHUNK_PX,
                       release="golden")
    for r, c in zip(GOLDEN_RUNGS, out["cos"]):
        print("-".join(map(str, r)), f"cos {c:.6f}")
    print(GOLDEN, os.path.getsize(GOLDEN), TILE, os.path.getsize(TILE))


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        record()


def test_shippable_config_drops_machine_paths():
    from georvq.ae.v2_bundle import shippable_config
    d = {"data": {"store_dir": "/data/x", "source": "ae"},
         "train": {"init_from": "/home/u/a.pt", "steps": 10,
                   "rd_refs": ["/home/u/r.json|t|64|ref", "rel|t|64|ref"]},
         "log": {"run_tag": "r"}}
    assert shippable_config(d) == {
        "data": {"store_dir": "", "source": "ae"},
        "train": {"init_from": "", "steps": 10, "rd_refs": ["rel|t|64|ref"]},
        "log": {"run_tag": "r"}}
