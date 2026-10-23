"""AE v2 release layer: a synthetic bottom-up tile through encode to a
georvq Zarr store, the rasterio-like reader and reconstruction, against
a tiny stats_cond model trained for a few steps on CPU."""
import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from georvq.ae import v2  # noqa: E402
from georvq.ae.v2 import requant_ae_np  # noqa: E402
from georvq.config import TierSpec, load_config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference.api import Codec  # noqa: E402
from georvq.inference.mean_codec import RVQPCAMeanChannel  # noqa: E402
from georvq.inference.records import RecordReader  # noqa: E402
from georvq.model.codec import sample_path_depths  # noqa: E402

H, WD, WIN = 200, 168, 16
TIERS = "model.tiers=['1:16:32:3','2:8:32:2','4:8:16:2']"
STATS = ("model.stats_cond={subtract_mean: true, cond_encoder: true, "
         "cond_decoder: true, bits_per_band: 6}")
# base 2 x 2 chunks, mid 4 x 3, fine 7 x 6, with partial edge chunks
CHUNK_PX = {"base": 128, "mid": 64, "fine": 32}



def _same_decode(a, b, frac=1e-3):
    """Two decodes of the same windows made in different batches. Float
    matrix products differ slightly with batch size on CPU, so a value
    next to an int8 rounding boundary can move by one step; on the tiny
    model here that happens to a few values, on the release model it was
    measured at 0 of 67M values. Nodata must agree exactly."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    if a.dtype.kind == "f":
        assert np.allclose(a, b, rtol=0, atol=1.0)
        d = np.abs(a - b)
    else:
        d = np.abs(a.astype(np.int64) - b.astype(np.int64))
    assert d.max(initial=0) <= 1
    assert (d > 0).mean() <= frac
    nod = np.iinfo(a.dtype).min if a.dtype.kind == "i" else None
    if nod is not None:
        assert np.array_equal(a == nod, b == nod)

def _field(h, w, seed=0):
    """Smooth unit-norm 64-band field: a few directions blended by
    low-frequency weights."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(5, 64))
    yy, xx = np.mgrid[0:h, 0:w] / 40.0
    wts = np.stack([np.sin(yy * rng.uniform(0.5, 2) + rng.uniform(0, 6))
                    * np.cos(xx * rng.uniform(0.5, 2) + rng.uniform(0, 6))
                    for _ in range(5)])
    f = np.einsum("khw,kc->chw", np.exp(2 * wts), dirs)
    f += 0.15 * rng.normal(size=f.shape)
    return (f / np.linalg.norm(f, axis=0, keepdims=True)).astype(np.float32)


def _books(x, L=2, K=16, seed=0):
    rng = np.random.default_rng(seed)
    r, books = x.copy(), []
    for _ in range(L):
        C = r[rng.choice(len(r), K, replace=False)].copy()
        for _ in range(8):
            a = ((r[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            for k in range(K):
                if (a == k).any():
                    C[k] = r[a == k].mean(0)
        a = ((r[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        r = r - C[a]
        books.append(C)
    return np.stack(books), r


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.manual_seed(0)
    tmp = tmp_path_factory.mktemp("aev2")
    north = requant_ae_np(_field(H, WD)[None])[0]          # (64, H, W) int8
    valid = np.ones((H, WD), bool)
    valid[:30, :50] = False                                 # north-west corner
    valid[120:140, 100:] = False
    north[:, ~valid] = v2.NODATA

    cfg = load_config("configs/smoke.yaml", [
        "model.bands=64", f"model.window={WIN}", TIERS, STATS])
    model = build_model(cfg, "cpu")
    x_all = torch.from_numpy(north.copy())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    rng = np.random.default_rng(1)
    for _ in range(150):
        ys = rng.integers(0, H - WIN, 16)
        xs = rng.integers(0, WD - WIN, 16)
        raw = torch.stack([x_all[:, y:y + WIN, x:x + WIN] for y, x in zip(ys, xs)])
        x, v = model.source.prep(raw, "cpu")
        st = model.patch_stats(x, v)
        Ls = sample_path_depths(16, model.specs, "cpu", model.cfg.min_depths,
                                path_frac=0.85)
        out, commit, _ = model.forward_train(x, Ls, stats=st, valid=v)
        loss = ((1 - (out * x).sum(1)) * v).sum() / v.sum() + commit
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    xs, vs = model.source.prep(torch.from_numpy(
        north[:, :192, :160].reshape(64, 12, WIN, 10, WIN)
        .transpose(1, 3, 0, 2, 4).reshape(-1, 64, WIN, WIN).copy()), "cpu")
    means = model.patch_stats(xs, vs).numpy()
    means = means[vs.reshape(len(vs), -1).any(1).numpy()]
    books, resid = _books(means)
    channel = RVQPCAMeanChannel.fit_residual(books, resid, total_bits=136)
    codec = v2.AEV2Codec(model, channel)

    # bottom-up source, as the AEF tiles are stored
    src = tmp / "tile.tif"
    t_north = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
    t_src = Affine(10.0, 0.0, 500000.0, 0.0, 10.0, 4000000.0 - 10.0 * H)
    with rasterio.open(src, "w", driver="GTiff", width=WD, height=H, count=64,
                       dtype="int8", nodata=v2.NODATA, crs="EPSG:32633",
                       transform=t_src, tiled=True, blockxsize=64,
                       blockysize=64, compress="deflate",
                       interleave="band") as ds:
        ds.write(north[:, ::-1])
        ds.update_tags(AREA_OR_POINT="Area")
        for b in range(1, 65):
            ds.set_band_description(b, f"A{b - 1:02d}")
    out = tmp / "tile.georvq.zarr"
    v2.encode_tile(src, out, codec, stats=True, chunk_px=CHUNK_PX, strip_px=48,
                   batch=256)
    return {"codec": codec, "src": src, "file": out, "north": north,
            "valid": valid, "t_north": t_north, "tmp": tmp}


def _direct(setup, depths):
    """Reference decode of the north-up tile: Codec on the coded mean."""
    codec, north, valid = setup["codec"], setup["north"], setup["valid"]
    gh, gw = -(-H // WIN), -(-WD // WIN)
    a = np.full((64, gh * WIN, gw * WIN), v2.NODATA, np.int8)
    a[:, :H, :WD] = north
    win = a.reshape(64, gh, WIN, gw, WIN).transpose(1, 3, 0, 2, 4)
    pv = (a[0] != v2.NODATA).reshape(gh, WIN, gw, WIN).transpose(0, 2, 1, 3)
    ys, xs = np.nonzero(pv.any((2, 3)))
    x, v = codec.model.source.prep(torch.from_numpy(win[ys, xs].copy()), "cpu")
    mu = codec.model.patch_stats(x, v).numpy()
    coded = torch.from_numpy(codec.channel.round_trip(mu))
    packets = codec.codec.encode(x, valid=v, mean=coded)
    y = codec.codec.decode(packets, depths=list(depths), valid=v)
    q = requant_ae_np(y.numpy(), v.numpy())
    out = np.full_like(a, v2.NODATA)
    for d, r, c in zip(q, ys, xs):
        out[:, r * WIN:(r + 1) * WIN, c * WIN:(c + 1) * WIN] = d
    return out[:, :H, :WD]


def test_full_depth_read_equals_direct_decode(setup):
    with v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        got = ds.read()
    want = _direct(setup, setup["codec"].full)[:, ::-1]
    assert got.dtype == np.int8 and got.shape == (64, H, WD)
    _same_decode(got, want)


def test_rung_read_equals_direct_decode(setup):
    with v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        got = ds.read(quality="3-1-0")
    _same_decode(got, _direct(setup, (3, 1, 0))[:, ::-1])


def test_georeferencing_matches_the_source(setup):
    with rasterio.open(setup["src"]) as s, \
            v2.open_dataset(setup["file"]) as ds:
        assert ds.transform == s.transform
        assert ds.crs == s.crs
        assert (ds.width, ds.height, ds.count) == (s.width, s.height, s.count)
        assert ds.dtypes == s.dtypes and ds.nodata == s.nodata
        assert ds.descriptions == s.descriptions
        assert list(ds.block_windows()) == list(s.block_windows())
        assert ds.product["orientation"] == "bottom-up"
        assert Affine(*ds.product["transform"]) == setup["t_north"]


def test_orientation_and_nodata_follow_the_source(setup):
    with rasterio.open(setup["src"]) as s, \
            v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        src = s.read()
        got = ds.read()
        assert np.array_equal(ds.dataset_mask(), s.dataset_mask())
        assert np.array_equal(ds.read_masks(), s.read_masks())
    inv = src[0] == v2.NODATA
    assert (got[:, inv] == v2.NODATA).all()
    assert (got[:, ~inv] != v2.NODATA).all()

    def cos(a, b):
        fa = (a.astype(np.float32) / 127.5) ** 2 * np.sign(a)
        fb = (b.astype(np.float32) / 127.5) ** 2 * np.sign(b)
        fa /= np.linalg.norm(fa, axis=0) + 1e-9
        fb /= np.linalg.norm(fb, axis=0) + 1e-9
        return float((fa * fb).sum(0)[~inv].mean())
    assert cos(got, src) > cos(got[:, ::-1], src) + 0.05


def test_window_read_is_a_crop_of_the_full_read(setup):
    with v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        full = ds.read(quality=(3, 2, 1))
        part = ds.read(window=Window(19, 37, 50, 70), quality="3-2-1")
        band = ds.read(5, window=((150, 200), (100, 168)), quality="3-2-1")
    _same_decode(part, full[:, 37:107, 19:69])
    _same_decode(band, full[4, 150:200, 100:168])


def test_quality_codes_fall_along_the_ladder(setup):
    codec = setup["codec"]
    n = len(codec.rungs)
    with RecordReader(setup["file"]) as rd:
        got = rd.read(0, rd.window_grid[0], 0, rd.window_grid[1],
                      depths=(0, 0, 0), mask=False, quality=True)
        q, live = got["quality"], got["base"].any(-1)
    mean = q[live][:, :n].astype(int)
    assert (mean > 0).all()
    steps = np.diff(mean, axis=1)
    assert (steps <= 1).mean() > 0.9
    # the model has had 150 steps, so the ladder is shallow but present
    assert (mean[:, -1] <= mean[:, 0]).mean() > 0.9
    assert (mean[:, -1] < mean[:, 0]).mean() > 0.5


def test_quality_bound_picks_rungs_per_window(setup):
    with v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        loose = ds.read(quality="mean<=1.0")
        cheapest = ds.read(quality=ds.rungs[0])
        _same_decode(loose, cheapest)
        _same_decode(ds.read(quality="p99<=0"), ds.read())
        q = v2.quality_value(ds.quality_codes(
            0, ds.reader.window_grid[0], 0,
            ds.reader.window_grid[1])[..., :len(ds.rungs)])
        bound = float(np.median(q[..., 3]))
        pick = ds.window_rungs(f"mean<={bound}")
        assert len(np.unique(pick)) > 1
        ok = np.take_along_axis(q, pick[..., None], -1)[..., 0] <= bound
        assert ok[pick < len(ds.rungs) - 1].all()
        mixed = ds.read(quality=f"mean<={bound}")
    # each window matches the rung it was assigned (rows of the north-up
    # grid map to the bottom-up output by flipping)
    W = WIN
    full = {k: _direct(setup, setup["codec"].rungs[k]) for k in np.unique(pick)}
    north = mixed[:, ::-1]
    for (r, c), k in np.ndenumerate(pick):
        sl = np.s_[:, r * W:min(H, (r + 1) * W), c * W:min(WD, (c + 1) * W)]
        _same_decode(north[sl], full[k][sl])


def test_point_read_is_one_read_per_array_then_cached(setup):
    with v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        rd = ds.reader
        assert rd.chunk_grids == {"base": (2, 2), "mid": (4, 3), "fine": (7, 6),
                                  "quality": (2, 2)}
        # source row 100 is north-up row 99, window (6, 4)
        plan = rd.plan(6, 7, 4, 5, depths=(3, 2, 1))
        masked = (0, 0) in rd.masked               # the point's base chunk
        assert sorted(name for name, _ in plan) == (
            ["base", "fine"] + ["mask"] * masked + ["mid"])
        before = dict(rd.stats)
        a = ds.read(window=Window(70, 100, 1, 1), quality="3-2-1")
        # one base chunk (and its mask), two mid levels and one fine level
        assert rd.stats["values"] - before["values"] == 4 + masked
        before = dict(rd.stats)
        b = ds.read(window=Window(68, 98, 3, 3), quality="3-2-1")
        assert rd.stats["values"] == before["values"]
        _same_decode(a, b[:, 2:3, 2:3])
        # a quality bound reads the quality chunk, then only what the
        # cache lacks
        assert len(rd.plan(6, 7, 4, 5, depths=(0, 0, 0), base=False, mask=False,
                           quality=True)) == 1
        ds.read(window=Window(70, 100, 1, 1), quality="p99<=0")
        assert rd.plan(6, 7, 4, 5, quality=True) == []


def test_header_records_the_given_source(setup):
    with v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        assert ds.product["source"] == str(setup["src"])
    out = setup["tmp"] / "named.georvq.zarr"
    url = "https://data.source.coop/tge-labs/aef/v1/annual/tile.tiff"
    v2.encode_tile(setup["src"], out, setup["codec"], stats=False,
                   chunk_px=CHUNK_PX, strip_px=48, batch=256, source=url)
    with v2.open_dataset(out, codec=setup["codec"]) as ds:
        assert ds.product["source"] == url


def test_empty_chunk_reads_as_nodata_without_a_read(setup):
    # the fixture's corner widened to cover fine chunk (0, 0), north-up
    # rows 0..31 and columns 0..31
    src, out = setup["tmp"] / "corner.tif", setup["tmp"] / "corner.georvq.zarr"
    with rasterio.open(setup["src"]) as s:
        a, prof = s.read(), s.profile
    a[:, H - 32:, :50] = v2.NODATA
    with rasterio.open(src, "w", **prof) as d:
        d.write(a)
    v2.encode_tile(src, out, setup["codec"], stats=True, chunk_px=CHUNK_PX,
                   strip_px=48, batch=256)
    with v2.open_dataset(out, codec=setup["codec"]) as ds:
        rd = ds.reader
        assert rd.empty["fine"] == {(0, 0)} and not rd.empty["mid"]
        before = dict(rd.stats)
        got = ds.read(window=Window(3, 170, 20, 30))
        assert rd.stats == before
        assert (got == v2.NODATA).all()
        full = ds.read()
        _same_decode(got, full[:, 170:200, 3:23])
        assert np.array_equal(full == v2.NODATA, a == v2.NODATA)


def test_reconstruct_keeps_the_source_profile(setup):
    out = setup["tmp"] / "recon.tif"
    with v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        v2.reconstruct(ds, out, quality="3-2-1", rows=64)
        want = ds.read(quality="3-2-1")
    with rasterio.open(setup["src"]) as s, rasterio.open(out) as r:
        for k in ("dtype", "count", "width", "height", "blockxsize",
                  "blockysize", "tiled", "compress", "interleave", "nodata"):
            assert r.profile[k] == s.profile[k], k
        assert r.transform == s.transform and r.crs == s.crs
        assert r.descriptions == s.descriptions
        assert r.tags()["AREA_OR_POINT"] == "Area"
        _same_decode(r.read(), want)


@pytest.fixture(scope="module")
def recon(setup):
    """The file decoded at 3-2-1 into a GeoTIFF: rasterio on it is the
    reference for what the dataset's read methods return."""
    out = setup["tmp"] / "recon321.tif"
    with v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        v2.reconstruct(ds, out, quality="3-2-1", rows=64)
    return out


def test_geometry_matches_rasterio_on_the_source(setup):
    with rasterio.open(setup["src"]) as s, v2.open_dataset(setup["file"]) as ds:
        for k in ("bounds", "res", "shape", "indexes", "transform", "crs"):
            assert getattr(ds, k) == getattr(s, k), k
        assert ds.meta == s.meta
        assert ds.driver == "GeoRVQZarr" and ds.overviews(1) == []
        for rc in [(0, 0), (H - 1, 0), (17, WD - 1), (33.4, 101.7)]:
            for off in ("center", "ul", "lr"):
                assert ds.xy(*rc, offset=off) == s.xy(*rc, offset=off)
            assert ds.index(*s.xy(*rc)) == s.index(*s.xy(*rc))
        b = s.bounds
        bb = (b.left + 31, b.bottom - 57, b.left + 402, b.bottom - 815)
        assert ds.window(*bb) == s.window(*bb)
        w = Window(13, 21, 40, 30)
        assert ds.window_transform(w) == s.window_transform(w)
        assert ds.window_bounds(w) == s.window_bounds(w)
        assert not ds.closed
    assert ds.closed


def test_band_indexes_are_checked_as_rasterio_checks_them(setup, recon):
    with rasterio.open(recon) as r, \
            v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        for bad in (0, -1, 65, [1, 0], [3, 65]):
            with pytest.raises(IndexError) as want:
                r.read(bad, window=Window(0, 0, 2, 2))
            with pytest.raises(IndexError) as got:
                ds.read(bad, window=Window(0, 0, 2, 2), quality="3-2-1")
            assert str(got.value) == str(want.value)
            with pytest.raises(IndexError):
                ds.read_masks(bad)
        assert ds.read(3, window=Window(0, 0, 4, 5), quality="3-2-1").shape == (5, 4)
        assert ds.read([3], window=Window(0, 0, 4, 5),
                       quality="3-2-1").shape == (1, 5, 4)


READ_CASES = [
    {},
    {"window": Window(19, 37, 50, 70)},
    {"window": ((150, 200), (100, 168))},
    {"window": Window(150, 180, 40, 40)},                   # cropped
    {"window": Window(-7, -5, 30, 20)},                     # cropped
    {"window": Window(-7, -5, 30, 20), "boundless": True},
    {"window": Window(150, 180, 40, 40), "boundless": True, "fill_value": 7},
    {"window": Window(300, 300, 10, 10), "boundless": True},
    {"window": Window(10.4, 20.6, 33.5, 17.4)},
    {"window": Window(3.5, 5.5, 8.5, 8.5), "boundless": True},
    {"out_shape": (37, 29)},
    {"out_shape": (H * 2 + 3, WD + 5)},
    {"window": Window(-20, 150, 90, 80), "boundless": True, "out_shape": (13, 31)},
    {"window": Window(-20, 150, 90, 80), "boundless": True, "out_shape": (170, 190)},
    {"window": Window(-0.5, 190.5, 30.5, 20.5), "boundless": True},
    {"window": Window(120.7, -3.2, 70.1, 50.3), "boundless": True,
     "out_shape": (11, 23)},
    {"window": Window(-50, -50, 300, 300), "boundless": True, "out_shape": (41, 37)},
    {"window": Window(40, 0, 90, 60), "fill_value": 5},
    {"window": Window(40, 0, 90, 60), "masked": True},
    {"window": Window(40, 0, 90, 60), "masked": True, "fill_value": 5},
    {"window": Window(-20, 150, 90, 80), "boundless": True, "masked": True},
    # rasterio masks the outside and pixels equal to fill_value here,
    # not nodata
    {"window": Window(-20, 150, 90, 80), "boundless": True, "masked": True,
     "fill_value": 7},
    {"window": Window(-20, 150, 90, 80), "boundless": True, "masked": True,
     "fill_value": 7, "out_dtype": "float32", "out_shape": (17, 23)},
    {"window": Window(40, 0, 90, 60), "out_dtype": "uint8"},
    {"window": Window(40, 0, 90, 60), "out_dtype": "float32", "masked": True},
]


@pytest.mark.parametrize("kw", READ_CASES)
@pytest.mark.parametrize("indexes", [None, 5, [7, 2]])
def test_read_matches_rasterio_on_the_decoded_tile(setup, recon, kw, indexes):
    with rasterio.open(recon) as r, \
            v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        want = r.read(indexes, **kw)
        got = ds.read(indexes, quality="3-2-1", **kw)
        assert got.shape == want.shape and got.dtype == want.dtype
        _same_decode(np.ma.getdata(got), np.ma.getdata(want))
        assert isinstance(got, np.ma.MaskedArray) == isinstance(want, np.ma.MaskedArray)
        if isinstance(want, np.ma.MaskedArray):
            assert np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(want))
            assert got.fill_value == want.fill_value
        mkw = {k: v for k, v in kw.items() if k in ("window", "boundless",
                                                    "out_shape")}
        m_got, m_want = ds.read_masks(indexes, **mkw), r.read_masks(indexes, **mkw)
        assert m_got.dtype == np.uint8 and np.array_equal(m_got, m_want)
        assert np.array_equal(ds.dataset_mask(**mkw), r.dataset_mask(**mkw))


def test_read_out_and_resampling(setup, recon):
    from rasterio.enums import Resampling
    with rasterio.open(recon) as r, \
            v2.open_dataset(setup["file"], codec=setup["codec"]) as ds:
        out = np.zeros((2, 9, 11), np.int8)
        got = ds.read([1, 2], out=out, window=Window(5, 5, 30, 20), quality="3-2-1")
        assert got is out
        assert np.array_equal(out, r.read([1, 2], window=Window(5, 5, 30, 20),
                                          out_shape=(9, 11)))
        with pytest.raises(NotImplementedError):
            ds.read(1, out_shape=(10, 10), resampling=Resampling.bilinear)
        a = ds.read(1, window=Window(5, 5, 8, 8), resampling=Resampling.bilinear,
                    quality="3-2-1")
        _same_decode(a, r.read(1, window=Window(5, 5, 8, 8)))


def test_ae_instance_layout_matches_the_spec():
    specs = [TierSpec(grid=1, dim=256, codes=2048, depth=8),
             TierSpec(grid=4, dim=192, codes=8192, depth=8),
             TierSpec(grid=16, dim=64, codes=16384, depth=8)]
    bits = np.array([7] * 32 + [6] * 32)
    ch = RVQPCAMeanChannel(np.zeros((2, 4096, 64)), np.zeros(64), np.eye(64),
                           np.ones(64), bits, np.ones(64))
    assert ch.cost_bits() == 440
    lay = v2.layout(specs, 64, ch, n_rungs=23)
    assert {k: (g.chunk_px, g.chunk_windows) for k, g in lay.groups.items()} == {
        "base": (8192, 128), "mid": (8192, 128), "fine": (1024, 16),
        "quality": (8192, 128)}
    assert lay.column_order == (["base", "mask"] + [f"mid_{i}" for i in range(1, 9)]
                                + [f"fine_{i}" for i in range(1, 9)] + ["quality"])
    rb = {n: c.record_bytes for n, c in lay.columns.items()}
    assert (rb["base"], rb["mid_1"], rb["fine_8"], rb["quality"]) == (66, 26, 448, 46)
    # bytes of one chunk's value per column
    n = {k: g.chunk_windows ** 2 for k, g in lay.groups.items()}
    assert [rb[c] * n[lay.kind_of[c]] for c in ("base", "mid_1", "fine_1",
                                                 "quality")] == [
        1081344, 425984, 114688, 753664]
    assert [t.name for t in lay.tiers] == ["coarse", "mid", "fine"]
    assert lay.tiers[0].in_base
    assert v2.rung_bits(specs, (8, 8, 2), ch) == 440 + 88 + 8 * 16 * 13 + 2 * 256 * 14


def test_output_name_and_info(setup, capsys):
    from georvq.inference.records import output_name
    assert output_name("/x/y/x21yf-0000008192-0000000000.tiff") == \
        "x21yf-0000008192-0000000000.georvq.zarr"
    assert v2.main(["info", str(setup["file"])]) == 0
    out = capsys.readouterr().out
    assert "georvq-zarr" in out and "fine" in out and "store" in out
    assert "quality for decoders ['" in out


def test_quality_code_round_trip():
    e = np.array([0.0, 1e-7, 1e-6, 1e-3, 0.02, 1.0, 0.0123, np.nan])
    c = v2.quality_code(e)
    assert list(c[:3]) == [0, 0, 0] and c[-1] == 0
    # codes round up: the value of a code is the smallest step at or
    # above the error, so a bound on it is never exceeded
    v = v2.quality_value(c[3:7])
    assert (v >= e[3:7] * (1 - 1e-12)).all()
    assert (v2.quality_value(c[3:7] - 1) < e[3:7]).all()
    assert np.allclose(v, e[3:7], rtol=0.075)


@pytest.fixture(scope="module")
def onnx_export(setup):
    pytest.importorskip("onnxruntime")
    out = setup["tmp"] / "onnx"
    meta = v2.export_onnx(setup["codec"], str(out))
    return str(out), meta


def test_onnx_backend_reads_as_torch(setup, onnx_export):
    out, meta = onnx_export
    assert meta["bitstream_id"] == setup["codec"].bitstream_id
    with v2.open_dataset(setup["file"], backend="onnx", onnx_dir=out) as ds:
        assert isinstance(ds.codec, v2.AEV2OnnxCodec)
        for q in (None, "3-1-0", "1-0-0"):
            got = ds.read(quality=q)
            with v2.open_dataset(setup["file"], codec=setup["codec"]) as dt:
                want = dt.read(quality=q)
            assert ((got == v2.NODATA) == (want == v2.NODATA)).all()
            d = np.abs(got.astype(int) - want.astype(int))
            assert d.max() <= 1 and (d > 0).mean() < 1e-3, q


def test_onnx_backend_read_without_torch(setup, onnx_export, tmp_path):
    import os
    import subprocess
    import sys
    from tests.test_onnx_stats import NO_TORCH
    out, _ = onnx_export
    with v2.open_dataset(setup["file"], backend="onnx", onnx_dir=out) as ds:
        want = ds.read(window=Window(10, 20, 60, 50), quality="3-2-1")
    np.save(tmp_path / "want.npy", want)
    code = NO_TORCH + f"""
import numpy as np
from rasterio.windows import Window
from georvq.ae import v2
with v2.open_dataset({str(setup["file"])!r}, backend="onnx",
                     onnx_dir={out!r}) as ds:
    got = ds.read(window=Window(10, 20, 60, 50), quality="3-2-1")
assert np.array_equal(got, np.load({str(tmp_path / "want.npy")!r}))
assert "torch" not in sys.modules
print("ok")
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=os.getcwd())
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_onnx_export_refuses_a_swapped_channel(setup, onnx_export, tmp_path):
    import shutil
    out, _ = onnx_export
    bad = tmp_path / "bad"
    shutil.copytree(out, bad)
    ch = RVQPCAMeanChannel.load(str(bad / v2.ONNX_CHANNEL_NAME))
    ch.mu0 = ch.mu0 + 0.01
    ch.save(str(bad / v2.ONNX_CHANNEL_NAME))
    with pytest.raises(ValueError, match="mean channel"):
        v2.AEV2OnnxCodec(str(bad))


def test_cli_decode_with_the_onnx_backend(setup, onnx_export, tmp_path):
    out, _ = onnx_export
    a, b = tmp_path / "onnx.npy", tmp_path / "ref.npy"
    assert v2.main(["decode", str(setup["file"]), str(a), "--backend", "onnx",
                    "--onnx-dir", out, "--window", "16,0,40,64",
                    "--quality", "3-2-1"]) == 0
    with v2.open_dataset(setup["file"], backend="onnx", onnx_dir=out) as ds:
        np.save(b, ds.read(window=((16, 56), (0, 64)), quality="3-2-1"))
    assert np.array_equal(np.load(a), np.load(b))
