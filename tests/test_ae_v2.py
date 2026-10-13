"""AE v2 release layer: a synthetic bottom-up tile through encode, the
rasterio-like reader and reconstruction, against a tiny stats_cond model
trained for a few steps on CPU."""
import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from georvq.ae import v2  # noqa: E402
from georvq.ae.cli import requant_ae_np  # noqa: E402
from georvq.config import TierSpec, load_config  # noqa: E402
from georvq.factory import build_model  # noqa: E402
from georvq.inference.api import Codec  # noqa: E402
from georvq.inference.gvq import GVQReader  # noqa: E402
from georvq.inference.mean_codec import RVQPCAMeanChannel  # noqa: E402
from georvq.model.codec import sample_path_depths  # noqa: E402

H, WD, WIN = 200, 168, 16
TIERS = "model.tiers=['1:16:32:3','2:8:32:2','4:8:16:2']"
STATS = ("model.stats_cond={subtract_mean: true, cond_encoder: true, "
         "cond_decoder: true, bits_per_band: 6}")
CHUNK_PX = {"base": 64, "mid": 32, "fine": 32, "quality": 64}


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
    out = tmp / "tile.gvq"
    v2.encode_tile(src, out, codec, stats=True, chunk_px=CHUNK_PX, batch=256)
    return {"codec": codec, "src": src, "gvq": out, "north": north,
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
    with v2.open_dataset(setup["gvq"], codec=setup["codec"]) as ds:
        got = ds.read()
    want = _direct(setup, setup["codec"].full)[:, ::-1]
    assert got.dtype == np.int8 and got.shape == (64, H, WD)
    assert np.array_equal(got, want)


def test_rung_read_equals_direct_decode(setup):
    with v2.open_dataset(setup["gvq"], codec=setup["codec"]) as ds:
        got = ds.read(quality="3-1-0")
    assert np.array_equal(got, _direct(setup, (3, 1, 0))[:, ::-1])


def test_georeferencing_matches_the_source(setup):
    with rasterio.open(setup["src"]) as s, \
            v2.open_dataset(setup["gvq"]) as ds:
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
            v2.open_dataset(setup["gvq"], codec=setup["codec"]) as ds:
        src = s.read()
        got = ds.read()
        assert np.array_equal(ds.read_masks(), src[0] != v2.NODATA)
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
    with v2.open_dataset(setup["gvq"], codec=setup["codec"]) as ds:
        full = ds.read(quality=(3, 2, 1))
        part = ds.read(window=Window(19, 37, 50, 70), quality="3-2-1")
        band = ds.read(5, window=((150, 200), (100, 168)), quality="3-2-1")
    assert np.array_equal(part, full[:, 37:107, 19:69])
    assert np.array_equal(band, full[4, 150:200, 100:168])


def test_quality_codes_fall_along_the_ladder(setup):
    codec = setup["codec"]
    n = len(codec.rungs)
    with GVQReader(setup["gvq"]) as rd:
        q = rd.read_records("quality", 1, 0, *rd.window_grid[:1], 0,
                            rd.window_grid[1])[:, :, 0]
        live = rd.read_records("base", 1, 0, rd.window_grid[0], 0,
                               rd.window_grid[1])[:, :, 0].any(-1)
    mean = q[live][:, :n].astype(int)
    assert (mean > 0).all()
    steps = np.diff(mean, axis=1)
    assert (steps <= 1).mean() > 0.9
    # the model has had 150 steps, so the ladder is shallow but present
    assert (mean[:, -1] <= mean[:, 0]).mean() > 0.9
    assert (mean[:, -1] < mean[:, 0]).mean() > 0.5


def test_quality_bound_picks_rungs_per_window(setup):
    with v2.open_dataset(setup["gvq"], codec=setup["codec"]) as ds:
        loose = ds.read(quality="mean<=1.0")
        cheapest = ds.read(quality=ds.rungs[0])
        assert np.array_equal(loose, cheapest)
        assert np.array_equal(ds.read(quality="p99<=0"), ds.read())
        q = v2.quality_value(ds.reader.read_records(
            "quality", 1, 0, ds.reader.window_grid[0], 0,
            ds.reader.window_grid[1])[:, :, 0, :len(ds.rungs)])
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
        assert np.array_equal(north[sl], full[k][sl])


def test_partial_read_fetches_only_its_chunks(setup):
    with v2.open_dataset(setup["gvq"], codec=setup["codec"]) as ds:
        before = dict(ds.reader.stats)
        ds.read(window=Window(64, H - 16, 16, 16), quality="3-1-0")
        req = ds.reader.stats["requests"] - before["requests"]
    # base and mid chunk, the mask table and one mask block; fine not read
    assert req <= 5


def test_reconstruct_keeps_the_source_profile(setup):
    out = setup["tmp"] / "recon.tif"
    with v2.open_dataset(setup["gvq"], codec=setup["codec"]) as ds:
        v2.reconstruct(ds, out, quality="3-2-1", rows=64)
        want = ds.read(quality="3-2-1")
    with rasterio.open(setup["src"]) as s, rasterio.open(out) as r:
        for k in ("dtype", "count", "width", "height", "blockxsize",
                  "blockysize", "tiled", "compress", "interleave", "nodata"):
            assert r.profile[k] == s.profile[k], k
        assert r.transform == s.transform and r.crs == s.crs
        assert r.descriptions == s.descriptions
        assert r.tags()["AREA_OR_POINT"] == "Area"
        assert np.array_equal(r.read(), want)


def test_ae_instance_layout_matches_the_spec():
    specs = [TierSpec(grid=1, dim=256, codes=2048, depth=8),
             TierSpec(grid=4, dim=192, codes=8192, depth=8),
             TierSpec(grid=16, dim=64, codes=16384, depth=8)]
    bits = np.array([7] * 32 + [6] * 32)
    ch = RVQPCAMeanChannel(np.zeros((2, 4096, 64)), np.zeros(64), np.eye(64),
                           np.ones(64), bits, np.ones(64))
    assert ch.cost_bits() == 440
    secs = {s.name: s for s in v2.layout(specs, 64, ch, n_rungs=23,
                                         window_grid=(128, 128))}
    assert [(s.chunk_windows, s.levels, s.record_bytes, s.chunk_bytes)
            for s in secs.values()] == [
        ((64, 64), 1, 66, 270336), ((32, 32), 8, 26, 212992),
        ((16, 16), 8, 448, 917504), ((64, 64), 1, 46, 188416)]
    assert v2.rung_bits(specs, (8, 8, 2), ch) == 440 + 88 + 8 * 16 * 13 + 2 * 256 * 14


def test_quality_code_round_trip():
    e = np.array([0.0, 1e-7, 1e-6, 1e-3, 0.02, 1.0, np.nan])
    c = v2.quality_code(e)
    assert list(c[:3]) == [0, 0, 0] and c[-1] == 0
    assert np.allclose(v2.quality_value(c[3:6]), e[3:6], rtol=0.04)
