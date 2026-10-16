"""AE v2 decodes through georvq.ae.open from the prototype Zarr store of
a file match decodes from the GeoParquet file exactly, on the synthetic
tile of test_ae_v2 (with an empty chunk and quality-bound reads) and on
the golden tile with the released bundle."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("zarr")
from rasterio.windows import Window  # noqa: E402

from georvq.ae import v2  # noqa: E402
from georvq.inference import georvq_zarr as gz  # noqa: E402
from tests.test_ae_v2 import CHUNK_PX, H, setup  # noqa: E402,F401
from tests.test_ae_v2_bundle import (GOLDEN, GOLDEN_RUNGS, TILE,  # noqa: E402
                                     TILE_GRID, _bundle_codec, _sha)


@pytest.fixture(scope="module")
def corner(setup):  # noqa: F811
    """The fixture tile with its corner widened to cover fine chunk
    (0, 0), so the file has an empty chunk, as GeoParquet and Zarr."""
    tmp = setup["tmp"]
    src, pq_file = tmp / "zcorner.tif", tmp / "zcorner.georvq.parquet"
    with rasterio.open(setup["src"]) as s:
        a, prof = s.read(), s.profile
    a[:, H - 32:, :50] = v2.NODATA
    with rasterio.open(src, "w", **prof) as d:
        d.write(a)
    v2.encode_tile(src, pq_file, setup["codec"], stats=True, chunk_px=CHUNK_PX,
                   strip_px=48, batch=256)
    return pq_file, gz.parquet_to_zarr(pq_file, tmp / "zcorner.georvq.zarr")


@pytest.mark.parametrize("via", ["path", "group"])
def test_decodes_match_the_parquet_file(setup, corner, via):  # noqa: F811
    import zarr
    codec = setup["codec"]
    pq_file, zr = corner
    src = zr if via == "path" else zarr.open_group(str(zr), mode="r")
    with v2.open_dataset(pq_file, codec=codec) as dp, \
            v2.open_dataset(src, codec=codec) as dz:
        assert dz.driver == "GeoRVQZarr" and dz.reader.empty["fine"] == {(0, 0)}
        assert dz.profile == dp.profile and dz.tags() == dp.tags()
        assert dz.descriptions == dp.descriptions
        cases = [(None, None), (Window(3, 170, 20, 30), None),
                 (Window(3, 170, 20, 30), dp.rungs[-1]),
                 (Window(10, 20, 100, 90), dp.rungs[0]),
                 (Window(40, 60, 64, 64), dp.rungs[len(dp.rungs) // 2]),
                 (None, "p99<=0.05"), (Window(0, 0, 168, 200), "mean<=0.02")]
        for win, q in cases:
            kw = {} if q is None else {"quality": q}
            assert np.array_equal(dp.read(window=win, **kw),
                                  dz.read(window=win, **kw)), (win, q)
        assert np.array_equal(dp.dataset_mask(), dz.dataset_mask())
        before = dict(dz.reader.stats)
        assert (dz.read(window=Window(3, 170, 20, 30)) == v2.NODATA).all()
        assert dz.reader.stats["requests"] == before["requests"]


@pytest.mark.parametrize("via", ["path", "group"])
def test_golden_tile_decodes_from_zarr(tmp_path, via):
    import zarr
    codec = _bundle_codec()
    out = gz.parquet_to_zarr(TILE, tmp_path / "golden.georvq.zarr")
    src = out if via == "path" else zarr.open_group(str(out), mode="r")
    g = np.load(GOLDEN)
    gh, gw = TILE_GRID
    W = codec.window
    with v2.open_dataset(src, codec=codec) as ds:
        for i, r in enumerate(GOLDEN_RUNGS):
            a = ds.read(quality=r)[:, ::-1]            # stored bottom-up
            tiles = a.reshape(64, gh, W, gw, W).transpose(1, 3, 0, 2, 4) \
                .reshape(gh * gw, 64, W, W)
            assert _sha(tiles) == str(g["sha_int8"][i]), r
