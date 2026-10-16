"""AE v2 stores opened as a zarr.Group decode as their path does, on the
synthetic tile of test_ae_v2 (with an empty chunk and quality-bound
reads) and on the golden tile with the released bundle; quality follows
the decoder a dataset decodes with."""
import os
import shutil

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
zarr = pytest.importorskip("zarr")
from rasterio.windows import Window  # noqa: E402

from georvq.ae import v2  # noqa: E402
from tests.test_ae_v2 import CHUNK_PX, H, setup  # noqa: E402,F401
from tests.test_ae_v2_bundle import (GOLDEN, GOLDEN_RUNGS, TILE,  # noqa: E402
                                     TILE_GRID, _bundle_codec, _sha)


@pytest.fixture(scope="module")
def corner(setup):  # noqa: F811
    """The fixture tile with its corner widened to cover fine chunk
    (0, 0), so the store has an empty chunk."""
    tmp = setup["tmp"]
    src, out = tmp / "zcorner.tif", tmp / "zcorner.georvq.zarr"
    with rasterio.open(setup["src"]) as s:
        a, prof = s.read(), s.profile
    a[:, H - 32:, :50] = v2.NODATA
    with rasterio.open(src, "w", **prof) as d:
        d.write(a)
    v2.encode_tile(src, out, setup["codec"], stats=True, chunk_px=CHUNK_PX,
                   strip_px=48, batch=256, source="corner.tif")
    return out


def test_a_group_decodes_as_its_path(setup, corner):  # noqa: F811
    codec = setup["codec"]
    g = zarr.open_group(str(corner), mode="r")
    with v2.open_dataset(corner, codec=codec) as dp, \
            v2.open_dataset(g, codec=codec) as dz:
        assert dz.driver == "GeoRVQZarr" and dz.reader.empty["fine"] == {(0, 0)}
        assert dz.profile == dp.profile and dz.tags() == dp.tags()
        cases = [(None, None), (Window(3, 170, 20, 30), dp.rungs[-1]),
                 (Window(10, 20, 100, 90), dp.rungs[0]),
                 (None, "p99<=0.05"), (Window(0, 0, 168, 200), "mean<=0.02")]
        for win, q in cases:
            kw = {} if q is None else {"quality": q}
            assert np.array_equal(dp.read(window=win, **kw),
                                  dz.read(window=win, **kw)), (win, q)
        assert np.array_equal(dp.dataset_mask(), dz.dataset_mask())


def _other_decoder(codec, did="f" * 16):
    """The codec under another decoder id, as a later decoder release of
    the same bitstream would be."""
    import copy
    c = copy.copy(codec)
    c.decoder_id = did
    return c


def _add_quality(store, did):
    """A quality array for decoder `did`: the store's own values under the
    new id (same rungs), listed in the quality group."""
    g = zarr.open_group(str(store), mode="r+")
    own = g["quality"].attrs["decoders"][0]
    q0 = g["quality"][own]
    info = dict(q0.attrs["georvq_quality"], decoder_id=did)
    a = g["quality"].create_array(did, shape=q0.shape, dtype="uint8",
                                  chunks=q0.chunks, compressors=None, fill_value=0,
                                  attributes={**dict(q0.attrs),
                                              "georvq_quality": info})
    a[...] = q0[...]
    g["quality"].update_attributes(
        {"decoders": list(g["quality"].attrs["decoders"]) + [did]})
    zarr.consolidate_metadata(str(store))


def test_quality_follows_the_decoder(setup, corner, tmp_path):  # noqa: F811
    codec = setup["codec"]
    other = _other_decoder(codec)
    win, bound = Window(10, 20, 100, 90), "p99<=0.05"
    with v2.open_dataset(corner, codec=codec) as ds:
        want_rung = ds.read(window=win, quality=ds.rungs[0])
        want_bound = ds.read(window=win, quality=bound)
    # a decoder the store holds no quality for: rungs read, bounds refused
    with v2.open_dataset(corner, codec=other) as ds:
        assert ds.reader.quality_decoder is None
        assert np.array_equal(ds.read(window=win, quality=ds.rungs[0]), want_rung)
        with pytest.raises(ValueError, match="no quality for decoder ffff"):
            ds.read(window=win, quality=bound)
    # once the store holds quality for it, the decoder's own array is used
    z2 = tmp_path / "z2.georvq.zarr"
    shutil.copytree(corner, z2)
    _add_quality(z2, "f" * 16)
    with v2.open_dataset(z2, codec=other) as ds:
        assert ds.reader.quality_decoder == "f" * 16
        assert np.array_equal(ds.read(window=win, quality=bound), want_bound)
    with v2.open_dataset(z2, codec=codec) as ds:
        assert ds.reader.quality_decoder == codec.decoder_id
        assert np.array_equal(ds.read(window=win, quality=bound), want_bound)


def test_golden_tile_decodes_through_a_group():
    if not os.path.exists(TILE):
        pytest.skip("tile fixture not recorded")
    codec = _bundle_codec()
    g = np.load(GOLDEN)
    gh, gw = TILE_GRID
    W = codec.window
    with v2.open_dataset(zarr.open_group(TILE, mode="r"), codec=codec) as ds:
        for i, r in enumerate(GOLDEN_RUNGS):
            a = ds.read(quality=r)[:, ::-1]            # stored bottom-up
            tiles = a.reshape(64, gh, W, gw, W).transpose(1, 3, 0, 2, 4) \
                .reshape(gh * gw, 64, W, W)
            assert _sha(tiles) == str(g["sha_int8"][i]), r
