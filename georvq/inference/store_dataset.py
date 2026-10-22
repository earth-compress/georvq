"""Reading georvq stores: a rasterio-like dataset over any release.

`StoreDataset` reads a store written by a release layer (georvq.ae,
georvq.tessera, georvq.s2) through its codec, with the georeferencing,
data type, nodata and band names the store's header gives, and with
rasterio's arguments and results. `open_dataset(store, spec, ...)` opens
one for a given release; `georvq.open` finds the release from the store
itself. `reconstruct` decodes a whole store into a GeoTIFF.

A store whose output lies on several grids (the header's
`output.grids`, Sentinel-2's 10, 20 and 60 m bands) reads as one
dataset per grid: a StoreDataset shows one grid at a time (the first by
default), `grid(name)` returns the dataset of another, and `read_grids`
decodes every grid of a region at once.

Nothing here imports torch: with an ONNX codec, reading needs only
numpy, zarr-python, rasterio and onnxruntime.
"""
from __future__ import annotations

import copy
import json
import math
import re

import numpy as np
from rasterio.transform import TransformMethodsMixin
from rasterio.windows import WindowMethodsMixin

from georvq.inference.records import RecordReader, span, tier_names
from georvq.inference.store_codec import (Q_OFFSET, Q_SCALE, parse_rung,
                                          quality_value)

STRIP_PX = 1024
_GDAL_ENV = {"GDAL_NUM_THREADS": "ALL_CPUS"}
_BOUND = re.compile(r"^\s*(\w+)\s*<=\s*([0-9.eE+-]+)\s*$")
_DEFAULT = object()


def flip_rows(t, height):
    """Affine of the same raster with its rows in the other order."""
    from affine import Affine
    return Affine(t.a, -t.b, t.c + t.b * height, t.d, -t.e, t.f + t.e * height)


def _convert(vals, dtype):
    """`vals` as `dtype`, as GDAL converts: integers clamped to the
    type's range, floats rounded first when the type is an integer."""
    dtype = np.dtype(dtype)
    if dtype == vals.dtype:
        return vals
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        if vals.dtype.kind in "iu":
            src = np.iinfo(vals.dtype)
            return np.clip(vals, max(info.min, src.min),
                           min(info.max, src.max)).astype(dtype)
        return np.clip(np.rint(vals), info.min, info.max).astype(dtype)
    return vals.astype(dtype, copy=False)


class StoreDataset(WindowMethodsMixin, TransformMethodsMixin):
    """Read-only view of a georvq store that reads like the rasterio
    DatasetReader of the SOURCE product.

    Georeferencing, shape and the read methods follow the source
    product, including its row order (a bottom-up source reads
    bottom-up), so code written against a rasterio dataset of the source
    can read this instead: `read`, `read_masks`, `dataset_mask`,
    `bounds`, `res`, `index`, `xy`, `window` and the other window helpers
    give the answers rasterio gives for the source, with decoded pixels
    in place of the source's. The data type, nodata value, band count
    and band names are those of the release's output, from the header's
    `output` (stores written before it existed give the product's).
    `read` takes a `quality` beyond rasterio's arguments; `quality`
    passed here is the default of every read. For a store with several
    output grids the dataset is that of one grid, `grid` (the header's
    first by default): its shape, transform, band count and names are
    the grid's, and `grids` names them all. `driver` is "GeoRVQZarr";
    `profile` and `meta` carry the source's driver (GTiff when the
    source records none), so either can be passed to
    `rasterio.open(..., "w")` to write a GeoTIFF. The store has no
    overviews, so a decimated read samples the full-resolution decode,
    where rasterio would read the source's overviews. `assets` are the
    store's files (georvq.inference.records.Assets).

    `store` is an open zarr.Group or a path or URL; `spec` the store's
    release; `reader_kw` go to georvq.inference.records.RecordReader
    (`cache`, `cache_bytes`, `trace`, `storage_options`). `read` needs a
    codec; the metadata, masks and assets do not. Without a codec,
    `resolver(header)` is called for one the first time `codec` is
    used. `options` are the release's output options
    (`spec.output_options`)."""

    driver = "GeoRVQZarr"
    mode = "r"

    def __init__(self, store, spec, codec=None, resolver=None, quality=None,
                 options=None, grid=None, **reader_kw):
        self.spec = spec
        self.options = spec.options(options or {})
        self.reader = RecordReader(store, **reader_kw)
        self.name = self.reader.source
        h = self.header = self.reader.header
        p = self.product = h["product"]
        if p.get("kind") != spec.product_kind:
            raise ValueError(f"{self.name}: product kind {p.get('kind')!r}, "
                             f"not {spec.product_kind}")
        fmt = (h.get("codec") or {}).get("format")
        if fmt != spec.codec_format:
            raise ValueError(f"{self.name}: codec format {fmt!r}, not "
                             f"{spec.codec_format}")
        self._codec = None
        self._resolver = resolver
        self._parent = None
        self.default_quality = quality
        self.product_height, self.product_width = (int(p["height"]),
                                                   int(p["width"]))
        out = self.output = h.get("output") or {
            "dtype": p["dtype"], "nodata": p["nodata"], "count": p["count"],
            "band_names": None}
        self.grids = tuple(g["name"] for g in out.get("grids") or ())
        self.nodata = out["nodata"]
        self._fill = 0 if self.nodata is None else self.nodata
        self.assets = self.reader.assets
        self.window_px = int(h["codec"]["window"])
        self.bottom_up = p["orientation"] == "bottom-up"
        self._set_grid(grid)
        self.rungs = [tuple(r["tuple"]) for r in h["rungs"]]
        self.full = tuple(int(t.split(":")[3]) for t in h["codec"]["tiers"])
        self.names = tier_names(len(self.full))
        self._meta = None
        self._crs = None
        self._closed = False
        self._quality_decoder = self.reader.quality_decoder or h["codec"].get(
            "decoder_id")
        if codec is not None:
            self.codec = codec

    def _set_grid(self, grid):
        """Make the dataset that of output grid `grid` (None: the first,
        or the product grid of a store with one grid)."""
        from affine import Affine
        out, p = self.output, self.product
        grids = out.get("grids")
        north = Affine(*p["transform"])
        if not grids:
            if grid is not None:
                raise ValueError(f"{self.name}: the store has one output "
                                 f"grid; grid={grid!r} names none")
            self.grid_name, f = None, 1
            self.height, self.width = self.product_height, self.product_width
            self.count = int(out["count"])
            self._band_names = out.get("band_names")
        else:
            by = {g["name"]: g for g in grids}
            if grid is not None and grid not in by:
                raise ValueError(f"{self.name}: no output grid {grid!r}; the "
                                 f"store has {list(by)}")
            g = by[grid] if grid is not None else grids[0]
            self.grid_name, f = g["name"], int(g["factor"])
            self.height = int(g.get("height", self.product_height // f))
            self.width = int(g.get("width", self.product_width // f))
            self.count = int(g["count"])
            self._band_names = g.get("band_names")
            if g.get("transform") is not None:
                north = Affine(*g["transform"])
            else:
                north = north * Affine.scale(f)
        if self.window_px % f:
            raise ValueError(f"{self.name}: grid factor {f} does not divide "
                             f"the {self.window_px} px window")
        self._factor = f
        self._wpx = self.window_px // f
        self.dtypes = (self.output["dtype"],) * self.count
        self.transform = (flip_rows(north, self.height) if self.bottom_up
                          else north)

    def grid(self, name) -> "StoreDataset":
        """The dataset of output grid `name`, sharing this one's reader
        and codec (closing either closes both)."""
        view = copy.copy(self)
        view._parent = self._parent or self
        view._set_grid(name)
        return view

    @property
    def codec(self):
        if self._parent is not None:
            c = self._parent.codec
            if c is not None and c is not self._codec:
                self.codec = c
            return self._codec
        if self._codec is None and self._resolver is not None:
            self.codec = self._resolver(self.header)
        return self._codec

    @codec.setter
    def codec(self, codec):
        bid = self.header["codec"]["bitstream_id"]
        if codec is not None and codec.bitstream_id != bid:
            raise ValueError(f"{self.name}: bitstream {bid}, codec holds "
                             f"{codec.bitstream_id}")
        self._codec = codec
        if codec is not None:
            self._use_decoder(codec)

    def _use_decoder(self, codec):
        """Quality values and rungs of the codec's decoder: the store's
        quality array for that decoder, or none when it has none, and
        then the codec's own rungs."""
        did = getattr(codec, "decoder_id", None)
        if did is None or did == self._quality_decoder:
            return
        r = self.reader
        r.select_quality(did if did in r.quality_decoders else None)
        self.header = r.header
        if r.layout.quality is not None:
            self.rungs = [tuple(x["tuple"]) for x in self.header["rungs"]]
        else:
            self.rungs = [tuple(x) for x in codec.rungs]
        self._quality_decoder = did

    @property
    def crs(self):
        if self._crs is None and self.product.get("crs_wkt"):
            from rasterio.crs import CRS
            self._crs = CRS.from_wkt(self.product["crs_wkt"])
        return self._crs

    @property
    def source_meta(self) -> dict:
        if self._meta is None:
            self._meta = json.loads(self.reader.aux) if self.reader.aux else {}
        return self._meta

    @property
    def profile(self) -> dict:
        prof = dict(self.source_meta.get("profile") or {
            "driver": "GTiff", "tiled": True, "blockxsize": 1024,
            "blockysize": 1024, "compress": "zstd", "interleave": "band"})
        prof.update(dtype=self.dtypes[0], nodata=self.nodata, width=self.width,
                    height=self.height, count=self.count, crs=self.crs,
                    transform=self.transform)
        return prof

    @property
    def descriptions(self):
        return tuple(self._band_names or
                     self.source_meta.get("descriptions") or
                     (None,) * self.count)

    def tags(self, bidx=0):
        m = self.source_meta
        return dict(m.get("tags", {}) if bidx == 0
                    else (m.get("band_tags") or [{}] * self.count)[bidx - 1])

    @property
    def block_shapes(self):
        p = self.profile
        return [(int(p.get("blockysize", 1024)), int(p.get("blockxsize", 1024)))
                ] * self.count

    def block_windows(self, bidx=0):
        from rasterio.windows import Window
        bh, bw = self.block_shapes[0]
        for i, r in enumerate(range(0, self.height, bh)):
            for j, c in enumerate(range(0, self.width, bw)):
                yield (i, j), Window(c, r, min(bw, self.width - c),
                                     min(bh, self.height - r))

    @property
    def shape(self):
        return self.height, self.width

    @property
    def indexes(self):
        return tuple(range(1, self.count + 1))

    @property
    def bounds(self):
        """As rasterio's DatasetBase.bounds; for a bottom-up raster
        `bottom` is above `top`, as rasterio reports it."""
        from rasterio.coords import BoundingBox
        a, b, c, d, e, f = tuple(self.transform)[:6]
        if b == d == 0:
            return BoundingBox(c, f + e * self.height, c + a * self.width, f)
        xs, ys = zip(*(self.transform * p for p in
                       ((0, 0), (0, self.height), (self.width, self.height),
                        (self.width, 0))))
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))

    @property
    def res(self):
        a, b, _, d, e, _ = tuple(self.transform)[:6]
        return math.sqrt(a * a + d * d), math.sqrt(b * b + e * e)

    @property
    def meta(self):
        return {"driver": self.profile.get("driver", "GTiff"),
                "dtype": self.dtypes[0], "nodata": self.nodata,
                "width": self.width, "height": self.height,
                "count": self.count, "crs": self.crs,
                "transform": self.transform}

    def overviews(self, bidx=1):
        """The file holds no overviews."""
        return []

    @property
    def closed(self):
        return self._closed

    def close(self):
        self.reader.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- quality ----

    def _depth_map(self, quality, wy0, wy1, wx0, wx1) -> np.ndarray:
        """(rows, cols) rung index per window. A rung or None (full depth)
        that is not a rung of the file raises ValueError; a bound resolves
        to the last rung, which is full depth, where no rung meets it."""
        shape = (wy1 - wy0, wx1 - wx0)
        if quality is None:
            t = self.full
        else:
            m = _BOUND.match(quality) if isinstance(quality, str) else None
            if m is None:
                try:
                    t = parse_rung(quality, self.full)
                except (TypeError, ValueError) as err:
                    if "does not fit" in str(err):
                        raise
                    raise ValueError(
                        f"quality {quality!r}: expected a rung such as "
                        "'8-8-2', '8,8,2' or (8, 8, 2), a bound such as "
                        f"'p99<=0.02' on one of the stats "
                        f"{list(self.header.get('quality', {}).get('stats', self.spec.quality_stats))}"
                        ", or None for full depth") from None
            else:
                return self._resolve_bound(m.group(1), float(m.group(2)),
                                           wy0, wy1, wx0, wx1)
        if t not in self.rungs:
            raise ValueError(f"{t} is not a rung of this file: "
                             f"{['-'.join(map(str, r)) for r in self.rungs]}")
        return np.full(shape, self.rungs.index(t), np.int64)

    def _resolve_bound(self, stat, bound, wy0, wy1, wx0, wx1):
        r = self.reader
        if r.quality_decoder != self._quality_decoder:
            raise ValueError(
                f"{self.name}: no quality for decoder {self._quality_decoder}; "
                f"the store has quality for "
                f"{sorted(r.quality_decoders) or 'no decoder'}")
        if r.layout.quality is None:
            raise ValueError(f"{self.name}: no quality column to resolve a "
                             "bound against")
        stats = self.header["quality"]["stats"]
        if stat not in stats:
            raise ValueError(f"quality stat {stat!r} not in {stats}")
        n = len(self.rungs)
        q = self.quality_codes(wy0, wy1, wx0, wx1)
        k = stats.index(stat)
        qh = self.header["quality"]
        ok = quality_value(q[..., k * n:(k + 1) * n],
                           qh.get("scale", Q_SCALE),
                           qh.get("offset", Q_OFFSET)) <= bound
        first = np.where(ok.any(-1), ok.argmax(-1), n - 1)
        return first.astype(np.int64)

    def quality_codes(self, wy0, wy1, wx0, wx1) -> np.ndarray:
        """(rows, cols, stats * rungs) quality codes of windows
        [wy0:wy1, wx0:wx1], in the north-up window grid."""
        return self.reader.read(wy0, wy1, wx0, wx1, depths=[0] * len(self.full),
                                base=False, mask=False, quality=True)["quality"]

    def window_rungs(self, quality, window=None) -> np.ndarray:
        """Rung index chosen for each coded window under `window`."""
        ny0, ny1, nx0, nx1 = self._north_px(window)
        W = self._wpx
        return self._depth_map(quality, ny0 // W, -(-ny1 // W), nx0 // W,
                               -(-nx1 // W))

    # ---- read ----

    def _north_px(self, window):
        """Source-pixel window -> north-up pixel bounds (y0, y1, x0, x1),
        offsets and lengths rounded."""
        if window is None:
            r0, r1, c0, c1 = 0, self.height, 0, self.width
        elif hasattr(window, "row_off"):
            r0, c0 = int(round(window.row_off)), int(round(window.col_off))
            r1, c1 = r0 + int(round(window.height)), c0 + int(round(window.width))
        else:
            (r0, r1), (c0, c1) = window
        if not (0 <= r0 < r1 <= self.height and 0 <= c0 < c1 <= self.width):
            raise ValueError(f"window rows {r0}:{r1} cols {c0}:{c1} outside "
                             f"the {self.height}x{self.width} raster")
        if self.bottom_up:
            r0, r1 = self.height - r1, self.height - r0
        return r0, r1, c0, c1

    def _check_open(self):
        if self._closed:
            from rasterio.errors import RasterioIOError
            raise RasterioIOError(f"Dataset is closed: {self.name}")

    def _bands(self, indexes):
        """(1-based band list, whether the result is 2-D), validated as
        rasterio validates them."""
        if indexes is None:
            return list(self.indexes), False
        two_d = isinstance(indexes, (int, np.integer))
        bands = [indexes] if two_d else list(indexes)
        if not bands:
            raise ValueError("No indexes to read")
        for b in bands:
            if b not in self.indexes:
                raise IndexError(f"band index {b} out of range (not in "
                                 f"{self.indexes})")
        return [int(b) for b in bands], two_d

    def _window(self, window, boundless):
        """`window` as a rasterio Window, cropped to the raster unless
        `boundless`, as rasterio's read does."""
        from rasterio.windows import Window
        if window is None:
            return Window(0, 0, self.width, self.height)
        if not isinstance(window, Window):
            window = Window.from_slices(*window, height=self.height,
                                        width=self.width, boundless=boundless)
        if not boundless:
            window = window.crop(self.height, self.width)
        return window

    @staticmethod
    def _out_hw(window, n_bands, out, out_shape, resampling):
        """Output (rows, cols): `out`'s, `out_shape`'s, or the window's
        lengths rounded as rasterio rounds them."""
        from rasterio.enums import Resampling
        from rasterio.errors import DatasetIOShapeError
        if out is not None and out_shape is not None:
            raise ValueError("out and out_shape are exclusive")
        shape = out.shape if out is not None else out_shape
        if shape is None:
            hw = (math.floor(window.height + 0.5), math.floor(window.width + 0.5))
        else:
            shape = tuple(int(v) for v in shape)
            if len(shape) == 3 and shape[0] != n_bands:
                raise DatasetIOShapeError("Dataset indexes and destination "
                                          "buffer are mismatched")
            hw = shape[-2:]
        exact = (hw == (window.height, window.width)
                 and float(window.row_off).is_integer()
                 and float(window.col_off).is_integer())
        if resampling not in (None, Resampling.nearest) and not exact:
            raise NotImplementedError(
                f"resampling {Resampling(resampling).name}: only nearest is "
                "implemented")
        return hw

    def _gather(self, window, oh, ow, fetch, k, fill):
        """(k, oh, ow) samples of `window` at nearest-neighbour positions,
        as GDAL picks them, and the (oh, ow) mask of those inside the
        raster. `fetch(y0, y1, x0, x1)` returns (k, y1 - y0, x1 - x0) for
        a north-up pixel block; it is called on runs of window rows of at
        most STRIP_PX rows that hold a sample. Samples outside the raster
        are `fill`."""
        def sample(off, length, n, size):
            # An output pixel whose footprint overlaps the raster takes the
            # nearest raster pixel, clamped at the edge, as GDAL's VRT
            # does for a boundless read.
            step, j = length / max(n, 1), np.arange(n)
            at = np.floor(off + (j + 0.5) * step + 1e-10).astype(np.int64)
            ins = (off + (j + 1) * step > 1e-10) & (off + j * step < size - 1e-10)
            return np.clip(at, 0, size - 1), ins
        ry, iy = sample(window.row_off, window.height, oh, self.height)
        cx, ix = sample(window.col_off, window.width, ow, self.width)
        inside = iy[:, None] & ix[None, :]
        out = None
        if iy.any() and ix.any():
            W = self._wpx
            ny = self.height - 1 - ry if self.bottom_up else ry
            jo = np.nonzero(ix)[0]
            c0, c1 = int(cx[jo].min()), int(cx[jo].max()) + 1
            cols = cx[jo] - c0
            dense = ((np.diff(cols) == 1).all() and (np.diff(jo) == 1).all()
                     and (np.diff(np.nonzero(iy)[0]) == 1).all()
                     and (np.abs(np.diff(ny[iy])) == 1).all())
            need = np.unique(ny[iy] // W)
            runs = np.split(need, np.nonzero(np.diff(need) > 1)[0] + 1)
            step = max(1, STRIP_PX // W)
            for run in runs:
                for i in range(0, len(run), step):
                    y0 = int(run[i]) * W
                    y1 = min(self.height, (int(run[min(i + step, len(run)) - 1]) + 1) * W)
                    blk = fetch(y0, y1, c0, c1)
                    if out is None:
                        out = np.full((k, oh, ow), fill, blk.dtype)
                    io = np.nonzero(iy & (ny >= y0) & (ny < y1))[0]
                    rows = ny[io] - y0
                    if dense:
                        # unresampled read: a run of the block's rows, in
                        # order or reversed, copied without a gather
                        seg = blk[:, rows.min():rows.max() + 1]
                        if len(rows) > 1 and rows[1] < rows[0]:
                            seg = seg[:, ::-1]
                        out[:, io[0]:io[-1] + 1, jo[0]:jo[-1] + 1] = seg
                    else:
                        out[:, io[:, None], jo[None, :]] = (
                            blk[:, rows][:, :, cols])
        return out, inside

    def _decode_north(self, ny0, ny1, nx0, nx1, quality, batch):
        """(count, rows, cols) output pixels of north-up pixels [ny0:ny1,
        nx0:nx1] of the dataset's grid."""
        W = self._wpx
        wy0, wy1 = ny0 // W, -(-ny1 // W)
        wx0, wx1 = nx0 // W, -(-nx1 // W)
        out = self._decode_windows(wy0, wy1, wx0, wx1, quality, batch)
        if self.grid_name is not None:
            out = out[self.grid_name]
        return out[:, ny0 - wy0 * W:ny1 - wy0 * W, nx0 - wx0 * W:nx1 - wx0 * W]

    def _grid_shapes(self):
        """{grid: (count, factor)}, or None for a store with one grid."""
        grids = self.output.get("grids")
        if not grids:
            return None
        return {g["name"]: (int(g["count"]), int(g["factor"])) for g in grids}

    def _decode_windows(self, wy0, wy1, wx0, wx1, quality, batch):
        """Output pixels of the whole windows [wy0:wy1, wx0:wx1]: (count,
        rows, cols), or for a store with several grids {grid: (count,
        rows, cols)} on each grid."""
        W = self.window_px
        gh, gw = wy1 - wy0, wx1 - wx0
        trace = self.reader.trace
        with span(trace, "cpu", "resolve rungs"):
            rung_of = self._depth_map(quality, wy0, wy1, wx0, wx1)
        depth_of = np.array(self.rungs)[rung_of]           # (gh, gw, tiers)

        # One fetch covers the mask, base and every level any window
        # needs. Windows without valid pixels resolve to the cheapest
        # rung, so they do not raise the depth read.
        need = depth_of.reshape(-1, depth_of.shape[-1]).max(0)
        got = self.reader.read(wy0, wy1, wx0, wx1, depths=[int(v) for v in need])
        vwin = got["valid"].reshape(gh, W, gw, W).transpose(0, 2, 1, 3)
        live = vwin.any((2, 3))

        dt = np.dtype(self.output["dtype"])
        shapes = self._grid_shapes()
        if shapes is None:
            out = np.full((self.count, gh * W, gw * W), self._fill, dt)
        else:
            out = {n: np.full((c, gh * W // f, gw * W // f), self._fill, dt)
                   for n, (c, f) in shapes.items()}
        if live.any():
            base = got["base"]
            tiers = [got.get(name) for name in self.names[1:]]
            for k in np.unique(rung_of[live]):
                r = self.rungs[k]
                ys, xs = np.nonzero(live & (rung_of == k))
                for i in range(0, len(ys), batch):
                    y, x = ys[i:i + batch], xs[i:i + batch]
                    fin = [tiers[t - 1][y, x, :r[t]] if r[t] else None
                           for t in range(1, len(r))]
                    kw = dict(self.options)
                    if trace is not None:
                        kw["trace"] = trace
                    dec = self.codec.decode_records(base[y, x], fin, r,
                                                    vwin[y, x], **kw)
                    with span(trace, "cpu", "assemble"):
                        if shapes is None:
                            _place(out, dec, y, x, W)
                        else:
                            for n, (_, f) in shapes.items():
                                _place(out[n], dec[n], y, x, W // f)
        return out

    def read_grids(self, window=None, quality=_DEFAULT, batch=256) -> dict:
        """Decoded pixels of every output grid of a region, decoded once:
        {grid name: (count, rows, cols)}. `window` is a rasterio Window
        or ((row0, row1), (col0, col1)) in north-up pixels of the
        product grid (the finest; None for the whole product), and each
        grid's array covers the pixels of that grid the window touches.
        `quality` as for `read`."""
        self._check_open()
        shapes = self._grid_shapes()
        if shapes is None:
            raise ValueError(f"{self.name}: the store has one output grid; "
                             "use read")
        if quality is _DEFAULT:
            quality = self.default_quality
        if self.codec is None:
            raise ValueError("reading pixels needs a codec: open the store "
                             "with georvq.open, which finds the store's "
                             "bundle, or pass codec=")
        H, Wd = self.product_height, self.product_width
        if window is None:
            r0, r1, c0, c1 = 0, H, 0, Wd
        elif hasattr(window, "row_off"):
            r0, c0 = int(window.row_off), int(window.col_off)
            r1, c1 = r0 + int(window.height), c0 + int(window.width)
        else:
            (r0, r1), (c0, c1) = window
        if not (0 <= r0 < r1 <= H and 0 <= c0 < c1 <= Wd):
            raise ValueError(f"window rows {r0}:{r1} cols {c0}:{c1} outside "
                             f"the {H}x{Wd} product")
        W = self.window_px
        wy0, wy1, wx0, wx1 = r0 // W, -(-r1 // W), c0 // W, -(-c1 // W)
        got = self._decode_windows(wy0, wy1, wx0, wx1, quality, batch)
        res = {}
        for n, (_, f) in shapes.items():
            o = wy0 * W // f, wx0 * W // f
            res[n] = got[n][:, r0 // f - o[0]:-(-r1 // f) - o[0],
                            c0 // f - o[1]:-(-c1 // f) - o[1]]
        return res

    def read(self, indexes=None, out=None, window=None, masked=False,
             out_shape=None, boundless=False, resampling=None,
             fill_value=None, out_dtype=None, *, quality=_DEFAULT, batch=256):
        """Decoded pixels, with rasterio's DatasetReader.read arguments
        and results: (bands, rows, cols), or (rows, cols) for an int
        `indexes` (1-based, validated against 1..count with IndexError).

        A window that extends off the raster is cropped to it unless
        `boundless`, where pixels outside the raster are `fill_value`
        (default nodata). Without `boundless`, a `fill_value` replaces
        nodata pixels, as in rasterio. Fractional windows and `out_shape`
        / `out` sample at nearest-neighbour positions as GDAL does; other
        `resampling` methods raise NotImplementedError. `masked` returns
        a numpy masked array masking nodata pixels and pixels outside the
        raster; for a boundless read with a `fill_value` it masks, as
        rasterio does, the pixels outside the raster and those equal to
        `fill_value`, and not nodata pixels. `out_dtype`
        converts as GDAL does, clamping integers to the type's range.

        `quality`: a rung ("8-8-2" or a tuple), a bound on a quality stat
        ("p99<=0.02", resolved per window to the cheapest rung meeting
        it, full depth where none does), or None for full depth; by
        default the dataset's `default_quality`."""
        self._check_open()
        if quality is _DEFAULT:
            quality = self.default_quality
        bands, two_d = self._bands(indexes)
        if self.codec is None:
            raise ValueError("reading pixels needs a codec: open the store "
                             "with georvq.open, which finds the store's "
                             "bundle, or pass codec=")
        win = self._window(window, boundless)
        oh, ow = self._out_hw(win, len(bands), out, out_shape, resampling)
        sel = [b - 1 for b in bands]
        all_bands = sel == list(range(self.count))

        def fetch(y0, y1, x0, x1):
            a = self._decode_north(y0, y1, x0, x1, quality, batch)
            return a if all_bands else a[sel]

        own = np.dtype(self.dtypes[0])
        vals, inside = self._gather(win, oh, ow, fetch, len(bands), self._fill)
        if vals is None:
            vals = np.full((len(bands), oh, ow), self._fill, own)
        nod = (vals == self.nodata if self.nodata is not None
               else np.zeros(vals.shape, bool))
        dtype = np.dtype(out.dtype if out is not None else out_dtype or own)
        res = _convert(vals, dtype)
        if fill_value is not None:
            if boundless:
                res[:, ~inside] = fill_value
            elif not masked:
                res[nod] = fill_value
        if masked:
            # rasterio reads a boundless window through a VRT whose nodata
            # is the fill_value: it masks the pixels outside the raster and
            # those equal to fill_value, and leaves nodata pixels unmasked
            mask = (~inside[None] | (res == fill_value)
                    if boundless and fill_value is not None else nod)
            res = np.ma.array(res, mask=mask, fill_value=(
                fill_value if fill_value is not None else self.nodata))
        if two_d:
            res = res[0]
        if out is not None:
            out[...] = np.ma.getdata(res).reshape(out.shape)
            return (np.ma.array(out, mask=np.ma.getmaskarray(res).reshape(
                out.shape), fill_value=res.fill_value) if masked else out)
        return res if masked else np.ascontiguousarray(res)

    def read_masks(self, indexes=None, out=None, out_shape=None, window=None,
                   boundless=False, resampling=None):
        """uint8 masks, 255 valid and 0 nodata or outside the raster, as
        rasterio's read_masks: (bands, rows, cols), or (rows, cols) for an
        int `indexes`. Every band shares the file's pixel validity. Needs
        no codec."""
        self._check_open()
        bands, two_d = self._bands(indexes)
        win = self._window(window, boundless)
        oh, ow = self._out_hw(win, len(bands), out, out_shape, resampling)
        m = self._mask(win, oh, ow)
        res = np.broadcast_to(m, (len(bands), oh, ow))
        if two_d:
            res = res[0]
        if out is not None:
            out[...] = res.reshape(out.shape)
            return out
        return np.ascontiguousarray(res)

    def dataset_mask(self, out=None, out_shape=None, window=None,
                     boundless=False, resampling=None):
        """uint8 (rows, cols), 255 where any band is valid, as rasterio's
        dataset_mask. Needs no codec."""
        self._check_open()
        win = self._window(window, boundless)
        oh, ow = self._out_hw(win, 1, out, out_shape, resampling)
        res = self._mask(win, oh, ow)
        if out is not None:
            out[...] = res
            return out
        return res

    def _mask(self, win, oh, ow):
        """uint8 (oh, ow) validity samples of `win`, 255 valid. On a grid
        coarser than the product's a pixel is valid when every product
        pixel it covers is."""
        f = self._factor

        def fetch(y0, y1, x0, x1):
            if f == 1:
                return self.reader.read_mask(y0, y1, x0, x1)[None]
            m = self.reader.read_mask(y0 * f, y1 * f, x0 * f, x1 * f)
            return m.reshape(y1 - y0, f, x1 - x0, f).all((1, 3))[None]
        m, _ = self._gather(win, oh, ow, fetch, 1, False)
        if m is None:
            return np.zeros((oh, ow), np.uint8)
        return m[0].astype(np.uint8) * np.uint8(255)


def _place(out, dec, y, x, W):
    """Copy decoded windows dec (n, C, W, W) into out at window rows y
    and columns x."""
    for d, yy, xx in zip(dec, y, x):
        out[:, yy * W:(yy + 1) * W, xx * W:(xx + 1) * W] = d


def open_dataset(store, spec, codec=None, resolver=None, quality=None,
                 options=None, grid=None, **reader_kw) -> StoreDataset:
    """A StoreDataset of `store` (an open zarr.Group, or a local path or a
    URL that zarr-python opens through fsspec) for the release `spec`.
    `codec` reads pixels; without one, `resolver(header)` is called for
    one when pixels are first read. `grid` picks an output grid of a
    store with several."""
    return StoreDataset(store, spec, codec=codec, resolver=resolver,
                        quality=quality, options=options, grid=grid,
                        **reader_kw)


def reconstruct(src, out_path, quality=None, rows=1024, creation_options=None):
    """Decode a whole store into a GeoTIFF with the source's profile,
    tags and band descriptions. `src` is an open StoreDataset."""
    import rasterio
    from rasterio.windows import Window
    prof = src.profile
    with rasterio.Env(**_GDAL_ENV), rasterio.open(
            out_path, "w", **prof, num_threads="all_cpus",
            **(creation_options or {})) as dst:
        dst.update_tags(**src.tags())
        for b in range(1, src.count + 1):
            if src.tags(b):
                dst.update_tags(b, **src.tags(b))
            if src.descriptions[b - 1] is not None:
                dst.set_band_description(b, src.descriptions[b - 1])
        for r0 in range(0, src.height, rows):
            win = Window(0, r0, src.width, min(rows, src.height - r0))
            dst.write(src.read(window=win, quality=quality), window=win)
