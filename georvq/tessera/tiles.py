"""Tessera v1.1 tiles, read from the dClimate run's Zarr store.

The v1.1 embeddings (the global run produced and hosted by dClimate,
which geotessera 0.11 reads by default) are published as one Zarr store
with a group per UTM zone, `utmNN`, holding int8 (time, 128, y, x)
embeddings and float32 (time, y, x) scales on the zone's 10 m grid, with
x = q * scale and a NaN scale marking a pixel with no embedding. Every
zone's grid is in the northern CRS of its zone (EPSG:326NN), so southern
tiles have negative northings. The store's attributes name the model
that made the embeddings (`geoemb:model`), which is checked here so that
embeddings of another version are never read as v1.1.

The unit of a release store is a geotessera tile, the 0.1 degree cell
centred at (lon, lat) in one year, cut as the chip-store builder
(scripts/build_store_tessera.py) cuts it: from the grid of the zone that
holds the cell's centre, the pixel envelope of the cell's footprint
(`tile_window`). NaN or infinite scales become the v1 nodata convention,
an all-zero vector with a zero scale (`clean_nodata`), so that the int8
codes are the source's unchanged.

`TesseraTile.read` reads one tile from the store (by default the public
copy on Source Cooperative) as a georvq.inference.store_codec
.ProductSource, and `TesseraTile.from_arrays` makes the same object from
arrays already read, such as a local copy. Its `source` is the URL of
the zone's embeddings array followed by the slice read from it.
"""
from __future__ import annotations

import math
import os

import numpy as np

V11_STORE = "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate"
# the bucket behind the Source Cooperative URL; a whole-tile read issues
# about 2,500 chunk requests, which the data.source.coop gateway serves
# too slowly to be usable, so reads go here, as geotessera's do
SOURCE_COOP = "https://data.source.coop/"
SOURCE_COOP_BUCKET = ("https://s3.us-west-2.amazonaws.com/"
                      "us-west-2.opendata.source.coop/")
# what `geoemb:model` of the v1.1 store reads; another value is another
# embedding space, which a v1.1 codec must not be given
V11_MODEL = "https://geotessera.org/model/1.1"
EMBEDDINGS = {"name": "TESSERA", "version": "1.1", "run": "dclimate",
              "model": V11_MODEL, "store": V11_STORE}
BANDS = 128


def utm_zone(lon: float) -> int:
    """UTM zone (1-60) of a longitude; a 0.1 degree tile belongs to the
    zone containing its centre."""
    return max(1, min(60, int(math.floor((lon + 180) / 6)) + 1))


def tile_name(lon, lat) -> str:
    return f"grid_{lon:.2f}_{lat:.2f}"


def tile_window(lon, lat, west, top, px, to_utm):
    """(r0, c0, h, w) pixel envelope of the 0.1 degree tile centred at
    (lon, lat) on a zone grid with origin (west, top) and pixel size px;
    to_utm maps lon/lat arrays to the grid's CRS."""
    d = np.array([-0.05, 0.0, 0.05])
    lo, la = np.meshgrid(lon + d, lat + d)
    e, n = to_utm(lo.ravel(), la.ravel())
    r0 = int(math.floor((top - np.max(n)) / px))
    r1 = int(math.ceil((top - np.min(n)) / px))
    c0 = int(math.floor((np.min(e) - west) / px))
    c1 = int(math.ceil((np.max(e) - west) / px))
    return r0, c0, r1 - r0, c1 - c0


def clean_nodata(q, s):
    """Map v1.1 nodata (NaN or infinite scale) onto the v1 convention:
    all-zero vector and zero scale. q (128, H, W) int8, s (H, W) float32;
    both are modified in place and returned."""
    bad = ~np.isfinite(s)
    s[bad] = 0.0
    q[:, bad] = 0
    return q, s


def open_store(store=V11_STORE):
    """The v1.1 store as a zarr group (a URL, a local path or an open
    group), refused when its attributes name another embedding model. A
    Source Cooperative URL is read from the bucket behind it."""
    import zarr
    if hasattr(store, "attrs") and hasattr(store, "group_keys"):
        g = store
    else:
        url = os.fspath(store)
        if url.startswith(SOURCE_COOP):
            url = SOURCE_COOP_BUCKET + url[len(SOURCE_COOP):]
        g = zarr.open_group(url, mode="r")
    model = dict(g.attrs).get("geoemb:model")
    if model != V11_MODEL:
        raise ValueError(f"{store}: embeddings of model {model!r}, not "
                         f"{V11_MODEL} (Tessera v1.1)")
    return g


def zone_grid(group, zone: int) -> dict:
    """The arrays and grid of zone `zone` of an open v1.1 store."""
    g = group[f"utm{zone:02d}"]
    t = [float(v) for v in g.attrs["spatial:transform"]]
    code = g.attrs.get("proj:code", f"EPSG:{32600 + zone}")
    return {"emb": g["embeddings"], "scales": g["scales"], "west": t[2],
            "top": t[5], "px": t[0], "years": [int(y) for y in g["time"][:]],
            "epsg": int(str(code).split(":")[1])}


def source_url(store, zone, time_index, r0, c0, h, w) -> str:
    """The embeddings array a tile is read from, with the slice read."""
    url = store if isinstance(store, str) else "zarr-group"
    return (f"{url.rstrip('/')}/utm{zone:02d}/embeddings"
            f"[{time_index},:,{r0}:{r0 + h},{c0}:{c0 + w}]")


class TesseraTile:
    """One Tessera v1.1 tile (a 0.1 degree cell, one year) as a
    georvq.inference.store_codec.ProductSource: `q` int8 (128, H, W),
    `s` float32 (H, W), north-up on the zone's 10 m grid."""

    orientation = "north-up"

    def __init__(self, year, lon, lat, q, s, epsg, transform, source,
                 window, zone):
        from georvq.inference.store_codec import crs_from_epsg
        q = np.asarray(q)
        s = np.asarray(s, np.float32)
        if q.dtype != np.int8 or q.ndim != 3 or q.shape[0] != BANDS:
            raise ValueError(f"expected int8 ({BANDS}, H, W) codes, got "
                             f"{q.dtype} {q.shape}")
        if s.shape != q.shape[1:]:
            raise ValueError(f"scale plane {s.shape} does not match the "
                             f"codes {q.shape}")
        self.year, self.lon, self.lat = int(year), float(lon), float(lat)
        self.q, self.s = clean_nodata(np.array(q), np.array(s))
        self.height, self.width = self.s.shape
        self.epsg, self.zone = int(epsg), int(zone)
        self.crs = crs_from_epsg(self.epsg)
        self.transform = tuple(float(v) for v in transform)
        self.source = source
        name = tile_name(self.lon, self.lat)
        self.product_id = f"{self.year}/{name}"
        r0, c0 = (int(v) for v in window)
        self.window = (r0, c0)
        self.product_meta = {
            "count": BANDS, "dtype": "int8", "nodata": None,
            "scale": "float32 per pixel, x = q * scale",
            "year": self.year, "lon": self.lon, "lat": self.lat,
            "utm_zone": self.zone,
            "zone_window": {"row": r0, "col": c0, "height": self.height,
                            "width": self.width},
            "embeddings": dict(EMBEDDINGS)}
        self.aux = None

    @classmethod
    def read(cls, year, lon, lat, store=V11_STORE):
        """Read the tile centred at (lon, lat) in `year` from the v1.1
        store (a URL, a local path or an open zarr group)."""
        from pyproj import Transformer
        g = open_store(store)
        z = utm_zone(lon)
        zg = zone_grid(g, z)
        if int(year) not in zg["years"]:
            raise ValueError(f"utm{z:02d} has no {year}; it has "
                             f"{zg['years']}")
        ti = zg["years"].index(int(year))
        fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{zg['epsg']}",
                                   always_xy=True)
        r0, c0, h, w = tile_window(lon, lat, zg["west"], zg["top"], zg["px"],
                                   fwd.transform)
        s = np.asarray(zg["scales"][ti, r0:r0 + h, c0:c0 + w], np.float32)
        q = np.asarray(zg["emb"][ti, :, r0:r0 + h, c0:c0 + w])
        return cls(year, lon, lat, q, s, zg["epsg"],
                   cls.zone_transform(zg, r0, c0),
                   source_url(store, z, ti, r0, c0, h, w), (r0, c0), z)

    @staticmethod
    def zone_transform(zg, r0, c0) -> tuple:
        px = zg["px"]
        return (px, 0.0, zg["west"] + c0 * px, 0.0, -px, zg["top"] - r0 * px)

    @classmethod
    def from_arrays(cls, year, lon, lat, q, s, epsg, transform, source,
                    window, zone=None):
        """A tile from arrays already read, `source` naming where from."""
        return cls(year, lon, lat, q, s, epsg, transform, source, window,
                   utm_zone(lon) if zone is None else zone)

    def read_strip(self, y0, y1):
        return self.q[:, y0:y1], self.s[y0:y1]
