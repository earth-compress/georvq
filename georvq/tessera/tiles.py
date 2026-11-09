"""Tessera v1.1 embeddings, read from the dClimate run's Zarr store.

The v1.1 embeddings (the global run produced and hosted by dClimate,
which geotessera 0.11 reads by default) are published as one Zarr store
with a group per UTM zone, `utmNN`, holding int8 (time, 128, y, x)
embeddings and float32 (time, y, x) scales on the zone's 10 m grid, with
x = q * scale and a NaN scale marking a pixel with no embedding. Every
zone's grid is in the northern CRS of its zone (EPSG:326NN), so southern
tiles have negative northings. Both arrays are sharded in 4096 x 4096 px
shards of 32 x 32 px chunks. The store's attributes name the model that
made the embeddings (`geoemb:model`), which is checked here so that
embeddings of another version are never read as v1.1.

The unit of a release store is one such shard of one year
(`TesseraShard`): the store mirrors the source's layout, as the AE
release mirrors its source tiles, and is named by year, zone and shard
row and column (`shard_id`). NaN or infinite scales become the v1
nodata convention, an all-zero vector with a zero scale
(`clean_nodata`), so that the int8 codes are the source's unchanged.

A geotessera tile is the 0.1 degree cell centred at (lon, lat) in one
year, cut as the chip-store builder (scripts/build_store_tessera.py)
cuts it: from the grid of the zone that holds the cell's centre, the
pixel envelope of the cell's footprint (`tile_window`). A tile can
straddle up to four shards; `tiles_in_region` lists the tiles that touch
a region of a zone's grid, which the release index records per store so
that georvq.tessera.index.read_tile can assemble a tile from the stores.
`TesseraTile.read` reads one tile straight from the source store.

A product's `source` is the URL of the zone's embeddings array followed
by the slice read from it.
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
SHARD_PX = 4096           # the source arrays' shard side, in pixels


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


# --------------------------------------------------------------------------
# shards: the release's product
# --------------------------------------------------------------------------

def shard_id(year, zone, row, col) -> str:
    """`{year}/utm{zone}/{row}_{col}`, the row zero-padded to 3 digits and
    the column to 2, which cover every zone's shard grid (446 rows of
    4096 px from 84 N to 80 S, 17 columns), so that names sort in grid
    order."""
    return f"{int(year)}/utm{int(zone):02d}/{int(row):03d}_{int(col):02d}"


def shards_of_window(r0, c0, h, w, shard_px=SHARD_PX) -> list:
    """(row, col) of every shard a zone-grid window touches."""
    return [(r, c) for r in range(r0 // shard_px, (r0 + h - 1) // shard_px + 1)
            for c in range(c0 // shard_px, (c0 + w - 1) // shard_px + 1)]


def tiles_in_region(zone, epsg, west, top, px, r0, c0, h, w) -> list:
    """The geotessera tiles of zone `zone` whose window touches the
    region [r0, r0 + h) x [c0, c0 + w) of the zone's grid (origin west,
    top, pixel size px): per tile {name, lon, lat, row, col, height,
    width}, the window in zone pixels. A tile belongs to the zone holding
    its centre."""
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    ys = np.linspace(r0, r0 + h, 9)
    xs = np.linspace(c0, c0 + w, 9)
    gy, gx = np.meshgrid(ys, xs)
    lon, lat = inv.transform(west + gx.ravel() * px, top - gy.ravel() * px)
    lo0 = max(-180 + 6 * (zone - 1), np.floor(min(lon) * 10) / 10 - 0.1)
    lo1 = min(-180 + 6 * zone, np.ceil(max(lon) * 10) / 10 + 0.1)
    la0 = max(-90.0, np.floor(min(lat) * 10) / 10 - 0.1)
    la1 = min(90.0, np.ceil(max(lat) * 10) / 10 + 0.1)
    out = []
    for i in range(int(round(lo0 * 10)), int(round(lo1 * 10))):
        clon = round(i / 10 + 0.05, 2)
        if utm_zone(clon) != zone:
            continue
        for j in range(int(round(la0 * 10)), int(round(la1 * 10))):
            clat = round(j / 10 + 0.05, 2)
            a, b, hh, ww = tile_window(clon, clat, west, top, px, fwd.transform)
            if a < r0 + h and a + hh > r0 and b < c0 + w and b + ww > c0:
                out.append({"name": tile_name(clon, clat), "lon": clon,
                            "lat": clat, "row": a, "col": b, "height": hh,
                            "width": ww})
    return out


class TesseraShard:
    """A region of one zone's grid in one year, normally one 4096 px
    shard of the source arrays, as a georvq.inference.store_codec
    .ProductSource, north-up on the zone's 10 m grid. Rows are read from
    the source (or the arrays given) a strip at a time."""

    orientation = "north-up"

    def __init__(self, year, zone, epsg, west, top, px, r0, c0, h, w,
                 source, strip, shard=None):
        from georvq.inference.store_codec import crs_from_epsg
        self.year, self.zone, self.epsg = int(year), int(zone), int(epsg)
        self.r0, self.c0 = int(r0), int(c0)
        self.height, self.width = int(h), int(w)
        self.crs = crs_from_epsg(self.epsg)
        self.transform = (float(px), 0.0, float(west) + self.c0 * px, 0.0,
                          -float(px), float(top) - self.r0 * px)
        self.zone_origin = (float(west), float(top), float(px))
        self.source = source
        self._strip = strip
        self.shard = None if shard is None else tuple(int(v) for v in shard)
        self.product_id = (shard_id(year, zone, *self.shard) if self.shard
                           else f"{self.year}/utm{self.zone:02d}/"
                                f"r{self.r0}_c{self.c0}")
        self.product_meta = {
            "count": BANDS, "dtype": "int8", "nodata": None,
            "scale": "float32 per pixel, x = q * scale",
            "year": self.year, "utm_zone": self.zone,
            "shard": (None if self.shard is None else
                      {"row": self.shard[0], "col": self.shard[1],
                       "px": int(SHARD_PX)}),
            "zone_window": {"row": self.r0, "col": self.c0,
                            "height": self.height, "width": self.width},
            "zone_origin": {"west": float(west), "top": float(top),
                            "px": float(px)},
            "embeddings": dict(EMBEDDINGS)}
        self.aux = None

    @classmethod
    def read(cls, year, zone, row, col, store=V11_STORE, source_store=None):
        """Shard (row, col) of zone `zone` in `year`, read lazily from the
        v1.1 store (a URL, a local path or an open zarr group).
        `source_store`, when given, is the store the product's `source`
        names instead, for a shard read from a local copy of it."""
        g = open_store(store)
        zg = zone_grid(g, int(zone))
        if int(year) not in zg["years"]:
            raise ValueError(f"utm{int(zone):02d} has no {year}; it has "
                             f"{zg['years']}")
        ti = zg["years"].index(int(year))
        sh = zg["emb"].shards or zg["emb"].chunks
        if tuple(sh[-2:]) != (SHARD_PX, SHARD_PX):
            raise ValueError(f"the source's shards are {sh}, not "
                             f"{SHARD_PX} px")
        H, W = zg["emb"].shape[-2:]
        r0, c0 = int(row) * SHARD_PX, int(col) * SHARD_PX
        if not (0 <= r0 < H and 0 <= c0 < W):
            raise ValueError(f"shard {row}_{col} is outside utm{int(zone):02d}")
        h, w = min(SHARD_PX, H - r0), min(SHARD_PX, W - c0)

        def strip(y0, y1):
            s = np.asarray(zg["scales"][ti, r0 + y0:r0 + y1, c0:c0 + w],
                           np.float32)
            q = np.asarray(zg["emb"][ti, :, r0 + y0:r0 + y1, c0:c0 + w])
            return clean_nodata(q, s)
        out = cls(year, zone, zg["epsg"], zg["west"], zg["top"], zg["px"],
                  r0, c0, h, w,
                  source_url(source_store or store, int(zone), ti, r0, c0, h, w),
                  strip, shard=(row, col))
        out._scales = lambda: np.asarray(
            zg["scales"][ti, r0:r0 + h, c0:c0 + w], np.float32)
        return out

    @classmethod
    def from_arrays(cls, year, zone, q, s, epsg, west, top, px, r0, c0,
                    source, shard=None):
        """A region from arrays already read: q int8 (128, H, W), s (H, W),
        its first pixel at (r0, c0) of the zone's grid."""
        q = np.asarray(q)
        s = np.asarray(s, np.float32)
        if q.dtype != np.int8 or q.ndim != 3 or q.shape[0] != BANDS:
            raise ValueError(f"expected int8 ({BANDS}, H, W) codes, got "
                             f"{q.dtype} {q.shape}")
        if s.shape != q.shape[1:]:
            raise ValueError(f"scale plane {s.shape} does not match the "
                             f"codes {q.shape}")
        q, s = clean_nodata(np.array(q), np.array(s))
        out = cls(year, zone, epsg, west, top, px, r0, c0, *s.shape, source,
                  lambda y0, y1: (q[:, y0:y1], s[y0:y1]), shard=shard)
        out._scales = lambda: s
        out.q, out.s = q, s
        return out

    def valid_fraction(self) -> float:
        """The share of the region's pixels with an embedding, from the
        scales alone (every pixel with a finite positive scale below 1
        has a non-zero vector)."""
        s = self._scales()
        with np.errstate(invalid="ignore"):
            return float((np.isfinite(s) & (s > 0) & (s < 1.0)).mean())

    def read_strip(self, y0, y1):
        return self._strip(y0, y1)
