"""Live Sentinel-2 L1C store: reads straight from the Major-TOM GeoTIFFs.

A drop-in for `ChipStore` (same attrs, the `open_shard(sh, group)[slot]`
contract, and `split()`) that copies no pixels. Major-TOM Core-S2L1C is
already a chip store: 106k scene directories of 13 single-band GeoTIFFs on
a regular 10 km grid, which is exactly the geometry a chip store would
have been built to reproduce. Rebuilding it as shards would duplicate
1.2 TB to gain nothing.

Geometry, straight from the source rasters: 1068x1068 at 10 m, 534x534 at
20 m, 178x178 at 60 m -- ratios 1/2/6 with 1068 = 178 x 6 exactly, so the
three native grids register with no resampling.

WHOLE SCENES, NOT WINDOWS. The rasters are LZW-compressed with 1024x1024
blocks over a 1068 px image, so any window read decompresses essentially
the whole band anyway: measured 4.7 ms for a full band against 0.4 ms for
a 96x96 window that hits one warm block. Reading the scene and letting the
pool slice rows out of it is both simpler and no more I/O.

CLOUDS ARE KEPT. `cloud_mask.tif` sits beside the bands and is deliberately
not read: cloud is part of the distribution a Sentinel-2 codec has to
carry, and filtering it out would train a model that is only honest on the
clear-sky subset.

The catalogue (scene paths + grid position) is derived from the directory
tree and cached under GEORVQ_CACHE; no store directory is written.
"""
from __future__ import annotations

import glob
import hashlib
import os
import threading

import numpy as np
import pandas as pd

from georvq.data.store import geographic_split

# band -> native group, in the order each group's planes are stacked
BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
BANDS_60M = ["B01", "B09", "B10"]
GROUP_BANDS = {"10m": BANDS_10M, "20m": BANDS_20M, "60m": BANDS_60M}
GROUP_PX = {"10m": 1068, "20m": 534, "60m": 178}
CHIP_PX = 1068
GRID_PERIOD = 6

# Major-TOM Core grid: rows every 10 km of latitude, columns every 10 km
# along each row's parallel. That IS the grid's definition, so deriving a
# cell's position from its name is exact-by-construction and saves opening
# 106k rasters just to read a corner coordinate.
KM_PER_DEG = 111.32
CELL_KM = 10.0


def cell_lat_lon(cell: str, col_cell: str) -> tuple[float, float]:
    """('142U', '142U_1091R') -> approximate (lat, lon) of the grid cell.

    Used only to bucket scenes for the geographic split, which needs
    neighbouring scenes to land in the same coarse cell, not survey-grade
    coordinates.
    """
    row = int(cell[:-1])
    lat = row * CELL_KM / KM_PER_DEG * (1.0 if cell[-1] == "U" else -1.0)
    tail = col_cell.split("_")[-1]
    col = int(tail[:-1])
    scale = max(np.cos(np.deg2rad(lat)), 1e-3)
    lon = col * CELL_KM / (KM_PER_DEG * scale) * \
        (1.0 if tail[-1] == "R" else -1.0)
    return float(lat), float(((lon + 180.0) % 360.0) - 180.0)


def _cache_dir() -> str:
    d = os.environ.get("GEORVQ_CACHE",
                       os.path.expanduser("~/.cache/georvq"))
    os.makedirs(d, exist_ok=True)
    return d


def build_index(root: str, refresh=False) -> pd.DataFrame:
    """Catalogue the scene directories under a Major-TOM S2 root.

    This walks the tree once (~106k directories) and caches the result;
    it reads no pixels and writes nothing into the dataset.
    """
    root = os.path.abspath(str(root))
    key = hashlib.sha1(root.encode()).hexdigest()[:12]
    cache = os.path.join(_cache_dir(), f"s2_scenes_{key}.parquet")
    if os.path.exists(cache) and not refresh:
        return pd.read_parquet(cache)

    print(f"live_s2: cataloguing scenes under {root} "
          f"(once; cached at {cache}) ...", flush=True)
    hits = sorted(glob.glob(os.path.join(root, "*", "*", "*", "B02.tif")))
    if not hits:
        raise FileNotFoundError(
            f"no */*/*/B02.tif under {root} -- is this a Major-TOM "
            f"Core-S2L1C root?")
    rows = []
    for i, h in enumerate(hits):
        scene_dir = os.path.dirname(h)
        parts = scene_dir.split(os.sep)
        cell, col_cell, scene = parts[-3], parts[-2], parts[-1]
        lat, lon = cell_lat_lon(cell, col_cell)
        rows.append({"chip_id": i, "shard": i // 4096, "slot": i % 4096,
                     "path": scene_dir, "scene_id": scene,
                     "cell": cell, "col_cell": col_cell,
                     "lat": lat, "lon": lon, "crs": "", "year": 0})
    idx = pd.DataFrame(rows)
    idx.to_parquet(cache, index=False)
    print(f"live_s2: {len(idx)} scenes catalogued", flush=True)
    return idx


class _LiveGroupShard:
    """`open_shard(sh, group)` result: indexable view serving live reads."""

    def __init__(self, store, base, group):
        self.store, self.base, self.group = store, base, group

    def __getitem__(self, slot):
        return self.store.read_group(self.base + int(slot), self.group)


class LiveS2Store:
    """ChipStore-compatible live reader over Major-TOM Core-S2L1C."""

    SHARD = 4096            # synthetic shard/slot factoring of chip_id

    def __init__(self, root, index=None, refresh=False):
        self.dir = os.path.abspath(str(root))
        self.manifest = {"kind": "live_s2", "root": self.dir,
                         "units": "uint16 reflectance x10000",
                         "clouds": "kept (cloud_mask.tif not applied)"}
        self.groups = [{"name": n, "bands": len(b), "px": GROUP_PX[n]}
                       for n, b in GROUP_BANDS.items()]
        self.chip_px = CHIP_PX
        self.bands = sum(len(b) for b in GROUP_BANDS.values())
        self.dtype = np.dtype(np.uint16)
        self.nodata = 0
        self.grid_period = GRID_PERIOD
        self.aux_bands = 0
        self.index = build_index(self.dir, refresh) if index is None else index
        self._paths = self.index.path.to_numpy()
        self._local = threading.local()

    # ---- ChipStore surface ----

    def open_shard(self, sh: int, group=None):
        assert group is not None, \
            "LiveS2Store is multi-group; open_shard needs a group name"
        return _LiveGroupShard(self, int(sh) * self.SHARD, group)

    def split(self, val_frac=0.04, seed=0, cell_deg=10.0):
        return geographic_split(self.index, val_frac, cell_deg, seed)

    # ---- live reads ----

    def _open(self, path):
        """Per-thread rasterio handles: handles are not thread-safe, and
        the pool runs several reader threads over the same scenes."""
        h = getattr(self._local, "h", None)
        if h is None:
            h = self._local.h = {}
        if path not in h:
            import rasterio
            if len(h) > 64:                  # bound the per-thread cache
                h.pop(next(iter(h))).close()
            h[path] = rasterio.open(path)
        return h[path]

    def read_group(self, chip_id: int, group: str) -> np.ndarray:
        """(bands, px, px) uint16 for one native group of one scene."""
        d = self._paths[int(chip_id)]
        names = GROUP_BANDS[group]
        px = GROUP_PX[group]
        out = np.empty((len(names), px, px), dtype=np.uint16)
        for i, b in enumerate(names):
            ds = self._open(os.path.join(d, f"{b}.tif"))
            a = ds.read(1)
            if a.shape != (px, px):
                # Major-TOM is uniform, but a truncated download is not:
                # fail loudly rather than feeding a mis-registered scene
                raise ValueError(
                    f"{d}/{b}.tif is {a.shape}, expected {(px, px)}")
            out[i] = a
        return out

    def __repr__(self):
        return (f"LiveS2Store({self.dir}: {len(self.index)} scenes, "
                f"live from GeoTIFFs, clouds kept)")
