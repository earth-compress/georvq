"""Live COPDEM chip store: window reads straight from the GLO-30 COGs.

A drop-in for ChipStore (same attrs, `open_shard(sh)[slot]` contract, and
`split()`), but with no shard files: each chip is read on demand from one
of six latitude-band VRT mosaics over /data/COPDEM/terrain and served as a
(1, 356, 356) float32 array on the native 1-arcsec latitude grid.

GLO-30 thins longitude columns toward the poles (tile width 3600 below
50deg down to 360 above 85deg). Chips are squared by NEAREST-NEIGHBOUR
column replication — output column j reads source column (j*n_src)//356 —
so the mapping back to source pixels is deterministic and exactly
invertible (`invert_cols`), unlike a warp. Chips are centred on Major-TOM
Core-DEM grid cells so cell-level joins against the other Core sources
keep working; pixels are NOT identical to the UTM-projected Major-TOM
chips the npy stores were built from.

The store directory (see scripts/build_live_dem.py) holds only
manifest.json (kind="live_dem"), index.parquet and the band VRTs.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np
import pandas as pd

from georvq.data.store import geographic_split

NODATA = -32767.0
CHIP_PX = 356
# source columns per chip: 356 arcsec of longitude at the band's posting
# (px per degree of longitude; the name-lat -> band table is measured
# from tile headers at build time and stored in the manifest — the
# nominal spec table is hemisphere-asymmetric and cannot be assumed)
N_SRC = {w: max(1, round(CHIP_PX * w / 3600)) for w in
         (3600, 2400, 1800, 1200, 720, 360)}


def col_map(n_src: int) -> np.ndarray:
    """Output column j -> source column (within the chip window)."""
    return (np.arange(CHIP_PX) * n_src) // CHIP_PX


def first_occurrence(n_src: int) -> np.ndarray:
    """Output columns that carry the first copy of each source column —
    chip[:, :, first_occurrence(n)] inverts the replication exactly."""
    m = col_map(n_src)
    return np.flatnonzero(np.r_[True, m[1:] != m[:-1]])


def invert_cols(chip: np.ndarray, n_src: int) -> np.ndarray:
    """(…, 356) replicated chip -> (…, n_src) native source columns."""
    return chip[..., first_occurrence(n_src)]


class _LiveShard:
    """`open_shard()` result: indexable view serving live chips."""

    def __init__(self, store, base):
        self.store = store
        self.base = base

    def __getitem__(self, slot):
        return self.store.read_chip(self.base + int(slot))


class LiveDEMStore:
    """ChipStore-compatible live reader (see module docstring)."""

    SHARD = 4096            # synthetic shard/slot factoring of chip_id

    def __init__(self, store_dir):
        self.dir = str(store_dir)
        with open(os.path.join(self.dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        m = self.manifest
        assert m.get("kind") == "live_dem", f"{self.dir} is not a live store"
        self.chip_px = m["chip_px"]
        self.bands = m["bands"]
        self.dtype = np.dtype(m["dtype"])
        self.nodata = m["nodata"]
        self.grid_period = m.get("grid_period")
        self.index = pd.read_parquet(
            os.path.join(self.dir, "index.parquet"))
        # O(1) hot-path lookup tables (pool threads bypass pandas),
        # indexed by chip_id, so row i of the index must be chip i
        ids = self.index.chip_id.to_numpy()
        if not np.array_equal(ids, np.arange(len(ids))):
            raise ValueError(
                f"{self.dir}/index.parquet: chip_id must equal the row "
                f"position (0..{len(ids) - 1} in order); read_chip indexes "
                f"its lookup tables by chip_id")
        self._band = self.index.band.to_numpy(np.int32)
        self._vrow0 = self.index.vrow0.to_numpy(np.int64)
        self._vcol0 = self.index.vcol0.to_numpy(np.int64)
        self._cmaps = {w: col_map(n) for w, n in N_SRC.items()}
        self._local = threading.local()

    # ---- ChipStore surface ----

    def open_shard(self, sh: int, group=None):
        return _LiveShard(self, int(sh) * self.SHARD)

    def split(self, val_frac=0.04, seed=0, cell_deg=10.0):
        return geographic_split(self.index, val_frac, cell_deg, seed)

    # ---- live reads ----

    def _handle(self, width: int):
        """Per-thread rasterio handle per band VRT (handles are not
        thread-safe; sharing one across pool readers corrupts reads)."""
        h = getattr(self._local, "h", None)
        if h is None:
            h = self._local.h = {}
        if width not in h:
            import rasterio
            h[width] = rasterio.open(
                os.path.join(self.dir, "vrt", f"band_{width}.vrt"))
        return h[width]

    def read_chip(self, chip_id: int) -> np.ndarray:
        """(1, 356, 356) float32 metres; nodata where GLO-30 has no tile."""
        w = int(self._band[chip_id])
        r0, c0 = int(self._vrow0[chip_id]), int(self._vcol0[chip_id])
        n_src = N_SRC[w]
        ds = self._handle(w)
        from rasterio.windows import Window
        win = Window(c0, r0, n_src, CHIP_PX)
        boundless = (r0 < 0 or c0 < 0 or r0 + CHIP_PX > ds.height
                     or c0 + n_src > ds.width)
        a = ds.read(1, window=win, boundless=boundless,
                    fill_value=NODATA) if boundless else \
            ds.read(1, window=win)
        return np.ascontiguousarray(
            a[None, :, self._cmaps[w]], dtype=np.float32)

    def __repr__(self):
        return (f"LiveDEMStore({self.dir}: {len(self.index)} cells, "
                f"live from {self.manifest.get('terrain_dir')})")
