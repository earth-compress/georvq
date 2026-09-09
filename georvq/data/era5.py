"""Live ERA5 chip store: hourly slices of one variable, read straight
from the cubes.zarr written by scripts/build_store_era5.py.

A drop-in for ChipStore (same attrs, `open_shard(sh)[slot]` contract and
`split()`): shard = cube, slot = hour within the cube's span. Chips are
served DEQUANTIZED as float32 in physical units, pressure levels as
bands (one band for a surface variable). The zarr's int16 packing
carries a per-(cube, level) affine, and the pool strips chip identity
from what it serves, so raw int16 crops could not be dequantized
downstream. Standardization is the source adapter's job
(georvq/sources/era5.py); the store carries no model-space choices.

The store directory holds only manifest.json (kind="era5") and
index.parquet, one row per (cube, hour) at the manifest's hour stride;
scripts/index_store_era5.py writes both. The index snapshots the cubes
committed to the zarr's manifest.parquet at index time, so a store
indexed while the download is still running is a frozen, valid subset.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


class _Era5Shard:
    """`open_shard()` result: one cube's hourly slices by slot."""

    def __init__(self, store, cube):
        self.store = store
        self.cube = cube

    def __getitem__(self, slot):
        return self.store.read_slice(self.cube, int(slot))


class Era5Store:
    """ChipStore-compatible live reader (see module docstring)."""

    def __init__(self, store_dir):
        self.dir = str(store_dir)
        with open(os.path.join(self.dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        m = self.manifest
        assert m.get("kind") == "era5", f"{self.dir} is not an era5 store"
        self.var = m["var"]
        self.chip_px = m["chip_px"]
        self.bands = m["bands"]
        self.dtype = np.dtype("float32")
        self.nodata = None
        self.grid_period = 1
        self.stride_h = m["stride_h"]
        # single-level store: one index row per (cube, hour, level),
        # slot = hour * n_levels + level, each slice one standardized
        # plane. Built by index_store_era5.py --single-level for models
        # that take one level as their band: the pool then holds 64 KB
        # planes instead of full columns, the (chip, level)
        # cross-product lives in the index, and the moments are baked
        # here (recorded in the manifest), so the source's prep is a
        # pass-through.
        self.single_level = bool(m.get("single_level", False))
        self._sl_mu = self._sl_sd = None
        if self.single_level:
            self.bands = 1
            self._sl_mu = np.asarray(m["level_mean"], dtype=np.float64)
            self._sl_sd = np.asarray(m["level_std"], dtype=np.float64)
        import zarr
        root = zarr.open(os.path.join(self.dir, m["zarr"]), mode="r")
        self._arr = root[self.var]
        self._pl = self._arr.ndim == 5
        self._scale = np.asarray(root[self.var + "_scale"],
                                 dtype=np.float64)
        self._offset = np.asarray(root[self.var + "_offset"],
                                  dtype=np.float64)
        self.index = pd.read_parquet(
            os.path.join(self.dir, "index.parquet"))

    # ---- ChipStore surface ----

    def open_shard(self, sh: int, group=None):
        return _Era5Shard(self, int(sh))

    def split(self, val_frac=0.04, seed=0, cell_deg=10.0):
        """Split by DRAW WINDOW rather than geography: the cubes of one
        window share a timestamp, so a spatial split would put the same
        synoptic state on both sides of the fence. `cell_deg` is accepted
        for interface compatibility and ignored."""
        wins = np.sort(self.index.window.unique())
        rng = np.random.default_rng(seed)
        n_val = round(len(wins) * val_frac) if val_frac > 0 else 0
        n_val = max(n_val, 1) if val_frac > 0 else 0
        val = set(rng.permutation(wins)[:n_val].tolist())
        is_val = self.index.window.isin(val)
        return (self.index[~is_val].reset_index(drop=True),
                self.index[is_val].reset_index(drop=True))

    # ---- live reads ----

    def read_slice(self, cube: int, hour: int) -> np.ndarray:
        if self.single_level:
            n = len(self._sl_mu)
            hour, lev = divmod(hour, n)
            q = np.asarray(self._arr[cube, lev, hour])
            x = ((q.astype(np.int64) + 32768) * self._scale[cube, lev]
                 + self._offset[cube, lev])
            x = (x - self._sl_mu[lev]) / self._sl_sd[lev]
            return x.astype(np.float32)[None]
        if self._pl:
            q = np.asarray(self._arr[cube, :, hour])
            s, o = self._scale[cube], self._offset[cube]
        else:
            q = np.asarray(self._arr[cube, hour])[None]
            s, o = self._scale[cube:cube + 1], self._offset[cube:cube + 1]
        x = ((q.astype(np.int64) + 32768) * s[:, None, None]
             + o[:, None, None])
        return x.astype(np.float32)

    def __repr__(self):
        return (f"Era5Store({self.dir}: {len(self.index)} slices of "
                f"{self.var!r}, {self.bands}x{self.chip_px}px)")
