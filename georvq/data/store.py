"""Chip store: the standard on-disk format every source is converted into.

A store is a directory of int-typed chip shards plus provenance:

    shard_XXXX.npy   memmap (shard_size, bands, chip_px, chip_px)
    index.parquet    one row per chip: chip_id, shard, slot, lat, lon, crs,
                     year + source-specific provenance columns
    manifest.json    store_version, source, geometry, dtype, nodata,
                     grid_period, dequant recipe, attribution, license

Training never touches the original source files — a store builder (see
georvq.data.build and scripts/build_store_*.py) converts each source once.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd


def geographic_split(index: pd.DataFrame, val_frac=0.04, cell_deg=10.0,
                     seed=0):
    """Split chips into train/val by lat/lon grid cells (no spatial leakage).

    Deterministic: cells are hashed with a fixed salt; ~val_frac of cells
    (and all chips inside them) go to val.
    """
    import hashlib
    cells = (np.floor(index.lat / cell_deg).astype(int).astype(str) + "_"
             + np.floor(index.lon / cell_deg).astype(int).astype(str))

    def cell_hash(c):
        h = hashlib.md5(f"{seed}:{c}".encode()).digest()
        return int.from_bytes(h[:4], "little") / 2**32

    is_val = np.array([cell_hash(c) < val_frac for c in cells])
    return index[~is_val].reset_index(drop=True), \
        index[is_val].reset_index(drop=True)


def open_store(store_dir):
    """ChipStore, or a live store when the manifest says so
    (kind="live_dem": chips read on demand from GLO-30 COGs;
    kind="era5": hourly variable slices read from a cube zarr)."""
    mpath = os.path.join(str(store_dir), "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            kind = json.load(f).get("kind")
        if kind == "live_dem":
            from georvq.data.live_dem import LiveDEMStore
            return LiveDEMStore(store_dir)
        if kind == "era5":
            from georvq.data.era5 import Era5Store
            return Era5Store(store_dir)
    return ChipStore(store_dir)


class ChipStore:
    """Read access to a chip store; tolerant of stores that predate the
    versioned manifest (geometry is then inferred from the first shard)."""

    def __init__(self, store_dir):
        self.dir = str(store_dir)
        mpath = os.path.join(self.dir, "manifest.json")
        self.manifest = {}
        if os.path.exists(mpath):
            with open(mpath) as f:
                self.manifest = json.load(f)

        m = self.manifest
        # multi-group stores (e.g. multi-resolution sources) keep one shard
        # file per group: shard_XXXX_<group>.npy
        self.groups = m.get("groups")           # [{name, bands, px}] or None
        suffix = f"_{self.groups[0]['name']}" if self.groups else ""
        self._shards_on_disk = sorted(
            int(os.path.basename(p)[6:10])
            for p in glob.glob(os.path.join(self.dir, f"shard_????{suffix}.npy")))
        if not self._shards_on_disk:
            raise FileNotFoundError(f"no shard files in {self.dir}")

        if m.get("store_version"):
            self.shard_size = m["shard_size"]
            self.chip_px = m["chip_px"]
            self.dtype = np.dtype(m["dtype"])
            self.bands = (m["bands"] if not self.groups
                          else sum(g["bands"] for g in self.groups))
        else:  # infer geometry from the first shard on disk
            probe = self.open_shard(self._shards_on_disk[0])
            self.shard_size, self.bands, self.chip_px, _ = probe.shape
            self.dtype = probe.dtype
        self.nodata = m.get("nodata")
        self.grid_period = m.get("grid_period")

        # Optional per-pixel aux-plane sidecars (scales_XXXX.npy: float16
        # (shard_size, px, px), same shard/slot layout — e.g. tessera dequant
        # scales). Surfaced to the pool as extra trailing byte-bands so the
        # source's `prep` can reassemble the dequantized values on the GPU.
        self.aux_bands = 0
        self._aux_shards = {}
        if glob.glob(os.path.join(self.dir, "scales_????.npy")):
            self.aux_bands = np.dtype(np.float16).itemsize

        idx = pd.read_parquet(os.path.join(self.dir, "index.parquet"))
        have = set(self._shards_on_disk)
        n0 = len(idx)
        idx = idx[idx.shard.isin(have)].reset_index(drop=True)
        if len(idx) < n0:
            print(f"store: {len(idx)}/{n0} chips have raw shards on disk",
                  flush=True)
        self.index = idx

    def shard_path(self, sh: int, group: str | None = None) -> str:
        suffix = f"_{group}" if group else ""
        return os.path.join(self.dir, f"shard_{int(sh):04d}{suffix}.npy")

    def open_shard(self, sh: int, group: str | None = None):
        """Memory-mapped (shard_size, bands, px, px) array (one group of a
        multi-group store, or the whole shard of a single-group store)."""
        return np.load(self.shard_path(sh, group), mmap_mode="r")

    def aux_bytes(self, sh: int, slot: int, r0: int, r1: int):
        """Aux plane of one chip's row range as (aux_bands, rows, px) int8 —
        the float16 plane's bytes as trailing pseudo-bands, so aux data can
        ride through the int8 pool and be reinterpreted in `Source.prep`."""
        if sh not in self._aux_shards:
            self._aux_shards[sh] = np.load(
                os.path.join(self.dir, f"scales_{int(sh):04d}.npy"),
                mmap_mode="r")
        a = np.ascontiguousarray(self._aux_shards[sh][slot][r0:r1])
        rows, px = a.shape
        return a.view(np.int8).reshape(rows, px, self.aux_bands) \
                .transpose(2, 0, 1)

    def split(self, val_frac=0.04, seed=0, cell_deg=10.0):
        return geographic_split(self.index, val_frac, cell_deg, seed)

    def __repr__(self):
        return (f"ChipStore({self.dir}: {len(self.index)} chips, "
                f"{self.bands}x{self.chip_px}px {self.dtype})")
