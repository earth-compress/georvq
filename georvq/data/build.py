"""Shared store-writing machinery for source builders.

`ShardWriter` handles everything that is not source-specific: shard memmap
allocation, slot writes, provenance records, periodic ATOMIC index
checkpoints (concurrent readers never see a partial file), append/resume
(always at a fresh shard above all existing — never mid-shard), and the
versioned manifest. A source builder just fetches arrays + metadata and
calls `add()`.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

STORE_VERSION = 1


class ShardWriter:
    def __init__(self, out_dir, shard_size, bands, chip_px, dtype=np.int8,
                 append=False):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.bands = bands
        self.chip_px = chip_px
        self.dtype = np.dtype(dtype)
        self.idx_path = os.path.join(out_dir, "index.parquet")

        self.prior = []
        if append and os.path.exists(self.idx_path):
            old = pd.read_parquet(self.idx_path)
            self.prior = old.to_dict("records")
            print(f"append: resuming after {len(self.prior)} existing chips",
                  flush=True)
        self.records = list(self.prior)
        # append starts a FRESH shard above all existing (never mid-shard;
        # safe when a concurrent consumer processes earlier shards)
        if append and self.prior:
            self.n_done = (max(r["shard"] for r in self.prior) + 1) * shard_size
        else:
            self.n_done = len(self.prior)
        self.n_start = self.n_done
        self._shard = None
        self._t0 = time.time()

    @property
    def prior_index(self) -> pd.DataFrame:
        """Existing records (for builder-side dedupe of source keys)."""
        return pd.DataFrame(self.prior)

    def _write_idx(self, df):
        # atomic: write-temp-then-rename
        tmp = self.idx_path + ".tmp"
        df.to_parquet(tmp)
        os.replace(tmp, self.idx_path)

    def _open_shard(self, sh, fresh):
        path = os.path.join(self.out_dir, f"shard_{sh:04d}.npy")
        if fresh:
            return np.lib.format.open_memmap(
                path, mode="w+", dtype=self.dtype,
                shape=(self.shard_size, self.bands, self.chip_px, self.chip_px))
        return np.lib.format.open_memmap(path, mode="r+")

    def add(self, arr, meta: dict, log_every=50, checkpoint_shards=4,
            target=None):
        sh, slot = divmod(self.n_done, self.shard_size)
        if slot == 0:
            self._shard = self._open_shard(sh, fresh=True)
        elif self._shard is None:  # resuming mid-shard
            path = os.path.join(self.out_dir, f"shard_{sh:04d}.npy")
            if os.path.exists(path):
                self._shard = self._open_shard(sh, fresh=False)
            else:   # resume shard was deleted (a consumer freed the raw):
                self.n_done = (sh + 1) * self.shard_size   # skip to a fresh one
                sh, slot = divmod(self.n_done, self.shard_size)
                self._shard = self._open_shard(sh, fresh=True)
        self._shard[slot] = arr
        self.records.append({"chip_id": self.n_done, "shard": sh,
                             "slot": slot, **meta})
        self.n_done += 1
        n_new = self.n_done - self.n_start
        if log_every and n_new % log_every == 0:
            rate = n_new / (time.time() - self._t0)
            msg = f"  {self.n_done} chips ({rate:.2f}/s"
            if target:
                msg += f", ETA {(target - n_new) / max(rate, 1e-9) / 3600:.1f}h"
            print(msg + ")", flush=True)
        if self.n_done % (self.shard_size * checkpoint_shards) == 0:
            self._write_idx(pd.DataFrame(self.records))   # periodic checkpoint

    def finalize(self, source: str, nodata=None, grid_period=None,
                 extra: dict | None = None) -> dict:
        if self._shard is not None:
            self._shard.flush()
        df = pd.DataFrame(self.records)
        self._write_idx(df)
        manifest = {
            "store_version": STORE_VERSION,
            "source": source,
            "chip_px": self.chip_px,
            "shard_size": self.shard_size,
            "bands": self.bands,
            "dtype": self.dtype.name,
            "nodata": nodata,
            "grid_period": grid_period,
            "n_chips": len(df),
            **(extra or {}),
        }
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        return manifest


def build_synthetic_store(out_dir, n_chips=8, bands=8, chip_px=64,
                          shard_size=4, nodata=-128, grid_period=2, seed=0):
    """Tiny random store for tests and training smokes."""
    rng = np.random.default_rng(seed)
    w = ShardWriter(out_dir, shard_size, bands, chip_px)
    for i in range(n_chips):
        a = rng.integers(-127, 128, size=(bands, chip_px, chip_px),
                         dtype=np.int8)
        if i % 3 == 0:      # some nodata pixels
            a[:, :4, :4] = nodata
        w.add(a, {"lat": float(rng.uniform(-60, 60)),
                  "lon": float(rng.uniform(-180, 180)),
                  "crs": "EPSG:32601", "year": 2024},
              log_every=0)
    return w.finalize("ae", nodata=nodata, grid_period=grid_period)
