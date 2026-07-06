"""Resident chip pool with background refresh, and deterministic val tokens.

The pool is one (P, bands, slot_rows, chip_px) int tensor on the GPU (or in
system RAM with `pool_device='cpu'` for far larger pools). Reader threads
(rate-limited, one RNG each) stream chips from the memmapped shards into
slots; `sample_batch()` gathers (B, bands, t, t) crops in a single indexing
kernel — no per-crop Python loop, no GIL contention with the readers.

Two residency modes:
- coverage: pool >= train set -> every chip resident exactly once (slot i =
  chip i); refreshes only re-roll each chip's row-strip.
- cycle: the pool is a moving FIFO window over a shuffled sweep of EVERY
  (chip, row-band) tile, so one epoch streams the entire store.
  Reuse-per-patch ~ (sample reads/s) / (refresh rows/s).
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pandas as pd
import torch


class ChipPool:
    def __init__(self, store, rows: pd.DataFrame, pool_size, device,
                 workers=2, refresh_chips_per_s=2.0, seed=0, slot_rows=None,
                 pool_device=None, cycle=False, period=1):
        """slot_rows: pool-slot height in pixels. None = whole chips; e.g.
        128 = full-width row-strips (contiguous per band, zero read
        amplification) -> ~8x more distinct chips resident per GiB.
        period: the store's pixel grid period; `sample_batch(align=True)`
        snaps crop offsets to multiples of it."""
        self.store = store
        self.rows = rows.reset_index(drop=True)
        self.device = device
        self.pool_device = (torch.device(pool_device)
                            if pool_device is not None else device)
        self.chip_px = store.chip_px
        self.bands = store.bands
        self.slot_rows = slot_rows or self.chip_px
        self.cycle = cycle
        self.period = max(int(period), 1)
        self.coverage = (pool_size >= len(self.rows)) and not cycle
        self.pool = torch.empty(
            (pool_size, self.bands, self.slot_rows, self.chip_px),
            dtype=torch.from_numpy(np.zeros(0, dtype=store.dtype)).dtype,
            device=self.pool_device)
        self.slot_row = torch.zeros(pool_size, dtype=torch.long,
                                    device=self.pool_device)   # chip-row per slot (side-info)
        self.cuda_lock = threading.Lock()
        self.chips_loaded = 0
        self.stop = threading.Event()
        self._sleep = workers / max(refresh_chips_per_s, 1e-3)
        if cycle:
            self._build_cursor(seed)

        print(f"pool: prefilling {pool_size} {self.pool_device.type}-resident "
              f"slots of ({self.bands},{self.slot_rows},{self.chip_px}) "
              f"({self.pool.nbytes / 2**30:.1f} GiB) "
              + (f"-> FIFO sweep of {self._n_tiles} tiles "
                 f"({self._n_tiles / pool_size:.1f} pool-fills/epoch) ..."
                 if cycle else f"from {len(self.rows)} train chips ..."),
              flush=True)
        from concurrent.futures import ThreadPoolExecutor

        def _fill(i):
            # stream straight into the slot -- never materialise all strips
            # at once (a large pool would otherwise need 2x its size in RAM)
            if cycle:
                ci, r0 = self._decode_tile(self._cursor[i])
                t = torch.from_numpy(self._load_tile(ci, r0)); self.slot_row[i] = ci
            else:
                ridx = i if self.coverage else None
                arr, ri = self._load(np.random.default_rng(seed + 17 * i), ridx)
                t = torch.from_numpy(arr); self.slot_row[i] = ri
            if self.pool_device.type == "cuda":
                with self.cuda_lock:
                    self.pool[i].copy_(t)
            else:
                self.pool[i].copy_(t)            # distinct slots -> race-free
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(_fill, range(pool_size)))
        self.chips_loaded = pool_size
        if cycle:                    # prefill consumed the first pool_size tiles
            self._cursor_pos = pool_size % self._n_tiles
            self.tiles_streamed = pool_size
        target = self._refresher_fifo if cycle else self._refresher
        self.workers = [
            threading.Thread(target=target, args=(seed + 1000 + w,),
                             daemon=True)
            for w in range(workers)]
        for w in self.workers:
            w.start()

    def _load(self, rng, row_idx=None):
        """Read one train slot (random chip, or a specific one in coverage
        mode), as a whole chip or a random full-width row-strip of it
        (contiguous per band - no read amplification)."""
        ri = int(rng.integers(0, len(self.rows))) if row_idx is None \
            else int(row_idx % len(self.rows))
        r = self.rows.iloc[ri]
        sh = self.store.open_shard(int(r["shard"]))
        chip = sh[int(r["slot"])]
        if self.slot_rows < self.chip_px:
            r0 = int(rng.integers(0, self.chip_px - self.slot_rows + 1))
            chip = chip[:, r0:r0 + self.slot_rows]
        return np.ascontiguousarray(chip), ri

    def _refresher(self, seed):
        rng = np.random.default_rng(seed)
        while not self.stop.is_set():
            slot = int(rng.integers(0, self.pool.shape[0]))
            # coverage mode: keep every chip resident, just re-roll its strip
            ridx = slot if self.coverage else None
            arr, ri = self._load(rng, ridx); self.slot_row[slot] = ri
            chip = torch.from_numpy(arr)
            if self.pool_device.type == "cuda":
                chip = chip.pin_memory()         # pinning only helps H2D
            with self.cuda_lock:
                self.pool[slot].copy_(chip, non_blocking=False)
            self.chips_loaded += 1
            time.sleep(self._sleep)

    # ---- cycle mode: FIFO sweep over a shuffled full-dataset tile cursor ----
    def _build_cursor(self, seed):
        """Shuffled sweep over every (chip, row-band) tile in the train set.
        n_per = ceil(chip_px/slot_rows) non-overlapping bands per chip (the
        last clamped to the bottom edge). One pass over the permutation = one
        epoch = all data seen once; the sweep reshuffles and bumps `epochs`
        on wrap."""
        self._n_per = int(np.ceil(self.chip_px / self.slot_rows))
        self._r0s = np.minimum(np.arange(self._n_per) * self.slot_rows,
                               self.chip_px - self.slot_rows).astype(int)
        self._n_tiles = len(self.rows) * self._n_per
        self._cur_rng = np.random.default_rng(seed + 99)
        self._cursor = self._cur_rng.permutation(self._n_tiles)
        self._cursor_pos = 0
        self._slot_ptr = 0
        self._cur_lock = threading.Lock()
        self.tiles_streamed = 0
        self.epochs = 0
        self.total_reads = 0

    def _decode_tile(self, t):
        """tile index -> (chip row index, r0 band offset)."""
        t = int(t)
        return t // self._n_per, int(self._r0s[t % self._n_per])

    def _load_tile(self, ci, r0):
        r = self.rows.iloc[ci]
        sh = self.store.open_shard(int(r["shard"]))
        return np.ascontiguousarray(
            sh[int(r["slot"])][:, r0:r0 + self.slot_rows])

    def _next_tile(self):
        """(slot, (chip_idx, r0)) for the next FIFO replacement."""
        with self._cur_lock:
            if self._cursor_pos >= self._n_tiles:
                self._cursor = self._cur_rng.permutation(self._n_tiles)
                self._cursor_pos = 0
                self.epochs += 1
            t = self._cursor[self._cursor_pos]
            self._cursor_pos += 1
            slot = self._slot_ptr
            self._slot_ptr = (self._slot_ptr + 1) % self.pool.shape[0]
            self.tiles_streamed += 1
        return slot, self._decode_tile(t)

    def _refresher_fifo(self, seed):
        while not self.stop.is_set():
            slot, (ci, r0) = self._next_tile()
            chip = torch.from_numpy(self._load_tile(ci, r0)); self.slot_row[slot] = ci
            if self.pool_device.type == "cuda":
                chip = chip.pin_memory()
            with self.cuda_lock:
                self.pool[slot].copy_(chip, non_blocking=False)
            self.chips_loaded += 1
            time.sleep(self._sleep)

    @property
    def reuse_estimate(self):
        """Empirical reads-per-patch-before-eviction (steady state ~
        sample reads/s / refresh rows/s). Excludes the prefill tiles."""
        if not self.cycle:
            return float("nan")
        ev = max(self.tiles_streamed - self.pool.shape[0], 1)
        return self.total_reads / ev

    def shutdown(self):
        """Stop reader threads before interpreter exit (daemon threads doing
        CUDA copies during finalization crash the runtime)."""
        self.stop.set()
        for w in self.workers:
            w.join(timeout=5)

    def sample_batch(self, batch, token=16, align=False, return_rows=False):
        """(B, bands, token, token) store-dtype on `self.device` - one gather
        kernel on the pool's device, then a small H2D copy if the pool lives
        on CPU.

        align: snap crop offsets to multiples of the store's grid period.
        Pool slots start at period-aligned offsets, so aligned crops all
        share the same phase relative to any periodic structure in the data
        (e.g. a 2x-upsampled source's aliasing grid) — the model can lock
        onto and reproduce it instead of averaging it away."""
        pdev, dev = self.pool_device, self.device
        p = self.period
        ci = torch.randint(0, self.pool.shape[0], (batch,), device=pdev)
        if align and p > 1:
            r = torch.randint(0, (self.slot_rows - token) // p, (batch,), device=pdev) * p
            c = torch.randint(0, (self.chip_px - token) // p, (batch,), device=pdev) * p
        else:
            r = torch.randint(0, self.slot_rows - token, (batch,), device=pdev)
            c = torch.randint(0, self.chip_px - token, (batch,), device=pdev)
        tt = torch.arange(token, device=pdev)
        bands = torch.arange(self.bands, device=pdev)
        with self.cuda_lock:
            out = self.pool[ci[:, None, None, None],
                            bands[None, :, None, None],
                            (r[:, None] + tt)[:, None, :, None],
                            (c[:, None] + tt)[:, None, None, :]]
        if self.cycle:
            self.total_reads += batch        # lock-free counter (benign races)
        crop = out.to(dev, non_blocking=True) if pdev != dev else out
        if return_rows:
            rows = self.slot_row[ci]
            return crop, (rows.to(dev, non_blocking=True) if pdev != dev else rows)
        return crop


class GroupChipLoader:
    """Batch loader for multi-group (multi-resolution) stores: all groups of
    the working set resident in RAM, consistent random crops across groups
    (offsets snapped to the coarsest grid). Same `sample_batch` contract as
    ChipPool, yielding a dict of per-group crops. Suited to working sets
    that fit memory; background-refresh streaming for multi-group stores is
    a roadmap item."""

    def __init__(self, store, rows: pd.DataFrame, device, period=None,
                 seed=0, max_chips=None):
        assert store.groups, "GroupChipLoader needs a multi-group store"
        self.store = store
        self.device = device
        rows = rows.reset_index(drop=True)
        if max_chips and len(rows) > max_chips:
            rows = rows.head(max_chips)
        self.rows = rows
        self.groups = {g["name"]: (g["bands"], g["px"]) for g in store.groups}
        self.chip_px = max(px for _, px in self.groups.values())
        # ratio of the finest grid to each group's grid (e.g. 1 / 2 / 6)
        self.ratio = {n: self.chip_px // px for n, (_, px) in self.groups.items()}
        self.period = period or max(self.ratio.values())
        self.cycle = False
        self.chips_loaded = len(rows)
        self.data = {n: [] for n in self.groups}
        for _, r in rows.iterrows():
            for n in self.groups:
                arr = store.open_shard(int(r["shard"]), n)[int(r["slot"])]
                self.data[n].append(torch.from_numpy(np.ascontiguousarray(arr)))
        self.data = {n: torch.stack(v) for n, v in self.data.items()}
        self.gen = torch.Generator().manual_seed(seed)

    def sample_batch(self, batch, token=16, align=True, return_rows=False):
        """token: crop size in FINEST-grid pixels (must divide by every
        group ratio). Crops are always aligned to the coarsest grid."""
        p = self.period
        ci = torch.randint(0, len(self.rows), (batch,), generator=self.gen)
        r0 = torch.randint(0, (self.chip_px - token) // p + 1, (batch,),
                           generator=self.gen) * p
        c0 = torch.randint(0, (self.chip_px - token) // p + 1, (batch,),
                           generator=self.gen) * p
        out = {}
        for n, ratio in self.ratio.items():
            t = token // ratio
            rr, cc = r0 // ratio, c0 // ratio
            crops = [self.data[n][ci[b], :, rr[b]:rr[b] + t, cc[b]:cc[b] + t]
                     for b in range(batch)]
            out[n] = torch.stack(crops).to(self.device, non_blocking=True)
        if return_rows:
            return out, ci.to(self.device)
        return out

    def shutdown(self):
        pass


def fixed_val_tokens(store, rows: pd.DataFrame, token=16, per_chip=16,
                     max_chips=256, max_tokens=4096, nodata=None,
                     sat_table=None):
    """Deterministic val tokens: aligned grid crops from held-out chips.
    With sat_table (per-row side-info, e.g. a location embedding), also
    returns the per-token side-info."""
    nodata = store.nodata if nodata is None else nodata
    shards = {}
    grid = np.linspace(0, store.chip_px - token, int(np.sqrt(per_chip)),
                       dtype=int)
    toks, sats = [], []
    for ridx, (_, r) in enumerate(rows.head(max_chips).iterrows()):
        sh = int(r["shard"])
        if sh not in shards:
            shards[sh] = store.open_shard(sh)
        a = np.asarray(shards[sh][int(r["slot"])])
        for rr in grid:
            for cc in grid:
                t = a[:, rr:rr + token, cc:cc + token]
                if nodata is None or (t[0] != nodata).any():
                    toks.append(t.copy())
                    if sat_table is not None:
                        sats.append(sat_table[ridx])
        if len(toks) >= max_tokens:
            break
    T = torch.from_numpy(np.stack(toks[:max_tokens]))
    if sat_table is not None:
        return T, torch.stack(sats[:max_tokens])
    return T
