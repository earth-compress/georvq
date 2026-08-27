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


class GroupBatch(dict):
    """A multi-resolution batch: `{group_name: (B, bands, px, px) tensor}`.

    Sources see a plain dict (it IS one, so `.items()` and `x["10m"]` work
    as before). The shared stack, though, handles batches generically --
    it slices them into chunks, reads `.shape[0]`, moves them to a device.
    Implementing just those four operations here means validation,
    figures and the train loop need no `isinstance` branch to carry a
    multi-group source, and a group dict can never be silently sliced as
    if it were one plane.
    """

    @property
    def device(self):
        return next(iter(self.values())).device

    @property
    def shape(self):
        """Only the batch dimension is shared across groups; the spatial
        dims differ by each group's ratio, so this is deliberately a
        1-tuple rather than a full shape."""
        return (next(iter(self.values())).shape[0],)

    def __getitem__(self, k):
        if isinstance(k, str):
            return dict.__getitem__(self, k)
        return GroupBatch({n: t[k] for n, t in self.items()})

    def to(self, *a, **kw):
        return GroupBatch({n: t.to(*a, **kw) for n, t in self.items()})

    def float(self):
        return GroupBatch({n: t.float() for n, t in self.items()})


class ChipPool:
    def __init__(self, store, rows: pd.DataFrame, pool_size, device,
                 workers=2, refresh_chips_per_s=2.0, seed=0, slot_rows=None,
                 pool_device=None, cycle=False, period=1):
        """slot_rows: pool-slot height in pixels. None = whole chips; e.g.
        128 = full-width row-strips (contiguous per band, zero read
        amplification) -> ~8x more distinct chips resident per GiB.
        period: the store's pixel grid period; `sample_batch(align=True)`
        snaps crop offsets to multiples of it.

        Multi-group (multi-resolution) stores are held as one resident
        tensor PER GROUP, all indexed by the same slot number. Everything
        that makes the pool what it is -- coverage vs cycle residency, the
        shuffled full-dataset cursor, the background refresh threads -- is
        group-agnostic and is written once; only allocation, the shard read
        and the gather divide by each group's ratio. `sample_batch` then
        yields a dict of per-group crops that share one crop origin.
        """
        self.store = store
        self.rows = rows.reset_index(drop=True)
        self.device = device
        self.pool_device = (torch.device(pool_device)
                            if pool_device is not None else device)
        self.chip_px = store.chip_px
        # aux-plane sidecar bytes (e.g. tessera dequant scales) ride along
        # as trailing pseudo-bands; the source's `prep` reinterprets them
        self.aux = getattr(store, "aux_bands", 0)
        self.bands = store.bands + self.aux
        self.cycle = cycle
        self.coverage = (pool_size >= len(self.rows)) and not cycle

        # --- group geometry (single-plane store = one group, ratio 1) ---
        self.groups = getattr(store, "groups", None)
        if self.groups:
            assert not self.aux, \
                "multi-group stores with aux sidecar bands are not supported"
            self.gbands = {g["name"]: g["bands"] for g in self.groups}
            self.ratio = {g["name"]: self.chip_px // g["px"]
                          for g in self.groups}
            # Crop origins are shared across groups, so they must land on
            # EVERY native grid: the period is at least the coarsest ratio
            # (6 for S2's 60m bands). Without this a 20m crop would start
            # half a pixel into its own grid and the groups would decode
            # out of phase with each other.
            period = max(int(period), max(self.ratio.values()))
        else:
            self.gbands = {None: self.bands}
            self.ratio = {None: 1}
        self.period = max(int(period), 1)

        self.slot_rows = slot_rows or self.chip_px
        if self.groups:
            assert self.slot_rows % self.period == 0, \
                (f"slot_rows {self.slot_rows} must be a multiple of the "
                 f"group period {self.period} so every group's strip is a "
                 f"whole number of its own rows")

        dtype = torch.from_numpy(np.zeros(0, dtype=store.dtype)).dtype
        self.gpools = {
            name: torch.empty(
                (pool_size, self.gbands[name], self.slot_rows // ratio,
                 self.chip_px // ratio),
                dtype=dtype, device=self.pool_device)
            for name, ratio in self.ratio.items()}
        # `pool` stays the finest (or only) plane: pool_size, dtype and the
        # nbytes report all read off it, and single-group behaviour is
        # unchanged.
        self.pool = self.gpools[min(self.ratio, key=lambda n: self.ratio[n])]
        self.slot_row = torch.zeros(pool_size, dtype=torch.long,
                                    device=self.pool_device)   # chip-row per slot (side-info)
        self.cuda_lock = threading.Lock()
        self.chips_loaded = 0
        self.stop = threading.Event()
        self._sleep = workers / max(refresh_chips_per_s, 1e-3)
        if cycle:
            self._build_cursor(seed)

        nbytes = sum(t.nbytes for t in self.gpools.values())
        shape = (f"({self.bands},{self.slot_rows},{self.chip_px})"
                 if not self.groups else
                 " + ".join(f"{n}({self.gbands[n]},"
                            f"{self.slot_rows // r},{self.chip_px // r})"
                            for n, r in self.ratio.items()))
        print(f"pool: prefilling {pool_size} {self.pool_device.type}-resident "
              f"slots of {shape} "
              f"({nbytes / 2**30:.1f} GiB) "
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
                data = self._load_tile(ci, r0); self.slot_row[i] = ci
            else:
                ridx = i if self.coverage else None
                data, ri = self._load(np.random.default_rng(seed + 17 * i),
                                      ridx)
                self.slot_row[i] = ri
            if self.pool_device.type == "cuda":
                with self.cuda_lock:
                    self._write_slot(i, data)
            else:
                self._write_slot(i, data)        # distinct slots -> race-free
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

    def _write_slot(self, i, data, pin=False):
        """Copy one slot's payload into the resident tensor(s).

        pin: stage through pinned host memory, which is only worth doing
        for the background refresh's H2D copies (the prefill already runs
        16-wide and would just pay the pinning cost).
        """
        items = data.items() if self.groups else [(None, data)]
        for name, arr in items:
            t = torch.from_numpy(arr)
            if pin and self.pool_device.type == "cuda":
                t = t.pin_memory()
            self.gpools[name][i].copy_(t, non_blocking=False)

    def _read_at(self, r, r0):
        """One slot's payload from the store: a `(bands, slot_rows, px)`
        array, or -- for a multi-group store -- one such array per group,
        each cropped at `r0` scaled into its own grid."""
        if not self.groups:
            sh = self.store.open_shard(int(r["shard"]))
            chip = sh[int(r["slot"])]
            if self.slot_rows < self.chip_px:
                chip = chip[:, r0:r0 + self.slot_rows]
            chip = np.ascontiguousarray(chip)
            if self.aux:
                chip = np.concatenate([chip, self.store.aux_bytes(
                    int(r["shard"]), int(r["slot"]), r0,
                    r0 + self.slot_rows)])
            return chip
        out = {}
        for name, ratio in self.ratio.items():
            sh = self.store.open_shard(int(r["shard"]), name)
            chip = sh[int(r["slot"])]
            if self.slot_rows < self.chip_px:
                g0, gh = r0 // ratio, self.slot_rows // ratio
                chip = chip[:, g0:g0 + gh]
            out[name] = np.ascontiguousarray(chip)
        return out

    def _rand_r0(self, rng):
        """A slot's top edge within its chip.

        Multi-group stores snap it to the grid period, because one origin
        is shared by every group and an unsnapped edge would start a 20m
        or 60m strip part-way into its own pixel. Single-plane stores keep
        the unsnapped draw they have always used -- aligning them too
        would change which pixels every existing preset trains on, which
        is not this change's business to do without saying so.
        """
        if self.slot_rows >= self.chip_px:
            return 0
        span = self.chip_px - self.slot_rows
        if not self.groups:
            return int(rng.integers(0, span + 1))
        p = self.period
        return int(rng.integers(0, span // p + 1)) * p

    def _load(self, rng, row_idx=None):
        """Read one train slot (random chip, or a specific one in coverage
        mode), as a whole chip or a random full-width row-strip of it
        (contiguous per band - no read amplification)."""
        ri = int(rng.integers(0, len(self.rows))) if row_idx is None \
            else int(row_idx % len(self.rows))
        r = self.rows.iloc[ri]
        return self._read_at(r, self._rand_r0(rng)), ri

    def _refresher(self, seed):
        rng = np.random.default_rng(seed)
        while not self.stop.is_set():
            slot = int(rng.integers(0, self.pool.shape[0]))
            # coverage mode: keep every chip resident, just re-roll its strip
            ridx = slot if self.coverage else None
            data, ri = self._load(rng, ridx); self.slot_row[slot] = ri
            with self.cuda_lock:
                self._write_slot(slot, data, pin=True)
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
        """Read one chip's row-slice out of its shard, with the source's aux
        bytes (e.g. Tessera's per-pixel scale plane) appended as trailing
        bands when the store has them.
        """
        return self._read_at(self.rows.iloc[ci], r0)

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
        """Background refresh in whole-dataset FIFO order.

        Unlike random refresh this sweeps the store front to back, so every
        chip is seen once per pass -- the mode to use on multi-TB stores,
        where a random draw would revisit a small resident subset and never
        reach most of the data. Rate is set by `refresh_per_s`, not by pool
        size, so shrinking the pool costs instantaneous diversity but not
        throughput.
        """
        while not self.stop.is_set():
            slot, (ci, r0) = self._next_tile()
            data = self._load_tile(ci, r0); self.slot_row[slot] = ci
            with self.cuda_lock:
                self._write_slot(slot, data, pin=True)
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
        # A multi-group store has no unaligned mode: one origin is shared
        # by every group, so it must always land on the coarsest grid.
        if self.groups:
            align = True
        ci = torch.randint(0, self.pool.shape[0], (batch,), device=pdev)
        if align and p > 1:
            r = torch.randint(0, (self.slot_rows - token) // p, (batch,), device=pdev) * p
            c = torch.randint(0, (self.chip_px - token) // p, (batch,), device=pdev) * p
        else:
            r = torch.randint(0, self.slot_rows - token, (batch,), device=pdev)
            c = torch.randint(0, self.chip_px - token, (batch,), device=pdev)

        def gather(name):
            ratio = self.ratio[name]
            t = token // ratio
            rr, cc = r // ratio, c // ratio
            tt = torch.arange(t, device=pdev)
            bands = torch.arange(self.gbands[name], device=pdev)
            return self.gpools[name][
                ci[:, None, None, None],
                bands[None, :, None, None],
                (rr[:, None] + tt)[:, None, :, None],
                (cc[:, None] + tt)[:, None, None, :]]

        with self.cuda_lock:
            if self.groups:
                out = GroupBatch({n: gather(n) for n in self.ratio})
            else:
                out = gather(None)
        if self.cycle:
            self.total_reads += batch        # lock-free counter (benign races)
        if pdev != dev:
            crop = out.to(dev, non_blocking=True)
        else:
            crop = out
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
    groups = getattr(store, "groups", None)
    if groups:
        ratio = {g["name"]: store.chip_px // g["px"] for g in groups}
        period = max(ratio.values())
    shards = {}
    grid = np.linspace(0, store.chip_px - token, int(np.sqrt(per_chip)),
                       dtype=int)
    if groups:
        # every crop origin has to land on all three native grids
        grid = (grid // period) * period
    toks, sats = [], []
    for ridx, (_, r) in enumerate(rows.head(max_chips).iterrows()):
        sh, slot = int(r["shard"]), int(r["slot"])
        if sh not in shards:
            shards[sh] = ({n: store.open_shard(sh, n) for n in ratio}
                          if groups else store.open_shard(sh))
        if groups:
            planes = {n: np.asarray(shards[sh][n][slot]) for n in ratio}
            # validity is read off the finest grid, the one `token` counts
            fine = min(ratio, key=lambda n: ratio[n])
        else:
            a = np.asarray(shards[sh][slot])
            if getattr(store, "aux_bands", 0):
                a = np.concatenate([a, store.aux_bytes(
                    sh, slot, 0, store.chip_px)])
        for rr in grid:
            for cc in grid:
                if groups:
                    t = {n: planes[n][:, rr // g:rr // g + token // g,
                                      cc // g:cc // g + token // g].copy()
                         for n, g in ratio.items()}
                    ok = nodata is None or (t[fine][0] != nodata).any()
                else:
                    t = a[:, rr:rr + token, cc:cc + token]
                    ok = nodata is None or (t[0] != nodata).any()
                    t = t.copy() if ok else t
                if ok:
                    toks.append(t)
                    if sat_table is not None:
                        sats.append(sat_table[ridx])
        if len(toks) >= max_tokens:
            break
    toks = toks[:max_tokens]
    if groups:
        T = GroupBatch({n: torch.from_numpy(np.stack([t[n] for t in toks]))
                        for n in ratio})
    else:
        T = torch.from_numpy(np.stack(toks))
    if sat_table is not None:
        return T, torch.stack(sats[:max_tokens])
    return T
