"""Build a Tessera chip store by equal-area random sampling of 512x512
crops from the 0.1-degree Tessera embedding tiles (github.com/ucam-eo/
geotessera).

Two datasets are supported, chosen with --dataset. `v1` (the default, used
for /data/tessera-samples/chips_512) downloads whole NPY tiles through
geotessera <= 0.9. `v1.1` reads the dClimate v1.1 run, which geotessera
0.11 serves by default as a Zarr store with one group per UTM zone; see
"v1.1" below. The two are different embedding spaces and must not be mixed
in one store.

Tessera tiles are UTM 10 m rasters of ~1110 rows x ~1110*cos(lat) cols x
128 bands, shipped as per-pixel max-abs-quantized int8 plus a per-pixel
float32 scale (dequant: x = q * scale). A whole tile (~110 MB) is the
download quantum, so each fetched tile yields up to --crops-per-tile
non-overlapping 512^2 crops at a random grid phase. Chips are stored RAW
int8 (the unit direction of each vector is scale-free; normalize at load,
see georvq/sources/tessera.py); the scale planes go to float16
scales_XXXX.npy sidecars with the same shard/slot layout so amplitude
stays available without entering chip-store v1.

Sampling: lon ~ U(-180,180), lat ~ arcsin(U(-1,1)) (uniform on the
sphere), point -> containing 0.1-degree tile, random year among the years
the registry has for that tile. Ocean auto-rejects (no tile). Nodata is
the all-bands-zero vector (matches scale == 0); crops that are fully
nodata are rejected, coastal partials kept.

v1.1: the Zarr store holds int8 (time, 128, y, x) embeddings and float32
(time, y, x) scales on each zone's 10 m UTM grid, the same per-pixel
max-abs quantisation as v1, with NaN scale marking unembedded pixels. The
0.1-degree tiles are kept as the sampling unit so that sampling matches v1:
tile availability per year comes from the dataset's Parquet tile registry
(a tile counts as present when its centre lies inside an embedded 2048 px
registry tile of the zone holding that centre), and a tile's pixel window
is the envelope of its footprint projected into the zone grid. Only the
chosen crops are read. NaN or infinite scales are written as the v1
nodata convention (zero scale, all-zero vector), so the stored int8 codes
are the source codes unchanged and the only loss is the float32 -> float16
cast of the scale (relative error <= 2**-11, as for v1). Needs geotessera
>= 0.11 (zarr 3, obstore, pyproj).

Usage:
  python scripts/build_store_tessera.py --n-chips 16384   # ~512 GiB store
  python scripts/build_store_tessera.py --n-chips 8 --out-dir /tmp/smoke
  python scripts/build_store_tessera.py --dataset v1.1 --n-chips 42263 \
      --out-dir /data/tessera-samples-v11/chips_512 --registry-cache DIR
SIGTERM or SIGUSR1 stops the build after writing the index and manifest;
rerun with --append to continue.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import math
import os
import shutil
import signal
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

from georvq.data.build import ShardWriter

PX = 512


def utm_epsg(lon: float, lat: float) -> str:
    """Standard UTM zone for a tile center (provenance only; Tessera derives
    the exact CRS from its landmask tiles, which may differ in the Norway/
    Svalbard exception zones)."""
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def tile_key(lon: float, lat: float) -> tuple[int, int]:
    """Containing-tile center in integer 1/100 degrees (tiles are 0.1-degree
    squares centered at 0.05 offsets)."""
    return (int(np.floor(lon * 10)) * 10 + 5,
            int(np.floor(lat * 10)) * 10 + 5)


class Sampler:
    """Draw (year, tile_lon, tile_lat) targets uniformly on the sphere."""

    def __init__(self, available, seed, seen=None):
        """available: list of (year, lon, lat) tile-center tuples from
        geotessera's registry."""
        self.years = {}      # tile_key -> [years]
        if isinstance(available, dict):   # already {tile_key: [years]}
            self.years = {k: sorted(v) for k, v in available.items()}
            available = ()
        for year, lon, lat in available:
            key = (int(round(lon * 100)), int(round(lat * 100)))
            self.years.setdefault(key, []).append(year)
        self.rng = np.random.default_rng(seed)
        self.seen = seen or set()

    def draw(self):
        """One accepted (year, lon, lat) tile target or None."""
        lon = float(self.rng.uniform(-180, 180))
        lat = float(np.degrees(np.arcsin(self.rng.uniform(-1, 1))))
        key = tile_key(lon, lat)
        years = self.years.get(key)
        if not years:
            return None                       # ocean / no coverage
        year = int(self.rng.choice(years))
        if (year, *key) in self.seen:
            return None
        self.seen.add((year, *key))
        return year, key[0] / 100.0, key[1] / 100.0


class ScaleSidecar:
    """float16 scales_XXXX.npy planes, same (shard, slot) layout as the
    chip shards; skipped entirely by ChipStore's shard_????.npy glob."""

    def __init__(self, out_dir, shard_size, px):
        self.out_dir, self.shard_size, self.px = out_dir, shard_size, px
        self.sh, self.mm = None, None

    def write(self, chip_id, plane):
        sh, slot = divmod(chip_id, self.shard_size)
        if sh != self.sh:
            path = os.path.join(self.out_dir, f"scales_{sh:04d}.npy")
            if os.path.exists(path):
                self.mm = np.lib.format.open_memmap(path, mode="r+")
            else:
                self.mm = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float16,
                    shape=(self.shard_size, self.px, self.px))
            self.sh = sh
        self.mm[slot] = plane


def fetch_tile(gt_local, target, crops_per_tile, max_nodata, rng):
    """Fetch one tile, return a list of (chip_array, scales, meta) crops."""
    year, lon, lat = target
    gt = gt_local.get()
    try:
        qp = gt.registry.fetch(year=year, lon=lon, lat=lat)
        sp = gt.registry.fetch(year=year, lon=lon, lat=lat, is_scales=True)
        q = np.load(qp)                       # (H, W, 128) int8
        s = np.load(sp)                       # (H, W) float32
        shutil.rmtree(os.path.dirname(qp), ignore_errors=True)
    except Exception as e:  # noqa: BLE001 - transient HTTP errors: skip
        print(f"  ! fetch failed ({type(e).__name__}): {year} "
              f"({lon},{lat}): {str(e)[:120]}", flush=True)
        return []
    h, w = q.shape[:2]
    nr, nc = h // PX, w // PX
    if nr == 0 or nc == 0:
        return []                             # high-latitude sliver tile
    r0 = int(rng.integers(0, h - nr * PX + 1))
    c0 = int(rng.integers(0, w - nc * PX + 1))
    cells = [(r, c) for r in range(nr) for c in range(nc)]
    rng.shuffle(cells)
    out = []
    for r, c in cells[:crops_per_tile]:
        ro, co = r0 + r * PX, c0 + c * PX
        crop = q[ro:ro + PX, co:co + PX]
        nd = float((crop == 0).all(axis=-1).mean())
        if nd >= max_nodata:
            continue
        arr = np.ascontiguousarray(crop.transpose(2, 0, 1))   # (128, px, px)
        # crop-center lat/lon: linear across the tile's 0.1-degree footprint
        out.append((arr, s[ro:ro + PX, co:co + PX], {
            "lat": lat + 0.05 - (ro + PX / 2) / h * 0.1,
            "lon": lon - 0.05 + (co + PX / 2) / w * 0.1,
            "crs": utm_epsg(lon, lat), "year": year,
            "tile_lon": lon, "tile_lat": lat,
            "row_off": ro, "col_off": co, "nodata_frac": nd,
        }))
    return out


# -- v1.1 (dClimate Zarr) ----------------------------------------------------

V11_DEFAULT_STORE = "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate"
V11_REGISTRY = "s3://tessera-embeddings/v1.1/dclimate.registry/parts"
V11_ICECHUNK = "s3://tessera-embeddings/v1.1/dclimate.icechunk"


def utm_zone(lon: float) -> int:
    """UTM zone (1-60) of a longitude; a 0.1-degree tile belongs to the zone
    containing its centre."""
    return max(1, min(60, int(math.floor((lon + 180) / 6)) + 1))


def registry_tile_years(df) -> dict:
    """{tile_key: [years]} from dClimate tile-registry rows.

    Each row is one 2048 px tile of a zone-hemisphere group in one year with
    its WGS84 bounds. A 0.1-degree tile is present in a year when its centre
    lies in an embedded row of the zone and hemisphere holding that centre.
    The bounds are an envelope, so tiles at registry-tile edges can be
    marked present over nodata; those crops are rejected after the read.
    """
    df = df[df["embedded"].astype(bool)]
    zone = df["zone"].str[:2].astype(int).to_numpy()
    north = (df["zone"].str[2] == "N").to_numpy()
    year = df["year"].astype(int).to_numpy()
    w, e = df["bbox_west"].to_numpy(), df["bbox_east"].to_numpy()
    s, n = df["bbox_south"].to_numpy(), df["bbox_north"].to_numpy()
    wrap = w > e                      # crosses the antimeridian: split
    w = np.concatenate([w, np.full(wrap.sum(), -180.0)])
    e = np.concatenate([np.where(wrap, 180.0, e), e[wrap]])
    s, n = np.concatenate([s, s[wrap]]), np.concatenate([n, n[wrap]])
    zone = np.concatenate([zone, zone[wrap]])
    north = np.concatenate([north, north[wrap]])
    year = np.concatenate([year, year[wrap]])
    # tile index i has centre (i + 0.5) / 10 degrees; clip to the zone's
    # columns and the group's hemisphere
    zi0 = -1800 + 60 * (zone - 1)
    i0 = np.maximum(np.ceil(w * 10 - 0.5), zi0).astype(np.int64)
    i1 = np.minimum(np.floor(e * 10 - 0.5), zi0 + 59).astype(np.int64)
    j0 = np.ceil(s * 10 - 0.5).astype(np.int64)
    j1 = np.floor(n * 10 - 0.5).astype(np.int64)
    j0 = np.where(north, np.maximum(j0, 0), j0)
    j1 = np.where(north, j1, np.minimum(j1, -1))
    ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
    cnt = ni * nj
    row = np.repeat(np.arange(len(cnt)), cnt)
    k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ii = i0[row] + k % ni[row]
    jj = j0[row] + k // ni[row]
    code = (ii + 1800) * 4096 + (jj + 900)
    bits = np.left_shift(1, year[row] - 2000).astype(np.int64)
    order = np.argsort(code, kind="stable")
    code, bits = code[order], bits[order]
    uniq, start = np.unique(code, return_index=True)
    mask = np.bitwise_or.reduceat(bits, start) if len(bits) else bits
    out = {}
    for c, m in zip(uniq.tolist(), mask.tolist()):
        i, j = c // 4096 - 1800, c % 4096 - 900
        out[(i * 10 + 5, j * 10 + 5)] = [
            2000 + b for b in range(m.bit_length()) if m >> b & 1]
    return out


def tile_window(lon, lat, west, top, px, to_utm):
    """(r0, c0, h, w) pixel envelope of the 0.1-degree tile centred at
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


class Backoff:
    """Shared delay that grows on failed reads and decays on successes, so
    persistent 429/503 responses lower the request rate of all workers.
    obstore already retries each request with its own backoff; this acts
    only when a whole tile read fails."""

    def __init__(self, max_delay=600.0):
        self.delay, self.max_delay = 0.0, max_delay
        self.lock = threading.Lock()

    def wait(self):
        if self.delay:
            time.sleep(self.delay)

    def fail(self):
        with self.lock:
            self.delay = min(max(2 * self.delay, 5.0), self.max_delay)

    def ok(self):
        with self.lock:
            self.delay = self.delay / 2 if self.delay > 1 else 0.0


class ZarrTiles:
    """Zone groups of a v1.1 Zarr store (geotessera's UTM-zone layout:
    utmNN groups in the northern CRS, southern northings negative)."""

    def __init__(self, url):
        import zarr
        from geotessera.store import zarr_store

        self.url = url
        self.root = zarr.open_group(zarr_store(url), mode="r")
        self.attrs = dict(self.root.attrs)
        self._zones, self._lock = {}, threading.Lock()

    def zone(self, z):
        with self._lock:
            if z not in self._zones:
                g = self.root[f"utm{z:02d}"]
                t = [float(v) for v in g.attrs["spatial:transform"]]
                self._zones[z] = {
                    "emb": g["embeddings"], "scales": g["scales"],
                    "west": t[2], "top": t[5], "px": t[0],
                    "years": [int(y) for y in g["time"][:]],
                    "crs": f"EPSG:{32600 + z}"}
            return self._zones[z]


def fetch_tile_zarr(tiles, target, crops_per_tile, max_nodata, rng,
                    backoff=None):
    """Read up to crops_per_tile crops of one v1.1 tile; returns the same
    (chip_array, scales, meta) tuples as fetch_tile."""
    from pyproj import Transformer

    year, lon, lat = target
    z = utm_zone(lon)
    if backoff is not None:
        backoff.wait()
    try:
        g = tiles.zone(z)
        ti = g["years"].index(year)
        fwd = Transformer.from_crs("EPSG:4326", g["crs"], always_xy=True)
        inv = Transformer.from_crs(g["crs"], "EPSG:4326", always_xy=True)
        r0w, c0w, h, w = tile_window(lon, lat, g["west"], g["top"], g["px"],
                                     fwd.transform)
        nr, nc = h // PX, w // PX
        if nr == 0 or nc == 0:
            return []
        r0 = int(rng.integers(0, h - nr * PX + 1))
        c0 = int(rng.integers(0, w - nc * PX + 1))
        cells = [(r, c) for r in range(nr) for c in range(nc)]
        rng.shuffle(cells)
        out = []
        for r, c in cells[:crops_per_tile]:
            ro, co = r0 + r * PX, c0 + c * PX
            R, C = r0w + ro, c0w + co
            s = np.asarray(g["scales"][ti, R:R + PX, C:C + PX],
                           dtype=np.float32)
            if not np.isfinite(s).any():
                continue                      # skip the embeddings read
            q = np.asarray(g["emb"][ti, :, R:R + PX, C:C + PX])
            q, s = clean_nodata(q, s)
            nd = float((q == 0).all(axis=0).mean())
            if nd >= max_nodata:
                continue
            ce, cn = (g["west"] + (C + PX / 2) * g["px"],
                      g["top"] - (R + PX / 2) * g["px"])
            clon, clat = inv.transform(ce, cn)
            out.append((np.ascontiguousarray(q), s, {
                "lat": float(clat), "lon": float(clon),
                "crs": utm_epsg(lon, lat), "year": year,
                "tile_lon": lon, "tile_lat": lat,
                "row_off": ro, "col_off": co, "nodata_frac": nd,
            }))
    except Exception as e:  # noqa: BLE001 - transient HTTP errors: skip
        if backoff is not None:
            backoff.fail()
        print(f"  ! read failed ({type(e).__name__}): {year} "
              f"({lon},{lat}): {str(e)[:160]}", flush=True)
        return []
    if backoff is not None:
        backoff.ok()
    return out


def load_v11_registry(cache_dir):
    """All dClimate registry rows (latest run per tile), cached as parts."""
    from geotessera.icechunk import TileRegistry

    return TileRegistry(V11_REGISTRY, cache_dir=cache_dir).tiles()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["v1", "v1.1"], default="v1",
                    help="v1: NPY tiles of the vultr run (geotessera <= 0.9); "
                         "v1.1: the dClimate run's Zarr store (geotessera "
                         ">= 0.11)")
    ap.add_argument("--store-url", default=V11_DEFAULT_STORE,
                    help="v1.1 Zarr store; geotessera reads a Source "
                         "Cooperative URL from the bucket's own S3 endpoint")
    ap.add_argument("--registry-cache", default=None,
                    help="v1.1: directory for the tile-registry parts "
                         "(~140 MB; default <out-dir>/../registry_cache)")
    ap.add_argument("--zarr-concurrency", type=int, default=16,
                    help="v1.1: concurrent chunk requests per array read")
    ap.add_argument("--out-dir", default="/data/tessera-samples/chips_512")
    ap.add_argument("--n-chips", type=int, default=16384)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--crops-per-tile", type=int, default=2,
                    help="non-overlapping 512^2 crops kept per fetched tile "
                         "(a tile is the download quantum; ~2 fit)")
    ap.add_argument("--max-nodata", type=float, default=1.0,
                    help="reject crops with nodata fraction >= this; default "
                         "1.0 = reject only fully-empty crops (keep coastal)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--append", action="store_true",
                    help="continue an existing store: resume shard/chip "
                         "numbering, dedupe against its index")
    args = ap.parse_args()

    stop = threading.Event()

    def _on_signal(signum, frame):
        print(f"signal {signum}: stopping after the current reads",
              flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)

    writer = ShardWriter(args.out_dir, args.shard_size, bands=128,
                         chip_px=PX, dtype=np.int8, append=args.append)
    scales = ScaleSidecar(args.out_dir, args.shard_size, PX)
    seen0 = set()
    prior = writer.prior_index
    if len(prior):
        seen0 = set(zip(prior.year,
                        (prior.tile_lon * 100).round().astype(int),
                        (prior.tile_lat * 100).round().astype(int)))

    gt_version = importlib.metadata.version("geotessera")
    scratch = os.path.join(args.out_dir, "_scratch")
    if args.dataset == "v1":
        from geotessera import GeoTessera

        os.makedirs(scratch, exist_ok=True)

        class GTLocal(threading.local):
            def get(self):
                if not hasattr(self, "gt"):
                    self.gt = GeoTessera(embeddings_dir=scratch)
                return self.gt

        gt_local = GTLocal()
        available = gt_local.get().registry.get_available_embeddings()
        n_avail = len(available)

        def fetch(t, rng):
            return fetch_tile(gt_local, t, args.crops_per_tile,
                              args.max_nodata, rng)
        provenance = {"dataset_version": "1.0", "dataset_variant": "vultr"}
    else:
        import zarr

        zarr.config.set({"async.concurrency": args.zarr_concurrency})
        cache = args.registry_cache or os.path.join(
            os.path.dirname(os.path.abspath(args.out_dir)), "registry_cache")
        t0 = time.time()
        reg = load_v11_registry(cache)
        available = registry_tile_years(reg)
        n_avail = sum(len(v) for v in available.values())
        print(f"registry: {len(reg)} rows read in {time.time() - t0:.0f}s",
              flush=True)
        tiles = ZarrTiles(args.store_url)
        backoff = Backoff()

        def fetch(t, rng):
            return fetch_tile_zarr(tiles, t, args.crops_per_tile,
                                   args.max_nodata, rng, backoff)
        provenance = {
            "dataset_version": "1.1", "dataset_variant": "dclimate",
            "source_urls": {
                "zarr": args.store_url,
                "zarr_read_endpoint": (
                    "https://s3.us-west-2.amazonaws.com/us-west-2.opendata."
                    "source.coop/tessera/tessera/zarr/v1.1-dclimate"),
                "icechunk": V11_ICECHUNK, "tile_registry": V11_REGISTRY},
            "source_store_attrs": {
                k: tiles.attrs.get(k) for k in (
                    "geoemb:model", "geoemb:build_version", "checkpoint_id",
                    "optical_min_obs", "geotessera:source_store",
                    "geotessera:source_snapshot")},
            "nodata_conversion": (
                "source NaN/inf scale -> scale 0 and all-zero int8 vector; "
                "int8 codes otherwise copied unchanged; float32 scale cast "
                "to float16"),
        }
    sampler = Sampler(available, args.seed, seen=seen0)
    print(f"registry: {n_avail} (year,tile) entries, "
          f"{len(sampler.years)} distinct tiles; "
          f"target {args.n_chips} new chips -> {args.out_dir}", flush=True)

    target = args.n_chips + writer.n_start
    crop_rng = np.random.default_rng(args.seed + 1)
    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=args.workers)
    pending = set()
    while writer.n_done < target and not stop.is_set():
        while len(pending) < args.workers * 2 and writer.n_done < target:
            t = sampler.draw()
            if t is not None:
                rng = np.random.default_rng(crop_rng.integers(2**63))
                pending.add(ex.submit(fetch, t, rng))
        if not pending:
            continue
        finished, pending = wait(pending, timeout=60,
                                 return_when=FIRST_COMPLETED)
        for f in finished:
            for arr, sc, meta in f.result():
                if writer.n_done >= target:
                    break
                # finite fp16 only: rare corrupt source scales overflow
                # to inf on cast (TesseraSource masks them via SCALE_MAX)
                scales.write(writer.n_done, np.nan_to_num(
                    sc, nan=0.0, posinf=65504, neginf=0.0
                ).astype(np.float16))
                writer.add(arr, meta, target=args.n_chips,
                           checkpoint_shards=1)
    ex.shutdown(wait=True, cancel_futures=True)
    if scales.mm is not None:
        scales.mm.flush()

    complete = writer.n_done >= target
    manifest = writer.finalize(
        "tessera", nodata=None, grid_period=1,
        extra={
            "nodata_convention": "all-bands-zero vector (scale == 0)",
            "dequant": ("x = q * scale (per-pixel float scale in "
                        "scales_XXXX.npy float16 sidecars, same shard/slot "
                        "layout); unit direction is scale-free: "
                        "normalize(q) as in georvq/sources/tessera.py"),
            **provenance,
            "geotessera_version": gt_version,
            "build_date": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
            "complete": complete,
            "licence": "CC-BY-SA 4.0",
            "attribution": ("Tessera embeddings by the University of "
                            "Cambridge EO group (github.com/ucam-eo/"
                            "geotessera)."
                            + (" v1.1 global run produced and hosted by "
                               "dClimate (github.com/dClimate/tessera-"
                               "embeddings)." if args.dataset == "v1.1"
                               else "")),
            "builder_config": vars(args) | {"out_dir": str(args.out_dir)},
        })
    shutil.rmtree(scratch, ignore_errors=True)
    el = time.time() - t0
    print(f"\n{'done' if complete else 'stopped early (rerun with --append)'}"
          f": {manifest['n_chips']} chips, {el/3600:.2f}h this run; "
          f"store -> {args.out_dir} "
          f"({manifest['n_chips'] * 128 * PX * PX / 2**30:.0f} GiB)")


if __name__ == "__main__":
    main()
