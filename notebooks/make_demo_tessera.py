"""Builds notebooks/demo_tessera.ipynb, the demo of the Tessera v1.1 release.

    python notebooks/make_demo_tessera.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/demo_tessera.ipynb

The notebook's text is maintained here; edit this file and rebuild
rather than editing the notebook's cells. It follows
notebooks/make_demo_ae.py section by section.
"""
import os
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CELLS = []


def md(s):
    CELLS.append(new_markdown_cell(textwrap.dedent(s).strip()))


def code(s):
    CELLS.append(new_code_cell(textwrap.dedent(s).strip()))


# --------------------------------------------------------------------------

md("""
*Draft for review.*

# The georvq Tessera codec (release tessera-v1.1)

The TESSERA v1.1 annual embeddings (the global run produced and hosted
by dClimate, which geotessera 0.11 reads by default) are published as
one Zarr store with a group per UTM zone, holding 128 int8 bands and a
float32 scale per pixel at 10 m (the embedding is `q * scale`), in
shards of 4096 x 4096 pixels. The tessera-v1.1 release codes each shard
of one year as one Zarr store, about 21 MB for a fully valid shard, that
can be read at 42 rates, from 0.32 to 9.9 bits per pixel, by window,
over HTTP, so the coded stores have the source's layout. The reader
returns float32 embeddings in the source's dequantised units, with zero
vectors at nodata, and `georvq.tessera.v1_1.to_geotessera` turns them
back into int8 codes and per-pixel scales.

This notebook opens a coded store, compares its reconstructions with the
source at several rates, measures what reads cost in bytes, requests
and time, and repeats two common analyses (similarity search and
clustering) on source and reconstruction. Errors are given as the
relative L2 error of the 128-band vector, `||decoded - source|| /
||source||` per pixel, which is the statistic the release measures and
stores. Release details are in `docs/TESSERA_RELEASE.md`, and the store
layout in `docs/GEORVQ_ZARR.md`.
""")

md("""
## Configuration

`DATA_ROOT` holds the coded stores (`{year}/utm{zone}/{row}_{col}.georvq.zarr`)
and their index. It can be a local directory or an HTTP URL, and points
at the published sample on Hugging Face. `STORAGE_OPTIONS` go to fsspec
when the stores are read by URL, for example a token for a private
repository. `SOURCE_STORE` is the Tessera v1.1 Zarr store on Source
Cooperative, used here only for comparison, and only ever read by
window.

The codec weights are not in the stores. On the first pixel read,
`georvq.open` finds the bundle the store's header names: in
`$GEORVQ_TESSERA_BUNDLE`, in the cache at
`~/.cache/georvq/bundles/<bitstream_id>/`, or by downloading the
decoder-only bundle (270 MB) from the Hugging Face release
(`decoder/` in `earth-compress/georvq-tessera-v1.1`). On a machine with
a GPU the torch backend uses it; everything below also runs on CPU, more
slowly.
""")

code("""
from rasterio.windows import Window

# ---- configuration: the only cell that names locations ----
DATA_ROOT = "https://huggingface.co/datasets/earth-compress/georvq-tessera-v1.1-sample/resolve/main"
STORAGE_OPTIONS = None    # e.g. {"headers": {"Authorization": "Bearer <token>"}}
SOURCE_STORE = "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate"
# the bucket behind SOURCE_STORE, which reads are sent to (georvq.tessera.tiles)
SOURCE_BUCKET = ("https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
                 "tessera/tessera/zarr/v1.1-dclimate")
# local copies of the sample work the same way, e.g.
# DATA_ROOT = "path/to/georvq-tessera-v1.1-sample"

STORE = "2025/utm30/080_07"           # west Cumbria: St Bees to Seascale, the western fells
COAST_STORE = "2025/utm30/081_07"     # the shard south of it, partly beyond the run's coverage
# a 2048 px window of STORE: the coast from St Bees Head southwards, Ennerdale Water and the fells
WINDOW = Window(col_off=1024, row_off=1536, width=2048, height=2048)

# Speed section: with a local DATA_ROOT, the store is served through a
# range server that adds the latency and throughput measured to Source
# Cooperative; with a URL, the hosted store is read as it is.
SIMULATE_REMOTE = not DATA_ROOT.startswith(("http://", "https://"))
""")

code("""
import os, sys, time, tempfile, urllib.request, json, io, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Rectangle
import rasterio
from rasterio.warp import transform_bounds
import rasterio.windows

import pandas as pd
import pyarrow.parquet as pq
import zarr
from zarr.abc.store import Store
from zarr.storage import FsspecStore, WrapperStore
import georvq
from georvq.inference.store_codec import quality_value
from georvq.inference.records import Trace
from georvq.tessera.v1_1 import raw_valid, to_geotessera
from georvq.tessera.tiles import clean_nodata
import ae_demo_http

# many small chunk requests to the source store: let zarr make more at once
zarr.config.set({"async.concurrency": 64})

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9, "axes.titlesize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "normal", "legend.frameon": False})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
    "#6250d6", "#e34948")
GREY, INK = "#8a8984", "#52514e"
NODATA_RGB = (0.86, 0.86, 0.84)


REMOTE = DATA_ROOT.startswith(("http://", "https://"))


def data_url(store):
    return f"{DATA_ROOT}/{store}.georvq.zarr"


def open_store(store, **kw):
    return georvq.open(data_url(store), storage_options=STORAGE_OPTIONS if REMOTE else None, **kw)


# the index, read once: store sizes and the bytes of whole-store reads
idx_path = f"{DATA_ROOT}/tessera_georvq_index.parquet"
if REMOTE:
    req = urllib.request.Request(idx_path, headers=(STORAGE_OPTIONS or {}).get("headers", {}))
    index = pq.read_table(io.BytesIO(urllib.request.urlopen(req).read()))
else:
    index = pq.read_table(idx_path)


def store_row(store):
    return next(r for r in index.to_pylist() if r["path"].endswith(f"{store}.georvq.zarr"))


def store_size(store):
    r = store_row(store)
    return r["store_bytes"] + r["metadata_bytes"]


# zarr-python makes the requests, so they are counted by a local server
# between it and DATA_ROOT, which forwards each one and records it
if REMOTE:
    METER, meter, _ = ae_demo_http.serve(upstream=DATA_ROOT,
                                         headers=(STORAGE_OPTIONS or {}).get("headers"))
else:
    METER, meter, _ = ae_demo_http.serve(root=DATA_ROOT)


def metered(store):
    \"\"\"The store's URL through the recording server.\"\"\"
    return f"{METER}/{store}.georvq.zarr"


class CountingStore(WrapperStore):
    \"\"\"A zarr store that records each request: key, bytes and time.\"\"\"
    def __init__(self, store):
        super().__init__(store)
        self.events = []

    def _rec(self, key, t0, buf):
        self.events.append({"kind": "get", "label": key, "t0": t0, "t1": time.perf_counter(),
                            "bytes": 0 if buf is None else len(buf)})

    async def get(self, key, prototype, byte_range=None):
        t0 = time.perf_counter()
        buf = await self._store.get(key, prototype, byte_range)
        self._rec(key, t0, buf)
        return buf

    async def get_partial_values(self, prototype, key_ranges):
        key_ranges, t0 = list(key_ranges), time.perf_counter()
        out = await self._store.get_partial_values(prototype, key_ranges)
        for (k, _), b in zip(key_ranges, out):
            self._rec(k, t0, b)
        return out

    def get_ranges(self, key, byte_ranges, *, prototype, **kw):
        # zarr-python 3.4 reads sharded arrays through get_ranges, which a
        # WrapperStore passes to the wrapped store; run zarr's own range
        # merging over self.get instead, so that each fetch is recorded
        kw = {k: v for k, v in kw.items() if v is not None}
        return Store.get_ranges(self, key, byte_ranges, prototype=prototype, **kw)


def source_part(d, rows, cols):
    \"\"\"Rows [r0, r1) and columns [c0, c1) of a coded store's shard, read from
    the source store: (int8 codes (128, h, w), float32 scales (h, w), with
    nodata as zero vectors and scales, requests, bytes, seconds). Only the
    chunks of the window are fetched.\"\"\"
    p = d.header["product"]
    ti = int(re.search(r"\\[(\\d+),", p["source"]).group(1))
    R0, C0 = p["zone_window"]["row"], p["zone_window"]["col"]
    (r0, r1), (c0, c1) = rows, cols
    for attempt in range(3):            # the bucket drops a connection now and then
        st = CountingStore(FsspecStore.from_url(SOURCE_BUCKET, read_only=True))
        g = zarr.open_group(st, mode="r")[f"utm{p['utm_zone']:02d}"]
        try:
            t0 = time.perf_counter()
            s = np.asarray(g["scales"][ti, R0 + r0:R0 + r1, C0 + c0:C0 + c1], np.float32)
            q = np.asarray(g["embeddings"][ti, :, R0 + r0:R0 + r1, C0 + c0:C0 + c1])
            secs = time.perf_counter() - t0
            break
        except Exception as e:
            print(f"retrying the source read ({type(e).__name__})")
    q, s = clean_nodata(q, s)
    return q, s, st, secs


def source_shard_bytes(d):
    \"\"\"Bytes of the source's shard objects (codes and scales) that a coded
    store mirrors, compressed as stored.\"\"\"
    p = d.header["product"]
    ti = int(re.search(r"\\[(\\d+),", p["source"]).group(1))
    z, sh = p["utm_zone"], p["shard"]
    n = 0
    for key in (f"utm{z:02d}/embeddings/c/{ti}/0/{sh['row']}/{sh['col']}",
                f"utm{z:02d}/scales/c/{ti}/{sh['row']}/{sh['col']}"):
        req = urllib.request.Request(f"{SOURCE_BUCKET}/{key}", method="HEAD",
                                     headers={"User-Agent": "georvq-demo"})
        with urllib.request.urlopen(req) as r:
            n += int(r.headers["Content-Length"])
    return n


def vec(a):
    \"\"\"float32 (128, rows, cols) as read -> (rows, cols, 128); NaN at nodata,
    where the reader returns zero vectors.\"\"\"
    x = np.moveaxis(np.asarray(a, np.float32), 0, -1).copy()
    x[(x == 0).all(-1)] = np.nan
    return x


def src_vec(q, s):
    \"\"\"Source codes and scales -> the dequantised embeddings q * scale as
    (rows, cols, 128); NaN where the pixel is invalid.\"\"\"
    x = np.moveaxis(q, 0, -1).astype(np.float32) * s[..., None]
    x[~raw_valid(q, s)] = np.nan
    return x


def err(x, y):
    \"\"\"Per-pixel relative L2 error ||y - x|| / ||x|| of field y against field x.\"\"\"
    return np.linalg.norm(y - x, axis=-1) / np.maximum(np.linalg.norm(x, axis=-1), 1e-6)


def overview(d, quality, step=8):
    \"\"\"A whole store decoded at `quality`, 1 pixel in `step`, a strip at a time.\"\"\"
    rows = []
    for y0 in range(0, d.height, 1024):
        a = d.read(window=Window(0, y0, d.width, min(1024, d.height - y0)), quality=quality)
        rows.append(a[:, ::step, ::step])
    return np.concatenate(rows, 1)


class PCARGB:
    \"\"\"The first three principal components of a field, as RGB.\"\"\"
    def fit(self, u, n=200_000):
        X = u[np.isfinite(u[..., 0])]
        X = X[np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)]
        self.mean = X.mean(0)
        self.axes = np.linalg.svd(X - self.mean, full_matrices=False)[2][:3]
        P = (X - self.mean) @ self.axes.T
        self.lo, self.hi = np.percentile(P, [1, 99], axis=0)
        return self

    def __call__(self, u):
        P = (np.nan_to_num(u) - self.mean) @ self.axes.T
        rgb = np.clip((P - self.lo) / (self.hi - self.lo), 0, 1)
        rgb[~np.isfinite(u[..., 0])] = NODATA_RGB
        return rgb


def corners(transform, h, w):
    xs, ys = zip(*[transform * (c, r) for c in (0, w) for r in (0, h)])
    return min(xs), min(ys), max(xs), max(ys)


def show(ax, img, transform, ticks=True, **kw):
    \"\"\"imshow north-up in projected km.\"\"\"
    h, w = img.shape[:2]
    x0, y0, x1, y1 = corners(transform, h, w)
    im = ax.imshow(img, extent=[x0 / 1e3, x1 / 1e3, y0 / 1e3, y1 / 1e3], **kw)
    if ticks:
        ax.set_xlabel("easting (km)")
        ax.set_ylabel("northing (km)")
    else:
        ax.set_xticks([]); ax.set_yticks([])
    return im


def box(ax, transform, h, w, colour="white"):
    x0, y0, x1, y1 = corners(transform, h, w)
    ax.add_patch(Rectangle((x0 / 1e3, y0 / 1e3), (x1 - x0) / 1e3, (y1 - y0) / 1e3,
                           fill=False, ec=colour, lw=1.5))


def lonlat(crs, transform, h, w):
    w_, s_, e_, n_ = transform_bounds(crs, "EPSG:4326", *corners(transform, h, w))
    ns = lambda v: f"{abs(v):.2f}°{'N' if v >= 0 else 'S'}"
    ew = lambda v: f"{abs(v):.2f}°{'E' if v >= 0 else 'W'}"
    return f"{ns(s_)} to {ns(n_)}, {ew(w_)} to {ew(e_)}"


def mb(n):
    return f"{n / 1e6:,.2f} MB"
""")

# --------------------------------------------------------------------------

md("""
## Opening a store

Each store codes one 4096 x 4096 px shard of the v1.1 source's zone
arrays in one year, so the stores tile each zone's grid as the source's
shards do. `georvq.open` opens a store with zarr-python, which reads the
group's zarr.json with one request: the georeferencing, the codec
header and the metadata of every array. The dataset it returns has the
attributes of a rasterio dataset (`crs`, `transform`, `width`, `height`,
`count`, `dtypes`, `nodata`, `profile`, `tags()`, `block_windows()`),
on the 10 m grid of the shard's UTM zone, and its `read` takes
rasterio's `window` argument, plus `quality`, the rate to decode at.
""")

code("""
path = data_url(STORE)
ds = open_store(STORE)
meter.clear()
georvq.open(metered(STORE))
open_req, open_bytes = len(meter.events), sum(e["bytes"] for e in meter.events)

print(f"{'store':<12}{path}")
print(f"{'source':<12}{ds.header['product']['source']}")
print(f"{'crs':<12}{ds.crs.to_string()}")
print(f"{'size':<12}{ds.width} x {ds.height} px, {ds.count} bands of {ds.dtypes[0]}, nodata {ds.nodata}")
print(f"{'transform':<12}{tuple(ds.transform)[:6]}")
print(f"{'shard':<12}{ds.header['product']['shard']}")
print(f"{'extent':<12}{lonlat(ds.crs, ds.transform, ds.height, ds.width)}")
print(f"{'codec':<12}bitstream {ds.header['codec']['bitstream_id']}, "
      f"decoder {ds.header['codec']['decoder_id']}, release {ds.header['codec']['release']['name']}")
print(f"{'quality':<12}{ds.header['quality']['name']}: {', '.join(ds.header['quality']['stats'])}")
print(f"{'licence':<12}{ds.header['licence']['spdx']}: {ds.header['licence']['attribution']}")
print(f"{'opened with':<12}{open_req} request, {mb(open_bytes)}")
""")

md("""
The header lists the rates the store supports. A rate is a depth tuple:
how many residual levels of each of the three tiers to decode (coarse,
64 px cells; mid, 16 px; fine, 4 px). The coarse tier starts at 4
levels, the fewest the model decodes, which gives 42 rungs. `bits_px`
counts every coded bit per pixel, the transmitted window mean included,
and `whole-store MB` is what reading the entire store at that rate
fetches, which is less because the stored records are compressed with
zstd. The 20 coarse levels are stored in one record with the mean, so
every coarse rung fetches the same bytes.
""")

code("""
rb = {r["rung"]: r["bytes"] for r in store_row(STORE)["rungs"]}
ladder = pd.DataFrame([{"rung": "-".join(map(str, r["tuple"])),
                        "bits/window": r["bits_window"],
                        "bits/px": round(r["bits_px"], 3),
                        "whole-store MB": round(rb["-".join(map(str, r["tuple"]))] / 1e6, 2)}
                       for r in ds.header["rungs"]])
src_bytes = source_shard_bytes(ds)
print(f"coded store {mb(store_size(STORE))}; source shard {mb(src_bytes)} as stored "
      f"(codes and scales, compressed), {mb(ds.width * ds.height * 132)} uncompressed; "
      f"ratio {src_bytes / store_size(STORE):.0f} to 1")
ladder.set_index("rung").T
""")

# --------------------------------------------------------------------------

md("""
## What the embeddings look like

Each pixel is a 128-dimensional vector. To show a field of them, the
figures project onto its first three principal components and use those
as red, green and blue. The whole store on the left is decoded at
20-0-0 and shown 1 pixel in 8, with its own projection; the window on
the right is fitted on the source window, and the same projection is
applied to its reconstruction, so equal colours mean equal embeddings.
The source is read for the window only.
""")

code("""
q_ref, s_ref, src_store, src_secs = source_part(ds, (WINDOW.row_off, WINDOW.row_off + WINDOW.height),
                                                (WINDOW.col_off, WINDOW.col_off + WINDOW.width))
src_win = (len(src_store.events), sum(e["bytes"] for e in src_store.events))
print(f"source window: {src_win[0]} requests, {mb(src_win[1])}, {src_secs:.1f} s")
x_ref = src_vec(q_ref, s_ref)
valid_ref = np.isfinite(x_ref[..., 0])

t = time.perf_counter()
full = ds.read(window=WINDOW)                             # quality=None: full depth, 20-13-12
print(f"decoded {full.shape} {full.dtype} in {time.perf_counter() - t:.1f} s "
      f"on {getattr(ds.codec, 'device', 'cpu')}")
ov = vec(overview(ds, "20-0-0"))
ov_t = ds.transform * ds.transform.scale(8, 8)

pca = PCARGB().fit(x_ref)
win_t = rasterio.windows.transform(WINDOW, ds.transform)
H, Wd = WINDOW.height, WINDOW.width

fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
show(ax[0], PCARGB().fit(ov)(ov), ov_t)
box(ax[0], win_t, H, Wd)
ax[0].set_title(f"whole store, decoded at 20-0-0\\n{lonlat(ds.crs, ds.transform, ds.height, ds.width)}")
show(ax[1], pca(x_ref), win_t)
ax[1].set_title(f"source window, {Wd} px\\n{lonlat(ds.crs, win_t, H, Wd)}")
show(ax[2], pca(vec(full)), win_t)
ax[2].set_title(f"decoded at 20-13-12 ({ds.header['rungs'][-1]['bits_px']:.2f} bits/px)\\nsame projection")
plt.show()
""")

# --------------------------------------------------------------------------

md("""
## The rate ladder

The cell below decodes the window at every rung and scores each against
the source by the per-pixel relative L2 error. The figure shows a 384 px
crop of six rungs: the coarse tier alone (`4-0-0`, `20-0-0`), the mid
tier added (`20-6-0`, `20-13-0`) and fine levels on top (`20-13-3`,
`20-13-12`), with the error maps below on a log scale.
""")

code("""
LADDER = ["4-0-0", "20-0-0", "20-6-0", "20-13-0", "20-13-3", "20-13-12"]
dec, scores = {}, []
t_all = time.perf_counter()
for r in ds.header["rungs"]:
    q = "-".join(map(str, r["tuple"]))
    a = ds.read(window=WINDOW, quality=q)
    e = err(x_ref, vec(a))[valid_ref]
    scores.append({"rung": q, "bits_px": r["bits_px"], "mean": e.mean(),
                   "p99": np.quantile(e, 0.99)})
    if q in LADDER:
        dec[q] = a
scores = pd.DataFrame(scores).set_index("rung")
print(f"{len(scores)} decodes of the window in {time.perf_counter() - t_all:.1f} s")
scores.loc[LADDER].style.format({"bits_px": "{:.3f}", "mean": "{:.4f}", "p99": "{:.4f}"})
""")

code("""
crop = np.s_[448:832, 1024:1408]        # rows and columns of the window
crop_t = rasterio.windows.transform(Window(WINDOW.col_off + 1024, WINDOW.row_off + 448, 384, 384),
                                    ds.transform)
fig, ax = plt.subplots(2, 7, figsize=(15, 4.9), constrained_layout=True)
show(ax[0, 0], pca(x_ref[crop]), crop_t, ticks=False)
ax[0, 0].set_title("source")
ax[1, 0].axis("off")
norm = LogNorm(0.03, 1)
for k, q in enumerate(LADDER, 1):
    u = vec(dec[q][(slice(None),) + crop])
    show(ax[0, k], pca(u), crop_t, ticks=False)
    ax[0, k].set_title(f"{q}  ({scores.loc[q, 'bits_px']:.2f} bits/px)")
    im = show(ax[1, k], np.maximum(err(x_ref[crop], u), 1e-3), crop_t, ticks=False,
              cmap="magma", norm=norm)
    ax[1, k].set_title(f"mean rel. L2 {scores.loc[q, 'mean']:.3f}")
fig.colorbar(im, ax=ax[1, :], shrink=0.9, label="relative L2 error (per pixel)", pad=0.01)
plt.show()
""")

md("""
The curve puts the 42 rungs on one axis. The blue points are measured
on this window here. The orange points cover the whole store and come
from its quality array, which holds, for every 64 px window and every
rung, the mean and 99th percentile of the relative L2 error measured
when the shard was encoded; they are averaged over the store's valid
pixels and need no decoding. Coarse levels cost 8 bits per window, mid
levels 176 and fine levels 3,072, which gives the three slopes.
""")

code("""
W = ds.window_px
gh, gw = ds.height // W, ds.width // W
n_r = len(ds.rungs)
qh = ds.header["quality"]
qc = ds.quality_codes(0, gh, 0, gw)                       # north-up window grid
mask = ds.dataset_mask() > 0
count = mask.reshape(gh, W, gw, W).sum((1, 3))
store_mean = (quality_value(qc[..., :n_r], qh["scale"], qh["offset"])
              * count[..., None]).sum((0, 1)) / count.sum()

fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
bpp = scores["bits_px"].values
ax.plot(bpp, scores["mean"], "-o", color=BLUE, ms=4, lw=1.5, label="this window, decoded")
ax.plot(bpp, store_mean, "-o", color=ORANGE, ms=4, lw=1.5, label="whole store, from the quality array")
for q in LADDER:
    i = list(scores.index).index(q)
    ax.annotate(q, (bpp[i], scores["mean"].iloc[i]), textcoords="offset points",
                xytext=(4, 5), fontsize=8, color=INK)
ax.set_xscale("log")
ax.set_xlabel("bits per pixel"); ax.set_ylabel("mean relative L2 error")
ax.grid(True, which="major", color="#e4e3df", lw=0.6)
ax.legend()
ax.set_title(f"rate and quality of the {n_r} rungs")
plt.show()
""")

# --------------------------------------------------------------------------

md("""
## What reads cost

Each read below opens the store afresh, so nothing is cached, and counts
the requests zarr-python makes after the open and the bytes they fetch,
as the recording server between it and `DATA_ROOT` sees them. The
source columns are what the same read fetches from the v1.1 source
store, counted by a zarr store that records each request zarr-python
makes to its bucket: its arrays are sharded in 4096 px shards of 32 x 32
px chunks (all 128 bands in one chunk), and a read fetches the shard
indexes and the chunks it touches. For the whole store the source
column is the size of the source's two shard objects as stored, which a
whole-shard read fetches. The three whole-store rows fetch the records
without decoding them.
""")

code("""
def cost(fn, store=STORE):
    with georvq.open(metered(store), codec=ds.codec) as d:
        meter.clear()
        fn(d)
        ev = list(meter.events)
        return len(ev), sum(e["bytes"] for e in ev)


px = Window(WINDOW.col_off + 700, WINDOW.row_off + 700, 1, 1)
_, _, st_px, _ = source_part(ds, (px.row_off, px.row_off + 1), (px.col_off, px.col_off + 1))
src_px = (len(st_px.events), sum(e["bytes"] for e in st_px.events))
src_whole = (2, src_bytes)
reads = [("one pixel at 20-13-12", lambda d: d.read(window=px, quality="20-13-12"), src_px)]
for q in ["20-0-0", "20-13-0", "20-13-3", "20-13-12"]:
    reads.append((f"{Wd} px window at {q}", lambda d, q=q: d.read(window=WINDOW, quality=q), src_win))
for q in [(20, 0, 0), (20, 13, 0), (20, 13, 12)]:
    reads.append((f"whole store at {'-'.join(map(str, q))}",
                  lambda d, q=q: d.reader.read(0, gh, 0, gw, depths=list(q)), src_whole))

costs = pd.DataFrame([dict(zip(("read", "requests", "bytes"), (name,) + cost(fn)),
                           source_requests=s[0], source_bytes=s[1])
                      for name, fn, s in reads]).set_index("read")
costs["source / georvq"] = (costs["source_bytes"] / costs["bytes"]).round(1)
print(f"zarr-python {zarr.__version__}; open: {open_req} request, {mb(open_bytes)} of metadata")
costs
""")

code("""
fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
y = np.arange(len(costs))[::-1]
ax.barh(y + 0.2, costs["bytes"], height=0.38, color=BLUE, label="georvq store")
ax.barh(y - 0.2, costs["source_bytes"], height=0.38, color=GREY, label="v1.1 source store")
for yy, (n, b) in zip(y, zip(costs["requests"], costs["bytes"])):
    ax.text(b * 1.15, yy + 0.2, f"{mb(b)}, {n} req", va="center", fontsize=8, color=INK)
for yy, b in zip(y, costs["source_bytes"]):
    ax.text(b * 1.15, yy - 0.2, mb(b), va="center", fontsize=8, color=INK)
ax.set_yticks(y, costs.index)
ax.set_xscale("log"); ax.set_xlim(1e4, 3e10)
ax.set_xlabel("bytes fetched (log scale)")
ax.legend(loc="lower right")
ax.set_title("bytes fetched per read, cold cache")
plt.show()
""")

# --------------------------------------------------------------------------

md("""
## How long a read takes over HTTP

This section times one cold read of the window, from `georvq.open` to
the float32 array in memory, at four rungs, and the same window read
from the v1.1 source store on Source Cooperative with zarr-python, as
in the section above. The codec is loaded beforehand and reused, so
loading the weights is not timed.

The first cell measures the latency (time to first byte of small range
requests) and the single-stream throughput (one 16 MB range request)
from this machine to the source's bucket. With a local `DATA_ROOT`
(`SIMULATE_REMOTE = True`), the coded store is then served from a local
range server that waits that latency before each response and limits
each response to that throughput. With `DATA_ROOT` a URL, as here, the
hosted store is read through a forwarding server, with no simulation.
zarr-python makes the requests and the server records each one; the
georvq reader's own trace records the decode stages. The source read
is the one made for the figures above, its requests recorded by the
zarr store it was read through.
""")

code("""
p0 = ds.header["product"]
ti0 = int(re.search(r"\\[(\\d+),", p0["source"]).group(1))
probe_url = (f"{SOURCE_BUCKET}/utm{p0['utm_zone']:02d}/embeddings/c/{ti0}/0/"
             f"{p0['shard']['row']}/{p0['shard']['col']}")
net = ae_demo_http.probe(probe_url)
print(f"latency {net['latency_s'] * 1e3:.0f} ms (median of "
      f"{', '.join(f'{v * 1e3:.0f}' for v in net['ttfb_all_s'])} ms), "
      f"throughput {net['throughput_Bps'] / 1e6:.2f} MB/s over one "
      f"{net['large_bytes'] / 1e6:.0f} MB request")
""")

code("""
SPEED_RUNGS = ["20-0-0", "20-13-0", "20-13-3", "20-13-12"]
codec = ds.codec
if SIMULATE_REMOTE:
    base, rec_g, srv_g = ae_demo_http.serve(root=DATA_ROOT, latency=net["latency_s"],
                                            bandwidth=net["throughput_Bps"])
else:
    base, rec_g, srv_g = ae_demo_http.serve(upstream=DATA_ROOT,
                                            headers=(STORAGE_OPTIONS or {}).get("headers"))
speed_url = f"{base}/{STORE}.georvq.zarr"

runs = {}
for q in SPEED_RUNGS:
    tr = Trace()
    rec_g.clear()
    t0 = time.perf_counter()
    with georvq.open(speed_url, codec=codec, trace=tr) as d:
        a = d.read(window=WINDOW, quality=q)
    # the server's requests in place of the reader's per-array reads
    events = [e for e in tr.events if e["kind"] != "get"] + list(rec_g.events)
    runs[f"georvq {q}"] = {"t0": t0, "t1": time.perf_counter(), "events": events,
                           "bytes": sum(e["bytes"] for e in rec_g.events)}
srv_g.shutdown()
ev = src_store.events
t_src0 = min(e["t0"] for e in ev)
runs["source store"] = {"t0": t_src0, "t1": t_src0 + src_secs, "events": ev,
                        "bytes": sum(e["bytes"] for e in ev)}
for k, r in runs.items():
    n = sum(e["kind"] == "get" for e in r["events"])
    print(f"{k:<18} {r['t1'] - r['t0']:7.2f} s  {n:4d} requests  {mb(r['bytes'])}")
""")

code("""
COLOURS = {"get": BLUE, "decode": RED, "cpu": GREY}


def lanes(events):
    \"\"\"Greedy lane assignment so overlapping events do not cover each other.\"\"\"
    ends, out = [], []
    for e in sorted(events, key=lambda e: e["t0"]):
        for i, end in enumerate(ends):
            if e["t0"] >= end:
                ends[i] = e["t1"]; out.append((i, e)); break
        else:
            ends.append(e["t1"]); out.append((len(ends) - 1, e))
    return out, len(ends)


def gantt(runs):
    groups = {k: {kind: lanes([e for e in r["events"] if e["kind"] == kind])
                  for kind in ("get", "decode", "cpu")} for k, r in runs.items()}
    heights = [sum(max(n, 1) for _, n in g.values()) + 1 for g in groups.values()]
    t_max = max(r["t1"] - r["t0"] for r in runs.values())
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 0.32 * sum(heights) + 1.4), sharex=True,
                             gridspec_kw={"height_ratios": heights}, constrained_layout=True)
    for ax, (k, r) in zip(axes, runs.items()):
        row, ticks = 0, []
        for kind in ("get", "decode", "cpu"):
            placed, n = groups[k][kind]
            for lane, e in placed:
                x0, w = e["t0"] - r["t0"], max(e["t1"] - e["t0"], 1e-3)
                ax.barh(row + lane, w, left=x0, height=0.75, color=COLOURS[kind], lw=0)
                if kind == "get" and w > 0.04 * t_max:
                    ax.text(x0 + w / 2, row + lane, mb(e["bytes"]).replace(" MB", ""), ha="center",
                            va="center", fontsize=6.5, color="white")
            ticks.append((row + (max(n, 1) - 1) / 2, {"get": "HTTP GET", "decode": "decode",
                                                        "cpu": "other"}[kind]))
            row += max(n, 1)
        ax.set_yticks([t for t, _ in ticks], [l for _, l in ticks])
        ax.invert_yaxis()
        ax.set_title(f"{k}:  {r['t1'] - r['t0']:.2f} s,  {mb(r['bytes'])},  "
                     f"{sum(e['kind'] == 'get' for e in r['events'])} requests", loc="left")
        ax.axvline(r["t1"] - r["t0"], color=INK, lw=0.8, ls=":")
    axes[-1].set_xlabel("seconds since open")
    axes[0].legend(handles=[Rectangle((0, 0), 1, 1, color=c) for c in COLOURS.values()],
                   labels=["HTTP GET (MB on the bar)", "decode stage", "other local work"],
                   loc="upper right", ncol=3, fontsize=8)
    plt.show()


gantt(runs)

stages = pd.DataFrame([{**{"read": k}, **{lab: sum(e["t1"] - e["t0"] for e in r["events"]
                                               if e["label"] == lab)
                                           for lab in ("mean channel", "codebook lookup",
                                                       "decoder forward", "postprocess")}}
                       for k, r in runs.items() if k.startswith("georvq")]).set_index("read")
(stages * 1e3).round(1).rename(columns=lambda c: c + " (ms)")
""")

md("""
The next cell reads the whole store (4096 px) through the same server
at the same rungs, and splits each read into the time spent fetching
and the time spent decoding.
""")

code("""
def spans(events, kind):
    \"\"\"Wall-clock time covered by events of one kind (overlaps counted once).\"\"\"
    t, end = 0.0, -1e18
    for e in sorted((e for e in events if e["kind"] == kind), key=lambda e: e["t0"]):
        t0 = max(e["t0"], end)
        if e["t1"] > t0:
            t += e["t1"] - t0
        end = max(end, e["t1"])
    return t


if SIMULATE_REMOTE:
    base, rec_g, srv_g = ae_demo_http.serve(root=DATA_ROOT, latency=net["latency_s"],
                                            bandwidth=net["throughput_Bps"])
else:
    base, rec_g, srv_g = ae_demo_http.serve(upstream=DATA_ROOT,
                                            headers=(STORAGE_OPTIONS or {}).get("headers"))
speed_url = f"{base}/{STORE}.georvq.zarr"
wide = {}
for q in SPEED_RUNGS:
    tr = Trace()
    rec_g.clear()
    t0 = time.perf_counter()
    with georvq.open(speed_url, codec=codec, trace=tr) as d:
        for y0 in range(0, d.height, 1024):        # 1024-row strips, to keep memory down
            d.read(window=Window(0, y0, d.width, 1024), quality=q)
    events = [e for e in tr.events if e["kind"] != "get"] + list(rec_g.events)
    wide[f"georvq {q}, whole store"] = {"t0": t0, "t1": time.perf_counter(), "events": events,
                                        "bytes": sum(e["bytes"] for e in rec_g.events)}
srv_g.shutdown()
gantt(wide)
pd.DataFrame([{"read": k, "MB": round(r["bytes"] / 1e6, 2),
               "requests": sum(e["kind"] == "get" for e in r["events"]),
               "fetch (s)": round(spans(r["events"], "get"), 2),
               "decode (s)": round(spans(r["events"], "decode"), 2),
               "total (s)": round(r["t1"] - r["t0"], 2)}
              for k, r in {**{k: v for k, v in runs.items() if k.startswith("georvq")},
                           **wide}.items()]).set_index("read")
""")

# --------------------------------------------------------------------------

md("""
## Reading to a quality bound

Instead of a rung, `read` accepts a bound on a quality statistic, such
as `quality="p99<=0.3"`. The reader looks up each window's measured
errors in the quality array and decodes every window at the cheapest
rung that meets the bound, or at full depth where none does. The map
shows the rung chosen for each window of the store. Within one read the
reader fetches each tier at the deepest level any window in the read
needs, so the bytes fetched sit between the ideal per-window cost and
the full-depth cost. The quality values are rounded up to the next step
of 7.5%, so a window read at the cheapest rung that meets the bound does
not measure above it when decoded as the values were measured.
""")

code("""
BOUND, LIMIT = "p99<=0.3", 0.3
rmap = ds.window_rungs(BOUND)                               # whole store, north-up window grid
bits = np.array([r["bits_window"] for r in ds.header["rungs"]])
names = ["-".join(map(str, r)) for r in ds.rungs]
used = np.unique(rmap)

d_q = cost(lambda d: d.read(window=WINDOW, quality=BOUND))
a_q = ds.read(window=WINDOW, quality=BOUND)
e_q = np.where(valid_ref, err(x_ref, vec(a_q)), np.nan)
nh, nw = H // W, Wd // W
e_w = e_q.reshape(nh, W, nw, W).transpose(0, 2, 1, 3).reshape(nh, nw, -1)
p99_w = np.nanquantile(e_w, 0.99, axis=-1)
wy, wx = WINDOW.row_off // W, WINDOW.col_off // W
p99_chosen = np.take_along_axis(quality_value(qc[..., n_r:2 * n_r], qh["scale"], qh["offset"]),
                                rmap[..., None], -1)[..., 0][wy:wy + nh, wx:wx + nw]
met = p99_chosen <= LIMIT

print(f"whole store at {BOUND}: {bits[rmap].sum() / 8 / 1e6:.2f} MB of records chosen per window, "
      f"against {rmap.size * bits[-1] / 8 / 1e6:.2f} MB at 20-13-12")
print(f"this window: {mb(d_q[1])} in {d_q[0]} requests at {BOUND}, "
      f"against {mb(costs.loc[f'{Wd} px window at 20-13-12', 'bytes'])} at 20-13-12")
print(f"{met.sum()} of the window's {met.size} windows have a rung meeting the bound: "
      f"{(p99_w[met] > LIMIT * (1 + 1e-6)).sum()} over it when decoded here "
      f"(worst p99 {p99_w[met].max():.3f}); the other {(~met).sum()} are read at full depth")

cmap = plt.get_cmap("viridis", len(used))
lut = np.zeros(len(ds.rungs), int); lut[used] = np.arange(len(used))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True,
                       gridspec_kw={"width_ratios": [1.15, 1]})
x0, y0, x1, y1 = corners(ds.transform, ds.height, ds.width)
im = ax[0].imshow(lut[rmap], cmap=cmap, vmin=-0.5, vmax=len(used) - 0.5,
                  extent=[x0 / 1e3, x1 / 1e3, y0 / 1e3, y1 / 1e3], interpolation="nearest")
cb = fig.colorbar(im, ax=ax[0], ticks=np.arange(len(used)), shrink=0.9)
cb.ax.set_yticklabels([names[i] for i in used]); cb.set_label("rung chosen")
box(ax[0], win_t, H, Wd)
ax[0].set_xlabel("easting (km)"); ax[0].set_ylabel("northing (km)")
ax[0].set_title(f"rung per 64 px window for {BOUND}")
ax[1].hist(p99_w.ravel(), bins=40, color=BLUE)
ax[1].axvline(LIMIT, color=INK, ls="--", lw=1)
ax[1].text(LIMIT * 1.01, ax[1].get_ylim()[1] * 0.9, "bound", color=INK, fontsize=8)
ax[1].set_xlabel("p99 of relative L2 error per window, decoded here")
ax[1].set_ylabel("windows")
ax[1].set_title(f"the {p99_w.size} windows of the window read")
plt.show()
""")

# --------------------------------------------------------------------------

md("""
## Similarity search and clustering

Two common uses of the embeddings, run on the source window and on
reconstructions at four rungs. The search takes the embedding of one
pixel (marked) and maps its Euclidean distance to every pixel of the
window; each product is queried with its own pixel, as a user holding
only that product would. The clustering fits k-means with 8 clusters
on the source and assigns the pixels of every product to the same
centroids. The table reports the correlation of each distance map
with the source's, the share of the source's 1% nearest pixels that
are also among the reconstruction's 1% nearest, and the share of
pixels assigned to the same cluster.
""")

code("""
from sklearn.cluster import KMeans

QUERY = (1000, 1200)                     # row, column in the window
SHOW = ["20-0-0", "20-13-0", "20-13-3", "20-13-12"]
prods = {"source": x_ref} | {q: vec(dec[q]) for q in SHOW}

X = x_ref[valid_ref]
km = KMeans(8, n_init=4, random_state=0).fit(
    X[np.random.default_rng(0).choice(len(X), 50_000, replace=False)])
dists, labels, rows = {}, {}, []
for k, u in prods.items():
    s = np.linalg.norm(np.nan_to_num(u) - u[QUERY], axis=-1).astype(np.float32)
    s[~valid_ref] = np.nan
    lab = np.full(valid_ref.shape, -1)
    lab[valid_ref] = km.predict(np.nan_to_num(u[valid_ref]))
    dists[k], labels[k] = s, lab
near = lambda s: s <= np.nanquantile(s, 0.01)
for k in SHOW:
    rows.append({"rung": k,
                 "distance r": np.corrcoef(dists["source"][valid_ref], dists[k][valid_ref])[0, 1],
                 "nearest 1% recall": (near(dists["source"]) & near(dists[k])).sum() / near(dists["source"]).sum(),
                 "same cluster": (labels[k] == labels["source"])[valid_ref].mean()})
pd.DataFrame(rows).set_index("rung").style.format("{:.3f}")
""")

code("""
fig, ax = plt.subplots(2, 5, figsize=(15, 6.3), constrained_layout=True)
qx, qy = win_t * (QUERY[1] + 0.5, QUERY[0] + 0.5)
ccmap = ListedColormap([BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED])
vmax = np.nanquantile(dists["source"], 0.99)
for j, (k, s) in enumerate(dists.items()):
    im = show(ax[0, j], s, win_t, ticks=False, cmap="viridis_r", vmin=0, vmax=vmax)
    ax[0, j].plot(qx / 1e3, qy / 1e3, "+", color="white", ms=10, mew=1.5)
    ax[0, j].set_title(k if k == "source" else f"{k}  ({scores.loc[k, 'bits_px']:.2f} bits/px)")
    show(ax[1, j], np.ma.masked_less(labels[k], 0), win_t, ticks=False, cmap=ccmap,
         vmin=-0.5, vmax=7.5, interpolation="nearest")
fig.colorbar(im, ax=ax[0, :], shrink=0.85, pad=0.01, label="Euclidean distance to the query")
ax[0, 0].set_ylabel("similarity search"); ax[1, 0].set_ylabel("k-means, 8 clusters")
plt.show()
""")

# --------------------------------------------------------------------------

md("""
## Nodata and partial stores

A Tessera pixel is nodata where its vector is all zero or its scale is
not between 0 and 1 (the v1.1 store marks it with a NaN scale). The
store keeps the shard's validity mask once, losslessly, and the reader
writes zero vectors back wherever it is unset, so nodata comes back
exactly. A window with no valid pixel is not coded, a chunk with none is
not stored, and a shard with none has no store, so partial shards are
smaller and their nodata costs almost nothing to read. The store shown
here is the shard south of the one above, of which a quarter, offshore,
has no embedding in the v1.1 run; the 1024 px window shown is the one,
on a 512 px step, whose valid share is closest to one half.
""")

code("""
cds = open_store(COAST_STORE, codec=ds.codec)
cmask = cds.dataset_mask() > 0
# valid share of 1024 px windows on a 512 px step (the nodata edge here
# follows the run's 2048 px tiles, so whole 1024 px blocks are all or nothing)
frac = cmask.reshape(4, 1024, 4, 1024).mean((1, 3))
steps = [(i, j) for i in range(0, 3073, 512) for j in range(0, 3073, 512)]
share = [cmask[i:i + 1024, j:j + 1024].mean() for i, j in steps]
ci, cj = steps[int(np.argmin(np.abs(np.array(share) - 0.5)))]
CWIN = Window(cj, ci, 1024, 1024)
cq, cs, _, _ = source_part(cds, (ci, ci + 1024), (cj, cj + 1024))
c_dec = cds.read(window=CWIN, quality="20-13-3")
nod_ref, nod_dec = ~raw_valid(cq, cs), (c_dec == 0).all(0)
empty = {g["kind"]: len(g.get("empty_chunks", [])) for g in cds.header["layout"]["groups"]}
zero = np.argwhere(frac == 0)
if len(zero):
    ei, ej = zero[0]
    empty_read = cost(lambda d: d.read(window=Window(ej * 1024, ei * 1024, 1024, 1024),
                                       quality="20-13-12"), store=COAST_STORE)
print(f"valid pixels {cmask.mean():.1%}; store {mb(store_size(COAST_STORE))} "
      f"(the fully valid store above: {mb(store_size(STORE))}); "
      f"source shard {mb(source_shard_bytes(cds))} as stored")
print(f"empty chunks per kind: {empty}")
print(f"window at row {ci}, column {cj}, {cmask[ci:ci + 1024, cj:cj + 1024].mean():.1%} valid: "
      f"nodata pixels {nod_ref.sum():,} in the source, "
      f"{nod_dec.sum():,} decoded, {(nod_ref != nod_dec).sum()} disagreeing")
if len(zero):
    print(f"reading an all-nodata block at 20-13-12: {empty_read[0]} requests, {mb(empty_read[1])}")

c_ov = vec(overview(cds, "20-0-0"))
cx = src_vec(cq, cs)
cp = PCARGB().fit(c_ov)                                      # colours from the whole store
cwin_t = rasterio.windows.transform(CWIN, cds.transform)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
show(ax[0], cp(c_ov), cds.transform * cds.transform.scale(8, 8), interpolation="nearest")
box(ax[0], cwin_t, 1024, 1024, ORANGE)
ax[0].set_title(f"whole store at 20-0-0, 1 px in 8 (grey = nodata)\\n"
                f"{lonlat(cds.crs, cds.transform, cds.height, cds.width)}")
show(ax[1], cp(cx), cwin_t); ax[1].set_title("source window (grey = nodata)")
show(ax[2], cp(vec(c_dec)), cwin_t); ax[2].set_title("decoded at 20-13-3")
plt.show()
""")

# --------------------------------------------------------------------------

md("""
## The index

`tessera_georvq_index.parquet` has one row per coded store. Its first
columns are the same in every georvq release's index: the store's
path, the product id, the source URL, the release, the CRS, the size
and bounds in the shard's UTM zone, the footprint in longitude and
latitude (`geometry` and its `bbox`), the codec ids, the store's sizes,
and the bytes a whole-store read fetches at each rung. Tessera's own
columns follow, from each store's header: the year, the UTM zone, the
shard's row and column and its first pixel in the zone's grid, the
grid's origin, the share of valid pixels, the embeddings version, and
`tiles`, the geotessera tiles that touch the store with their windows.
It is GeoParquet, so pyarrow, GeoPandas or DuckDB can query it without
opening the coded stores.
""")

code("""
import pyarrow.compute as pc

idx = index                                  # read with the helpers above, from idx_path
print(f"{idx.num_rows} stores; columns: {', '.join(idx.column_names)}")

def at(rungs, name):
    return next(r["bytes"] for r in rungs if r["rung"] == name)

bbox = idx["bbox"].to_pylist()
stores = pd.DataFrame({
    "store": [p.removesuffix(".georvq.zarr") for p in idx["path"].to_pylist()],
    "lon": [(b["xmin"] + b["xmax"]) / 2 for b in bbox],
    "lat": [(b["ymin"] + b["ymax"]) / 2 for b in bbox],
    "valid": idx["valid_fraction"].to_pylist(),
    "tiles": [len(x) for x in idx["tiles"].to_pylist()],
    "store MB": [(b + m) / 1e6 for b, m in zip(idx["store_bytes"].to_pylist(),
                                               idx["metadata_bytes"].to_pylist())],
    "20-13-0 MB": [at(r, "20-13-0") / 1e6 for r in idx["rungs"].to_pylist()]})

# stores whose footprint box intersects a box over Europe
eu = dict(west=-10, south=35, east=30, north=70)
bx = {k: pc.struct_field(idx["bbox"], k) for k in ("xmin", "ymin", "xmax", "ymax")}
sel = idx.filter(pc.and_(pc.and_(pc.greater(bx["xmax"], eu["west"]), pc.less(bx["xmin"], eu["east"])),
                         pc.and_(pc.greater(bx["ymax"], eu["south"]), pc.less(bx["ymin"], eu["north"]))))
print(f"in the Europe box: {sel['path'].to_pylist()}")
try:
    import duckdb
    print(duckdb.sql(f"select path, valid_fraction, store_bytes from read_parquet('{idx_path}') "
                     "order by store_bytes").df())
except ImportError:
    print("duckdb is not installed; the same query works with read_parquet() in DuckDB")

fig, ax = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
try:
    import geopandas as gpd
    land = gpd.read_file("https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip")
    land.plot(ax=ax, color="#e9e7e1", edgecolor="#c9c6bd", lw=0.4, zorder=0)
except Exception as e:                       # no geopandas or no network: points only
    print(f"no basemap ({type(e).__name__})")
ax.scatter(stores["lon"], stores["lat"], s=stores["store MB"] * 3, color=BLUE, alpha=0.85)
ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
ax.grid(True, color="#e4e3df", lw=0.6, zorder=-1)
ax.set_aspect("equal")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("stores in the index (marker area by store size; land from Natural Earth)")
plt.show()
stores.round(2)
""")

md("""
Many tools address the embeddings by geotessera tile, the 0.1 degree
cell centred at a given longitude and latitude. A tile lies in one to
four stores, and the index lists, for every store, the tiles that touch
it with their windows, so `georvq.tessera.find_tile` finds a tile's
stores from the index alone and `georvq.tessera.read_tile` reads the
tile from them as one array, with the tile's shape, georeferencing and
nodata.
""")

code("""
from georvq.tessera import find_tile, read_tile

info = find_tile(idx_path, "2025/grid_-3.45_54.35")       # the coast at Ravenglass
tile = read_tile(idx_path, "2025/grid_-3.45_54.35", quality="20-13-0", codec=ds.codec)
print(f"window {info['window']} (row, col, height, width) of utm{info['zone']}, in "
      f"{', '.join(s['path'].rsplit('/', 3)[-1] for s in info['stores'])}; read as {tile.shape} {tile.dtype}")
""")

# --------------------------------------------------------------------------

md("""
## Writing a GeoTIFF, and decoding without torch

`ds.profile` is a GeoTIFF profile for the decoded output (CRS,
transform, float32 bands, nodata 0, block size, compression), so a
decoded window can be written as a GeoTIFF as below. For a whole store,
`georvq.inference.store_dataset.reconstruct(ds, "out.tif",
quality="20-13-3")` does the same in strips. For code that reads the
embeddings' int8 form, `georvq.tessera.v1_1.to_geotessera` turns a
decoded array back into int8 codes of shape (rows, cols, 128) and a
float32 scale per pixel, set so that every valid pixel has a code of
127 or -127, as in the source store.

`backend="onnx"` decodes with onnxruntime and numpy from the ONNX
export in the full bundle, without torch, and downloads the full bundle
(905 MB) for it. It decodes in fp32. On CUDA the torch backend decodes
in bf16 by default and on CPU in fp32, so the comparison below is made
against the decode used above and, on CUDA, against an fp32 torch
decode (`bf16=False`) as well.
""")

code("""
out = os.path.join(tempfile.mkdtemp(), "window_20-13-3.tif")
prof = dict(ds.profile, width=Wd, height=H, transform=win_t, blockxsize=512, blockysize=512)
with rasterio.open(out, "w", **prof) as dst:
    dst.write(dec["20-13-3"])
    dst.update_tags(**ds.tags())
with rasterio.open(out) as chk:
    print(f"{os.path.basename(out)}: {chk.width} x {chk.height}, {chk.count} bands of {chk.dtypes[0]}, "
          f"{chk.crs.to_string()}, {chk.profile.get('compress')}, {mb(os.path.getsize(out))}; "
          f"identical to the array: {np.array_equal(chk.read(), dec['20-13-3'])}")

q8, s8 = to_geotessera(dec["20-13-3"])
x8 = q8.astype(np.float32) * s8[..., None]
x8[~valid_ref] = np.nan
print(f"to_geotessera: int8 {q8.shape} codes and float32 {s8.shape} scales, {mb(q8.nbytes + s8.nbytes)}; "
      f"mean relative L2 {err(vec(dec['20-13-3']), x8)[valid_ref].mean():.4f} against the decoded "
      f"floats, {err(x_ref, x8)[valid_ref].mean():.4f} against the source "
      f"(the decoded floats: {scores.loc['20-13-3', 'mean']:.4f})")

t = time.perf_counter()
with open_store(STORE, backend="onnx") as dx:
    a_onnx = dx.read(window=WINDOW, quality="20-13-3")
t_onnx = time.perf_counter() - t
compare = [(f"torch {'bf16' if ds.codec.bf16 else 'fp32'} on {ds.codec.device.type}", dec["20-13-3"])]
if ds.codec.bf16:
    with open_store(STORE, bf16=False) as d32:
        compare.append(("torch fp32", d32.read(window=WINDOW, quality="20-13-3")))
print(f"onnx backend, CPU: {t_onnx:.1f} s including loading the graphs")
for name, a in compare:
    d = err(vec(a), vec(a_onnx))[valid_ref]
    print(f"onnx against {name}: relative L2 difference mean {d.mean():.2e}, largest {d.max():.2e}; "
          f"largest absolute difference {np.abs(a_onnx - a).max():.2e}")
print("mean relative L2 against the source: " + ", ".join(
    f"{name} {err(x_ref, vec(a))[valid_ref].mean():.5f}" for name, a in [("onnx", a_onnx)] + compare))
""")


def build(path=None):
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "demo_tessera.ipynb")
    nbformat.write(nb, path)
    return path


if __name__ == "__main__":
    print(build())
