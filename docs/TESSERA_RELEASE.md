# The Tessera release codec

*Draft for review.*

This document describes the released georvq codec for the TESSERA v1.1
annual embeddings, the global run produced and hosted by dClimate,
which geotessera 0.11 reads by default from Source Cooperative. The
model that this codec uses is [released on Hugging Face](https://huggingface.co/earth-compress/georvq-tessera-v1.1).
We use this codec to encode each 4096 x 4096 px shard of the source's
zone arrays, in one year, into a Zarr store that holds the compressed
records used by the decoder, so that the coded stores have the source's
layout, as the AlphaEarth release's stores have their source tiles'.
These stores can be read from wherever they are hosted at any of 42
bitrates, with windowed range reads over HTTP, from 0.32 to 9.9 bits
per pixel, with the relative L2 error of the 128-band vector going from
0.29 to 0.12 averaged over the sample's stores. A 0.1
degree geotessera tile lies in one to four stores, and
`georvq.tessera.read_tile` reads it from them through the release's
index. The code that this release uses is in `georvq/tessera/v1_1.py`,
with the shard and tile readers in `georvq/tessera/tiles.py` and the
index and `read_tile` in `georvq/tessera/index.py`, on
the shared store codec and reader in `georvq/inference/store_codec.py`
and `georvq/inference/store_dataset.py`, and the store format is
defined in `georvq/inference/records.py`.

The codec is for the v1.1 embeddings only. The earlier TESSERA v1
embeddings, which the superseded Tessera v2 release coded, are a
different embedding space, so a model trained on one does not code the
other. The codec format (`georvq-tessera/v1.1`), the bundle format,
each store header's `product.embeddings` and the bundle's
`embeddings` all name the version, so `georvq.open` refuses a bundle
for other embeddings, and `encode_shard` refuses a shard of them.

## Who is this for

| you are | recommended points of entry |
|---|---|
| Tessera data user, reading tiles | `georvq.tessera.read_tile`, `georvq.open`, the Reading and Rate sections below |
| archive operator, bulk-encoding shards | `python -m georvq.tessera.v1_1 encode` on a GPU, the quality array, the Encoding cost section |
| codec researcher | `georvq/{model,train,...}` and `docs/OVERVIEW.md`; `georvq/tessera/v1_1.py` is a worked example of a release layer on the Zarr store |

## Model details

This information is for readers familiar with the georvq codebase, and
can be skipped by those who only want the data.

- The model is the `tessera_v11` run (`configs/tessera/tessera_v11.yaml`)
  at step 133,000, a `stats_cond` model which transmits each window's
  mean, with 8-bit (int8) codebooks. Its window is 64 px with 128 bands,
  its tiers are `1:256:256:20` / `4:256:2048:13` / `16:256:4096:12`, and
  its minimum depths are `4-0-0`. Validation stopped improving from
  about step 131,000, at a relative L2 error of 0.137 at full depth
  against an unquantised ceiling of 0.099. The weights are frozen, and
  there is no further training or fine-tuning.
- The model works in x = q * scale / 32 (`XNORM`), and a pixel is valid
  where its vector is not all zero and 0 < scale < 1. The v1.1 store
  marks a pixel without an embedding by a NaN scale, which the reader
  turns into the same nodata.
- Each window's mean is subtracted before the encoder and added back
  after the decoder, so it has to be transmitted with every window, at
  every rung. It costs 1280 bits (160 bytes) per window, against 32 bits
  for the four coarse levels of the cheapest rung, so at the lowest
  rates the mean is nearly all of the bitstream. The model was trained
  with exact means, and its finer tiers do not correct an error in the
  mean: a coded mean's error appears in the decode unchanged at every
  rung, the full depth included. The mean is coded by `RotatedChannel`,
  rotated onto the principal axes of 506,161 training window means and
  each component companded with asinh by its standard deviation and
  coded in 7 to 13 bits over a fixed range. The range is the hard bound
  of the data: every value of a valid pixel is at most 127 times its
  scale, which is below 1, so |x_d| <= 127 / 32 = 3.97, against 0.63 for
  the largest value of the 10.8 billion valid pixels of the training
  store, so no window's mean is ever clipped. The width was the smallest
  of those tried (in steps of 128 bits) for which, on 22 tiles and on
  144 validation chips, the decoded error with the coded mean stayed within
  0.5% of the error with the exact mean at 4-0-0, 20-0-0, 20-13-0 and
  20-13-12; 1152 bits failed on the Greenland ice sheet, where the
  decoded error is small enough for the mean's error to show. The
  channel is fitted by `scripts/fit_mean_channel.py --box`, and its
  tables enter the `bitstream_id`.
- The `bitstream_id` is `74feb29cecbc2ba7` and the `decoder_id` is
  `09bba7c1316e71ea`. A store records the `bitstream_id` and
  `decoder_id` it was encoded with, and a reader holding a different
  codec refuses to decode it.
- Decoding runs on PyTorch, on CPU or CUDA. Decoder passes run in bf16
  on CUDA by default (`bf16=False` turns this off) and in fp32 on CPU,
  whilst encoder passes always run in fp32. On the sample tiles the mean
  relative L2 error of a bf16 decode is the same as an fp32 decode's to
  four decimal places. Decode can also run through ONNX Runtime, which
  needs the `[onnx]` extra, by passing `backend="onnx"`. Its output
  matches torch fp32 to about 2e-5.

## Data format

Each store codes one shard of the v1.1 Zarr store's arrays in one year:
the 4096 x 4096 px region of a UTM zone's 10 m grid that one shard of
the source's `embeddings` and `scales` arrays covers, aligned with it
exactly, in the zone's CRS (the northern UTM zone, so southern shards
have negative northings, as in the v1.1 store) and transform. The
stores are named `{year}/utm{zone}/{row}_{col}.georvq.zarr` by the
shard's row and column on the zone's shard grid, the row zero-padded
to three digits and the column to two, which covers every zone's grid
(446 shard rows from 84 N to 80 S and 17 columns), so that names sort
in grid order. Each header records the shard slice of the zone's
embeddings array as `product.source`, and the shard's place in the
zone's grid as `product.shard`, `product.zone_window` and
`product.zone_origin`. A shard with no valid pixel has no store. A
shard is 64 x 64 windows, so the window grid of every store is the
zone's.

The index, `tessera_georvq_index.parquet`, is the place to find which
coded store covers an area or a tile and how many bytes it costs at
each rate. It has the columns every georvq index has (the store's path,
product id, source URL, CRS, bounds and footprint, `bitstream_id`,
sizes and bytes per rung), followed by Tessera's own, taken from each
store's header: `year`, `utm_zone`, `shard_row`, `shard_col`,
`zone_row`, `zone_col`, `zone_west`, `zone_top`, `zone_px`,
`valid_fraction`, `embeddings_version` and `tiles`, the geotessera
tiles whose window touches the store, each with its name, centre and
window in the zone's grid. A geotessera tile is cut as the chip-store
builder (`scripts/build_store_tessera.py --dataset v1.1`) cuts it: on
the grid of the UTM zone that holds the tile's centre, the pixel
envelope of the cell's footprint, about 1110 rows by 1110 x
cos(latitude) columns, so it lies in one to four stores and is listed
in each. The index is built by `python -m georvq.tessera.v1_1 index`
(`georvq/tessera/index.py`) and published with the stores. The stores
are plain Zarr v3, so zarr-python and xarray open them directly, and
the georvq header is JSON in the group's attributes.

Records are grouped into square chunks, with one array per kind of
record (`docs/GEORVQ_ZARR.md`):

| kind | chunk | per window |
|---|---|---|
| `base` (`mask` chunked alike) | 2048 px (four per store) | mean (160 B) and the 20 coarse levels (20 B), 180 B |
| `mid` | 2048 px (four per store) | per level, 16 cells x 11 bits, 22 B |
| `fine` | 1024 px (16 per store) | per level, 256 cells x 12 bits, 384 B |
| `quality/<decoder_id>` | 2048 px | mean and p99 of the relative L2 error per rung, 84 B |

The chunks are this small because the usual read is of one geotessera
tile, about 1100 x 800 px, which touches one to four chunks of `base`
and `mid` and two to nine of `fine`, and each request to a CDN costs
more than the bytes of a small chunk. A fully valid store is about
20.8 MB, of which `fine` is nearly all; the 17 sample stores take 8.4 to
20.9 MB (316 MB in all), against 2.2 GB for a shard's int8 codes and
float32 scales uncompressed.

## Reading

`georvq.open(store, ...)`, given a path, a URL or an open `zarr.Group`,
returns a lazy-loaded read-only dataset with the shard's georeferencing
and shape (`crs`, `transform`, `width`, `height`, `count` (128),
`dtypes` (float32), `nodata` (0.0), `profile`, `bounds`, `res` and
rasterio's window helpers), whose `read` takes rasterio's `window`
argument and `quality`, the rate. Reads return float32 embeddings in
the source's units (`q * scale`); `units="model"` returns the model's
x = source / 32 instead. Invalid pixels come back as zero vectors, from
the stored mask. `georvq.tessera.v1_1.to_geotessera` turns output back
into int8 codes of shape (rows, cols, 128) and a float32 scale per
pixel, set so that every valid pixel has a code of 127 or -127, as in
the source.

The store's header names the release that decodes it, and `open` finds
that release's bundle by its `bitstream_id`, either at
`GEORVQ_TESSERA_BUNDLE`, in a local cache
(`~/.cache/georvq/bundles/<bitstream_id>/`), or by downloading the
decoder-only bundle from the `earth-compress` organisation on Hugging
Face ([`earth-compress/georvq-tessera-v1.1`](https://huggingface.co/earth-compress/georvq-tessera-v1.1)).
A bundle whose ids do not match the store is refused, including one at
`GEORVQ_TESSERA_BUNDLE`. The device defaults to CUDA when available,
and the backend to torch.

```python
import georvq
import georvq.tessera

ds = georvq.open("2024/utm35/107_06.georvq.zarr")
w = ((0, 512), (0, 512))                               # rows, cols in pixels
a = ds.read(window=w, quality="20-13-0")               # (128, 512, 512) float32
b = ds.read(window=w, quality="p99<=0.3")              # per window, the cheapest rung meeting the bound

index = "tessera_georvq_index.parquet"                 # a path or a URL
t = georvq.tessera.read_tile(index, 26.15, 44.45, 2024, quality="20-13-0")
info = georvq.tessera.find_tile(index, "2024/grid_26.15_44.45")
# t is float32 (128, height, width) on the tile's window; info has its
# window, transform, EPSG code and the stores it was read from
```

`read_tile(index, lon, lat, year, quality=..., window=...)` reads the
tile centred at (lon, lat), or named as `"grid_26.15_44.45"`, from the
stores the index lists for it, and assembles them, so the result has the
shape, georeferencing and nodata of the tile, with the windows of the
shard stores decoded. It takes the same `quality` as `read`, a
`window` in the tile's pixels, and `georvq.open`'s keywords, and reuses
the codec of the first store for the others. Reading a tile this way is
not bit for bit the read of a store coded per tile, since the windows
lie on the shard grid rather than the tile's own; on the eleven sample
tiles the two reads differ from each other by about as much as each
differs from the source (0.015 to 0.26 in relative L2), whilst their
errors against the source agree to within 1% of each other.

`quality` takes a rung (`"20-13-0"`, `"20,13,0"` or a tuple), a bound
on a quality statistic (`"mean<=X"` or `"p99<=X"`), or `None` for full
depth. A bound is resolved per window to the cheapest rung whose
encode-time statistic meets it, and to full depth where none does. The
statistics are the relative L2 error `||decoded - source|| /
||source||` of the decoded float field against the prepared source,
over the window's valid pixels, measured with the decode precision the
header records (bf16 for the sample). They are stored with 32 codes per
decade, a step of about 7.5%, rounded up to the next step, so a window
read to a bound that one of its rungs meets is never above it. On the
17 sample stores (four 1024 px blocks of each), no window read to `p99<=0.3` exceeded the bound
when decoded and scored against the source.

To encode shards, read them from the v1.1 store with
`georvq.tessera.tiles.TesseraShard.read(year, zone, row, col)`, which
reads a strip at a time, or from the command line, by shard or by the
tiles whose shards are wanted:

```bash
python -m georvq.tessera.v1_1 encode out_dir/ --year 2024 --shard 35,107,6 --bundle <bundle dir>
python -m georvq.tessera.v1_1 encode out_dir/ --year 2024 --tile 26.15,44.45 --bundle <bundle dir>
python -m georvq.tessera.v1_1 index out_dir/ out_dir/tessera_georvq_index.parquet --url-prefix ""
```

A shard's source arrays are 0.5 to 1.8 GB of compressed codes and 14
to 50 MB of scales. Read through zarr-python they take thousands of
chunk requests, so for the sample each shard object was fetched whole
into a partial local copy of the v1.1 store and read from there: the 17
shards, 24 GB, took 29 minutes with 16 concurrent range requests, a
single request reaching about 4 MB/s from our server.

## Rate ladder

The 42 rungs fill one tier at a time: coarse from 4 to 20 levels, then
mid from 1 to 13, then fine from 1 to 12. A level costs 8 bits per
window in the coarse tier, 176 in mid and 3,072 in fine, and every rung
also carries the 1280-bit mean. The relative L2 column is from the 17
sample stores' quality values (each window's mean, weighted by its valid
pixels, over every window, decoded in bf16 as a reader decodes them),
which are rounded up by at most 7.5%; a decode of 68 sampled 1024 px
blocks of the stores gave errors 5 to 6% lower. The stores span 2017 to
2025 and latitudes 45 S to 72 N, and at full depth they range from 0.027
(Greenland ice sheet) to 0.20 (Hohe Tauern).

| rung | bits/window | b/px | read b/px | relative L2 |
|---|---|---|---|---|
| 4-0-0 | 1,312 | 0.320 | 0.337 | 0.294 |
| 20-0-0 | 1,440 | 0.352 | 0.337 | 0.272 |
| 20-2-0 | 1,792 | 0.438 | 0.412 | 0.241 |
| 20-6-0 | 2,496 | 0.609 | 0.567 | 0.219 |
| 20-13-0 | 3,728 | 0.910 | 0.843 | 0.204 |
| 20-13-1 | 6,800 | 1.660 | 1.425 | 0.185 |
| 20-13-3 | 12,944 | 3.160 | 2.727 | 0.161 |
| 20-13-6 | 22,160 | 5.410 | 4.731 | 0.142 |
| 20-13-12 | 40,592 | 9.910 | 8.815 | 0.122 |

"read b/px" is the bytes a whole-store read fetches, as the index
records them, per pixel of the 17 stores: the records compressed with
zstd, the mask and the shard indexes included.

The header's `rungs` field lists all 42 with their bits per window and
per pixel.

## Getting the model

Weights are not in git. They are published on Hugging Face as
[`earth-compress/georvq-tessera-v1.1`](https://huggingface.co/earth-compress/georvq-tessera-v1.1),
without pickled files: model tensors, codebooks, mean-channel tables,
the rung table and a `bundle.json` holding the ids. The full bundle at
the root of the repository (904.8 MB, of which 442.2 MB is the ONNX
export) encodes and decodes; `decoder/` (270.1 MB) is the decoder-only
bundle that `georvq.open` downloads.

```bash
pip install -e .                    # decode and encode
pip install -e ".[onnx]"            # adds ONNX Runtime decode
hf download earth-compress/georvq-tessera-v1.1 --local-dir georvq-tessera-v1.1
```

A sample of 17 encoded stores, every shard that eleven chosen tiles
touch, with their index is on Hugging Face as
[`earth-compress/georvq-tessera-v1.1-sample`](https://huggingface.co/datasets/earth-compress/georvq-tessera-v1.1-sample),
in the same `{year}/utm{zone}/` layout, and `georvq.open` reads them by
URL, e.g.
`https://huggingface.co/datasets/earth-compress/georvq-tessera-v1.1-sample/resolve/main/2024/utm35/107_06.georvq.zarr`.

## Encoding cost

Encoding a shard on an A100, with the quality values that take one
decode per rung at all 42 rungs, took 53 to 490 s (230 s on average)
for the 17 sample stores, once the shard had been read, on a machine
whose other card and most of whose CPUs were in use by other jobs.
