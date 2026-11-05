# The Tessera release codec

*Draft for review.*

This document describes the released georvq codec for the TESSERA v1.1
annual embeddings, the global run produced and hosted by dClimate,
which geotessera 0.11 reads by default from Source Cooperative. The
model that this codec uses is [released on Hugging Face](https://huggingface.co/earth-compress/georvq-tessera-v1.1).
We use this codec to encode each 0.1 degree tile of one year into a
Zarr store that holds the compressed records used by the decoder. These
stores can be read from wherever they are hosted at any of 42 bitrates,
with windowed range reads over HTTP, from 0.32 to 9.9 bits per pixel,
with the relative L2 error of the 128-band vector going from 0.32 to
0.13 averaged over the sample tiles. The code that this release uses is
in `georvq/tessera/v1_1.py`, with the tile reader in
`georvq/tessera/tiles.py` and the index in `georvq/tessera/index.py`, on
the shared store codec and reader in `georvq/inference/store_codec.py`
and `georvq/inference/store_dataset.py`, and the store format is
defined in `georvq/inference/records.py`.

The codec is for the v1.1 embeddings only. The earlier TESSERA v1
embeddings, which the superseded Tessera v2 release coded, are a
different embedding space, so a model trained on one does not code the
other. The codec format (`georvq-tessera/v1.1`), the bundle format,
each store header's `product.embeddings` and the bundle's
`embeddings` all name the version, so `georvq.open` refuses a bundle
for other embeddings, and `encode_tile` refuses a tile of them.

## Who is this for

| you are | recommended points of entry |
|---|---|
| Tessera data user, reading tiles | `georvq.open`, the Reading and Rate sections below |
| archive operator, bulk-encoding tiles | `python -m georvq.tessera.v1_1 encode` on a GPU, the quality array, the Encoding cost section |
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

Each store codes one 0.1 degree tile of one year, cut from the v1.1
Zarr store as the chip-store builder (`scripts/build_store_tessera.py
--dataset v1.1`) cuts it: on the grid of the UTM zone that holds the
tile's centre, the pixel envelope of the cell's footprint, so a tile is
about 1110 rows by 1110 x cos(latitude) columns, and its CRS is the
northern UTM zone (southern tiles have negative northings, as in the
v1.1 store). The stores are named `{year}/grid_{lon}_{lat}.georvq.zarr`,
and each header records the slice of the v1.1 store's embeddings array
it was read from as `product.source`, and the tile's place in the zone's
grid as `product.zone_window`.

A tile index, `tessera_georvq_index.parquet`, is the place to find which
coded store covers an area and how many bytes it costs at each rate. It
has the columns every georvq index has (the store's path, product id,
source URL, CRS, bounds and footprint, `bitstream_id`, sizes and bytes
per rung), followed by Tessera's own, taken from each store's header:
`year`, `tile_lon`, `tile_lat`, `utm_zone`, `zone_row`, `zone_col`,
`valid_fraction` and `embeddings_version`. It is built by
`python -m georvq.tessera.v1_1 index` (`georvq/tessera/index.py`) and
published with the stores. The stores are plain Zarr v3, so zarr-python
and xarray open them directly, and the georvq header is JSON in the
group's attributes.

Records are grouped into square chunks, with one array per kind of
record (`docs/GEORVQ_ZARR.md`). A tile is not a multiple of the 64 px
window, so the last row and column of windows are padded with invalid
pixels and coded, and the mask restores the tile's extent:

| kind | chunk | per window |
|---|---|---|
| `base` (`mask` chunked alike) | 2048 px (the whole tile) | mean (160 B) and the 20 coarse levels (20 B), 180 B |
| `mid` | 2048 px (the whole tile) | per level, 16 cells x 11 bits, 22 B |
| `fine` | 1024 px (two or four per tile) | per level, 256 cells x 12 bits, 384 B |
| `quality/<decoder_id>` | 2048 px | mean and p99 of the relative L2 error per rung, 84 B |

The chunks are this small because a tile is about 1100 x 800 px and
each request to a CDN costs more than the bytes of a small chunk. On
the Bucharest tile of the sample (2024, fully valid) the store is
1.22 MB, of which `base` takes 0.044 MB, the mask 0.002 MB, `mid`
0.071 MB, `fine` 1.045 MB, `quality` 0.014 MB and the group metadata
0.048 MB. The eleven sample stores take 0.52 to 1.63 MB, against 58 to
163 MB for a tile's int8 codes and float32 scales.

## Reading

`georvq.open(store, ...)`, given a path, a URL or an open `zarr.Group`,
returns a lazy-loaded read-only dataset with the tile's georeferencing
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

ds = georvq.open("2024/grid_26.15_44.45.georvq.zarr")
w = ((0, 512), (0, 512))                               # rows, cols in pixels
a = ds.read(window=w, quality="20-13-0")               # (128, 512, 512) float32
b = ds.read(window=w, quality="p99<=0.3")              # per window, the cheapest rung meeting the bound
```

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
eleven sample tiles, no window read to `p99<=0.3` exceeded the bound
when decoded and scored against the source.

To encode tiles, read them from the v1.1 store with
`georvq.tessera.tiles.TesseraTile.read(year, lon, lat)` (a whole tile is
about 2,500 chunk requests and took about three minutes from our server)
or from the command line:

```bash
python -m georvq.tessera.v1_1 encode out_dir/ --year 2024 --tile 26.15,44.45 --bundle <bundle dir>
python -m georvq.tessera.v1_1 index out_dir/ out_dir/tessera_georvq_index.parquet --url-prefix ""
```

## Rate ladder

The 42 rungs fill one tier at a time: coarse from 4 to 20 levels, then
mid from 1 to 13, then fine from 1 to 12. A level costs 8 bits per
window in the coarse tier, 176 in mid and 3,072 in fine, and every rung
also carries the 1280-bit mean. The relative L2 column is pooled over
the valid pixels of the eleven sample tiles, decoded in bf16 as a
reader decodes them. The tiles span 2017 to 2025 and latitudes 45 S to
72 N, and at full depth they range from 0.026 (Greenland ice sheet) to
0.232 (Bucharest).

| rung | bits/window | b/px | relative L2 |
|---|---|---|---|
| 4-0-0 | 1,312 | 0.320 | 0.318 |
| 20-0-0 | 1,440 | 0.352 | 0.296 |
| 20-2-0 | 1,792 | 0.438 | 0.264 |
| 20-6-0 | 2,496 | 0.609 | 0.241 |
| 20-13-0 | 3,728 | 0.910 | 0.225 |
| 20-13-1 | 6,800 | 1.660 | 0.202 |
| 20-13-3 | 12,944 | 3.160 | 0.175 |
| 20-13-6 | 22,160 | 5.410 | 0.154 |
| 20-13-12 | 40,592 | 9.910 | 0.133 |

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

A sample of eleven encoded tiles with their index is on Hugging Face as
[`earth-compress/georvq-tessera-v1.1-sample`](https://huggingface.co/datasets/earth-compress/georvq-tessera-v1.1-sample),
in the same `{year}/` layout, and `georvq.open` reads them by URL, e.g.
`https://huggingface.co/datasets/earth-compress/georvq-tessera-v1.1-sample/resolve/main/2024/grid_26.15_44.45.georvq.zarr`.

## Encoding cost

Encoding a tile on an A100, with the quality values that take one
decode per rung at all 42 rungs, takes about 5 s, once the tile has
been read.
