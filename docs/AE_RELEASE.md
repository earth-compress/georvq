# The AlphaEarth release codec

This document describes the released georvq codec for the AlphaEarth Foundations annual
embedding tiles. The model that this codec uses is [released on Hugging Face](https://huggingface.co/earth-compress/georvq-ae-v2). 
We use this codec to encode source tiles into a Zarr store that contains the compressed
blobs used by the decoder. These Zarr stores can be streamed from their location on the cloud at any of 23 bitrates, with windowed
range reads over HTTP, allowing users to access lossily compressed AlphaEarth Foundations data at bitrates from ~0.1-7.5 bits per pixel, with cosine similarity going from 97.2% to 99.8%. The code that this release uses is in `georvq/ae/v2.py`, on the shared store codec and reader in `georvq/inference/store_codec.py` and `georvq/inference/store_dataset.py`, and the store format is defined in `georvq/inference/records.py`.

## Who is this for

| you are | recommended points of entry |
|---|---|
| AE data user, reading tiles | `georvq.open`, `python -m georvq.ae.v2 decode/reconstruct/info`, the Reading and Rate sections below |
| archive operator, bulk-encoding source tiles | `python -m georvq.ae.v2 encode` on a GPU, the quality array, the Encoding cost section |
| codec researcher | `georvq/{model,train,...}` and `docs/OVERVIEW.md`; `georvq/ae/v2.py` is a worked example of a release layer on the Zarr store |

## Model details

Note: this information only makes sense if you are familiar with the georvq codebase, and you can just ignore it if you're more interested in the data.

- The model is a `stats_cond` model, which transmits each window's mean,
  with 8-bit (int8) codebooks. Its window is 64 px with 64 bands, its
  tiers are `1:256:2048:8` / `4:192:8192:8` / `16:64:16384:8`, and its
  minimum depths are `2-0-0`. The weights are frozen, and there is no
  further training or fine-tuning.
- Each window's mean is subtracted before the encoder and added back
  after the decoder, so it has to be transmitted with every window, at
  every rung. It costs 440 bits (55 bytes) per window, against 22 bits
  for the two coarse levels of the cheapest rung, so at the lowest rates
  the mean is most of the bitstream. It is coded by
  `RVQPCAMeanChannel`, with two RVQ levels of 4096 codes (2 x 12 bits)
  followed by a PCA residual coder with a fixed allocation of 416 bits.
  Its median relative L2 error on held-out validation means is 0.0060,
  and on the validation set, decoding with the coded mean instead of the
  exact one costs about 0.0001 cosine at every rung. The mean channel's
  tables enter the `bitstream_id`.
- The `bitstream_id` is `f6ad2ff121a473df` and the `decoder_id` is
  `d9b0af741f66b682`. A store records the `bitstream_id` and `decoder_id`
  it was encoded with, and a reader holding a different codec refuses to
  decode it.
- Decoding runs on PyTorch, on CPU or CUDA. Decoder passes run in bf16
  on CUDA by default (`--fp32` or `bf16=False` turns this off) and in
  fp32 on CPU, whilst encoder passes always run in fp32. Decode can also
  run through ONNX Runtime, which needs the `[onnx]` extra (onnxruntime
  1.19 or later), by passing `backend="onnx"`, or `--backend onnx` on
  the command line. Its output matches torch to about 1e-6.

## Data Format

We are using a mirrored structure to [Taylor Geospatial's AlphaEarth release on source.coop](https://source.coop/tge-labs/aef), where each of our Zarr stores corresponds to one of their 8192x8192 tiles. Because of the structure of the georvq model's outputs, we've opted to use a Zarr store so that we can make independent chunks for different parts of the model's output. These stores are named `{source stem}.georvq.zarr` and
placed at the path that mirrors the source tile's path
(`annual/{year}/{zone}/{stem}.georvq.zarr`). 

A tile index, `aef_georvq_index.parquet`, is the place to find which
coded store covers an area and how many bytes it costs at each rate. It
has the source index's columns with paths to the coded stores, plus each
store's `bitstream_id` and bytes per rung. It is built by
`scripts/build_ae_index.py` (`georvq/ae/index.py`) and published with the
stores, and the sample's index is in the
[sample dataset](https://huggingface.co/datasets/earth-compress/georvq-ae-v2-sample).
The stores are plain Zarr v3, so zarr-python and xarray open them
directly. The georvq header is JSON in the group's attributes,
and the source tile's profile, tags, band descriptions and orientation
are kept as JSON in `georvq_aux`. 

Records are grouped into square chunks, with one array per kind of
record. The chunk size differs by kind of record (`docs/GEORVQ_ZARR.md`):
one `base` chunk covers the whole tile, and `fine` is chunked at 2048 px:

| kind | chunk | per window |
|---|---|---|
| `base` (`mask` chunked alike) | 8192 px (the whole tile) | mean (55 B) and the 8 coarse levels (11 B), 66 B |
| `mid` | 8192 px (the whole tile) | per level, 16 cells x 13 bits, 26 B |
| `fine` | 2048 px (4 x 4 chunks) | per level, 256 cells x 14 bits, 448 B |
| `quality/<decoder_id>` | 8192 px | mean and p99 of `1 - cos` per rung, 46 B |

Each level of a chunk of a tier is its own inner chunk of the tier's
shard, so reading a tier to depth `L` reads `L` inner chunks per chunk,
which zarr-python fetches together. Code arrays are stored uncompressed
with fixed bit widths. The validity mask is stored once for the tile,
compressed, and has no stored chunk for a fully valid tile, and windows
with no valid pixel are not coded. On the tile measured below (2021,
zone 44N, 11% nodata) the store is 64.0 MB, of which `base` takes
1.08 MB, the mask 0.01 MB, `mid` 3.41 MB, `fine` 58.7 MB, `quality`
0.76 MB and the group metadata 0.05 MB.

Functionally, most users will not need to worry about anything beyond this, as our reader handles the filesystem and requests for you. For those interested to learn more about how those stores are arranged, refer to `docs/GEORVQ_ZARR.md`. 

## Reading

The main principle being followed in the reader's design is to mirror the behaviour of a rasterio file object's read method, so that after replacing `rasterio.open()` with `georvq.open()` in your code, the `.read` will work as it did before. This is to make the transition as straightforward as possible for people who are already using the AlphaEarth Foundations dataset on source.coop.

`georvq.open(store, ...)`, given a path, a URL or an open `zarr.Group`, returns a lazy-loaded read-only dataset with the
source tile's georeferencing and shape: `crs`, `transform`, `width`,
`height`, `count` (64), `dtypes` (int8), `nodata` (-128), `profile`,
`meta`, `tags()`, `descriptions`, `block_shapes`, `block_windows()`,
`bounds`, `res`, and rasterio's window helpers (`index`, `xy`,
`window`). `driver` is "GeoRVQZarr", whilst `profile` and `meta`
carry the source's driver ("GTiff"), so either can be passed to
`rasterio.open(..., "w")`.

The AEF source tiles store their rows bottom-up. The store codes them
north-up, but `read`, `read_masks` and `dataset_mask` return them in the
source's own order, with rasterio's arguments and results, so code
written against the rasterio dataset of the source tile can read the
store instead. Two differences remain. The store has no overviews, so a
decimated read samples the full-resolution decode, and there is no
`colorinterp`. Invalid pixels come back as -128 exactly, from the
stored mask.

The store's header names the release that decodes it, and `open` finds
that release's bundle by its `bitstream_id`, either at
`GEORVQ_AE_BUNDLE`, in a local cache
(`~/.cache/georvq/bundles/<bitstream_id>/`), or by downloading it from
the `earth-compress` organisation on Hugging
Face ([`earth-compress/georvq-ae-v2`](https://huggingface.co/earth-compress/georvq-ae-v2)).
A bundle whose ids do not match the store is refused, including one at
`GEORVQ_AE_BUNDLE`. The device defaults to CUDA when available, and the
backend to torch.

```python
import georvq

ds = georvq.open("annual/2021/44N/<stem>.georvq.zarr")
w = ((0, 1024), (0, 1024))                            # rows, cols in source pixels
a = ds.read(window=w, quality="8-8-2")                # (64, 1024, 1024) int8
b = ds.read(3, window=w, quality="p99<=0.02")         # band 3 only, 1-based
m = ds.read_masks(window=w)                           # (64, 1024, 1024) uint8, 255 valid
v = ds.dataset_mask(window=w)                         # (1024, 1024) uint8
```

`window` takes a rasterio `Window` or `((row0, row1), (col0, col1))` in
source pixels. `quality` takes a rung (`"8-8-2"`, `"8,8,2"` or a
tuple), a bound on a quality statistic (`"mean<=X"` or `"p99<=X"`), or
`None` for full depth. A bound is resolved per window to the cheapest
rung whose encode-time statistic meets it, and to full depth where none
does, at the cost of one extra read of the quality array. You should
expect a bound to hold to within a few percent rather than exactly,
because the statistics are stored log-quantised. The statistics are
`1 - cos` of the decoded float field against the prepared source, over
the window's valid pixels, measured with the decode precision the header
records. They are stored with 32 codes per decade, a step of about 7.5%,
and the bound is applied to the stored value, so a bound is met to within
half a step. For example, `p99<=0.02` admits windows whose measured p99
is up to about 0.0198.

The same reads are available from the command line, together with
commands to inspect a store, fetch the release it names and encode new
tiles:

```bash
python -m georvq.ae.v2 info  tile.georvq.zarr             # header, rungs, bytes per array
python -m georvq.ae.v2 fetch tile.georvq.zarr             # download the release it names
python -m georvq.ae.v2 decode tile.georvq.zarr out.npy \
    --window 0,0,1024,1024 --quality 8-8-2                 # or --quality "p99<=0.02"
python -m georvq.ae.v2 reconstruct tile.georvq.zarr recon.tif --quality 8-8-2
python -m georvq.ae.v2 encode src.tif out_dir/ --bundle <bundle dir> --device cuda
```

`decode` writes a `.npy` in source row order and prints the bytes of
records it read. `reconstruct` writes a tiled GeoTIFF with the
source's profile, block size, compression, tags and band descriptions,
although COG layout and overviews are not reproduced. `encode` writes
`{stem}.georvq.zarr` into the output directory (the current one by
default), and `--no-quality` leaves out the quality array, which
otherwise costs one decode per rung at encode time.

The other `encode` options set the provenance that the header records.
`--source` sets the path or URL of the source tile. It defaults to the
`src` argument as given, and `encode` warns when that is a local
absolute path. `--release`, `--release-uri` and `--revision` set the
release record that readers use to find the bundle.

## Rate ladder

The 23 rungs fill one tier at a time: coarse from 2 to 8 levels, then
mid from 1 to 8, then fine from 1 to 8. A level costs 11 bits per
window in the coarse tier, 208 in mid and 3,584 in fine, and every rung
also carries the 440-bit mean. The table gives two quality columns.
"val cos" is cosine on the held-out validation windows, scored through
the training run's own validation harness with the coded mean at both
ends, so it is the figure that the training run itself reports. "Real
tiles" is closer to what a data user will see. It is the int8
reconstruction against the int8 source over the valid pixels of 16
held-out 8192 px tiles, pixel-weighted. The tiles span 2018-2025,
latitudes -75 to 75 and valid fractions from 0.51 to 1.00, and at full
depth they range from 0.9963 to 0.9998.

| rung | bits/window | b/px | val cos | real tiles |
|---|---|---|---|---|
| 2-0-0 | 462 | 0.113 | 0.9722 | 0.9733 |
| 8-0-0 | 528 | 0.129 | 0.9798 | 0.9805 |
| 8-1-0 | 736 | 0.180 | 0.9858 | – |
| 8-2-0 | 944 | 0.230 | 0.9886 | – |
| 8-4-0 | 1,360 | 0.332 | 0.9912 | 0.9913 |
| 8-8-0 | 2,192 | 0.535 | 0.9932 | 0.9933 |
| 8-8-1 | 5,776 | 1.410 | 0.9951 | – |
| 8-8-2 | 9,360 | 2.285 | 0.9962 | 0.9963 |
| 8-8-4 | 16,528 | 4.035 | 0.9973 | 0.9974 |
| 8-8-8 | 30,864 | 7.535 | 0.9981 | 0.9981 |

The header's `rungs` field lists all 23 with their bits per window and
per pixel, and `info` prints them.

## Getting the model

Weights are not in git. They are published on Hugging Face as
[`earth-compress/georvq-ae-v2`](https://huggingface.co/earth-compress/georvq-ae-v2), without pickled
files: model tensors, codebooks, mean-channel tables, the rung table
and a `bundle.json` holding the ids.

```bash
git clone <this repo> && cd georvq
pip install -e .                    # decode, encode and reconstruct
pip install -e ".[onnx]"            # adds ONNX Runtime decode
hf download earth-compress/georvq-ae-v2 \
    --local-dir georvq-ae-v2
```

A sample of six encoded tiles with their index is on Hugging Face as
[`earth-compress/georvq-ae-v2-sample`](https://huggingface.co/datasets/earth-compress/georvq-ae-v2-sample), in the same
`annual/{year}/{zone}/` layout, and `georvq.open` reads them by URL, e.g.
`https://huggingface.co/datasets/earth-compress/georvq-ae-v2-sample/resolve/main/annual/2025/32N/xh72hhha3ih6mroc7-0000008192-0000008192.georvq.zarr`.
The full set of encoded tiles will be hosted at
`<host>/annual/{year}/{zone}/{stem}.georvq.zarr` `[pending: hosting]`.

## Encoding and decoding cost

The data comes with precomputed quality metrics that are calculated by encoding once and then decoding 
at 23 different bitrates. This means that the time taken for encoding is actually dominated by decodes. The encoding takes about 20 s per tile on an A100, but with quality measures, the process takes 113 s per tile. 

For a user, decoding on one A100 takes 7-9 s for a whole tile at any rung, because the decoder runs at full resolution whatever the depth, and fetching the codes is under 1.3 s of that. A single 1024 px block takes about 0.1 s of GPU time, so reading one over HTTP is dominated by the requests: about 1.6 s in all at the latency and throughput measured from our server to Source Cooperative. On CPU through ONNX Runtime, decoding runs at about 45 windows/s
with 16 threads, about 6 minutes for a full tile.
