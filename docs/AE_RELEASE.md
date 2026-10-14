Draft for review: wording not yet approved by the maintainer; `[pending: ...]` marks values still being measured or built.

# The AlphaEarth release codec

This document describes the released georvq codec for the AlphaEarth Foundations annual
embedding tiles. The model that this codec uses is [released on HuggingFace]([pending: hosting]). 
We use this codec to encode source tiles into a geoparquet file that contains the compressed
blobs used by the decoder. These GeoParquet files can be streamed from their location on the cloud at any of 23 bitrates, with windowed
range reads over HTTP, allowing users to access lossily compressed AlphaEarth Foundations data at bitrates from ~0.1-7.5 bits per pixel, with cosine similarity going from 97.2% to 99.8%. The code that this release uses is in `georvq/ae/v2.py`, and the GeoParquet file format is defined in `georvq/inference/georvq_parquet.py`.

## Who is this for

| you are | recommended points of entry |
|---|---|
| AE data user, reading tiles | `georvq.ae.open`, `python -m georvq.ae.v2 decode/reconstruct/info`, the Reading and Rate sections below |
| archive operator, bulk-encoding source tiles | `python -m georvq.ae.v2 encode` on a GPU, the quality column, the Encoding cost section |
| codec researcher | `georvq/{model,train,...}` and `docs/OVERVIEW.md`; `georvq/ae/v2.py` is a worked example of a release layer on the GeoParquet container |

## Model details

Note: this information only makes sense if you are familiar with the georvq codebase, and you can just ignore it if you're more interested in the data.

- Model: a `stats_cond` model with 8-bit (int8) codebooks: window 64 px, 64 bands, tiers `1:256:2048:8` /
  `4:192:8192:8` / `16:64:16384:8`, minimum depths `2-0-0`. The weights
  are frozen; there is no further training or fine-tuning.
- Each window's mean is subtracted before the encoder and added back
  after the decoder, so it is transmitted with every window, at every
  rung. It is coded by `RVQPCAMeanChannel`: two RVQ levels of 4096
  codes (2 x 12 bits) followed by a PCA residual coder with a fixed
  allocation of 416 bits, 440 bits (55 bytes) per window in all. Its
  median relative L2 error on held-out validation means is 0.0060; on
  the validation set, decoding with the coded mean instead of the exact
  one costs about 0.0001 cosine at every rung. The mean channel's
  tables enter the `bitstream_id`.
- `bitstream_id`: `f6ad2ff121a473df` (`decoder_id` `d9b0af741f66b682`). A file records the
  `bitstream_id` and `decoder_id` it was encoded with, and a reader
  holding a different codec refuses to decode it.
- Backends: PyTorch, CPU or CUDA. Decoder passes run in bf16 on CUDA by
  default (`--fp32` or `bf16=False` turns this off) and in fp32 on CPU;
  encoder passes always run in fp32. Torch-free decode runs through ONNX
  Runtime (`backend="onnx"`, or `--backend onnx` on the command line);
  its output matches torch to about 1e-6.

## Data Format

We are using a mirrored structure to [Taylor Geospatial's AlphaEarth release on source.coop](https://source.coop/tge-labs/aef), where each of our geoparquet files corresponds to one of their 8192x8192 tiles. Because of the structure of the georvq model's outputs, we've opted to use a geoparquet file so that we can make independent chunks for different parts of the model's output. These files are named `{source stem}.georvq.parquet` and
placed at the path that mirrors the source tile's path
(`annual/{year}/{zone}/{stem}.georvq.parquet`). 

A tile index, `aef_georvq_index.parquet`, has the source index's
columns with paths to the coded files, plus each file's `bitstream_id`
and bytes per rung (`georvq/ae/index.py`, `scripts/build_ae_index.py`;
published with the files `[pending: hosting]`). The files are
GeoParquet 1.1, so pyarrow and DuckDB read their rows directly; the
georvq header is JSON in the file's key-value metadata, and the source
tile's profile, tags, band descriptions and orientation are kept as
JSON in `georvq_aux`. 

Records are grouped into square chunks, one row and one row group per
chunk, with a chunk size per kind of record:

| kind | chunk | per window |
|---|---|---|
| `base` (with `mask`) | 8192 px (the whole tile) | mean (55 B) and the 8 coarse levels (11 B), 66 B |
| `mid` | 4096 px (2 x 2 chunks) | per level, 16 cells x 13 bits, 26 B |
| `fine` | 1024 px (8 x 8 chunks, the source's block) | per level, 256 cells x 14 bits, 448 B |
| `quality` | 8192 px | mean and p99 of `1 - cos` per rung, 46 B |

Each level of a tier is its own column, so reading a tier to depth `L`
reads the first `L` columns of its row group, one contiguous byte range.
Code columns are stored uncompressed with fixed bit widths. The
validity mask is stored once for the tile, compressed, and is empty for
a fully valid tile; windows with no valid pixel are not coded. On the
tile measured below (2021, zone 44N, 11% nodata) the file is 64.2 MB:
`base` with its mask 1.10 MB, `mid` 3.41 MB, `fine` 58.8 MB, `quality`
0.75 MB and the footer 0.14 MB.

Functionally, most users will not need to worry about anything beyond this, as our reader handles the filesystem and requests for you. For those interested to learn more about how those files are arranged, refer to `docs/GEORVQ_PARQUET.md`. 

## Reading

The main principle being followed in the reader's design is to mirror the behaviour of a rasterio file object's read method, so that after replacing `rasterio.open()` with `georvq.ae.open()` in your code, the `.read` will work as it did before. This is to make the transition as straightforward as possible for people who are already using the AlphaEarth Foundations dataset on source.coop.

`georvq.ae.open(path_or_url, ...)` returns a lazy-loaded read-only dataset with the
source tile's georeferencing and shape: `crs`, `transform`, `width`,
`height`, `count` (64), `dtypes` (int8), `nodata` (-128), `profile`,
`tags()`, `descriptions`, `block_shapes` and `block_windows()`. The
AEF source tiles store their rows bottom-up; the file codes them
north-up and `read` returns them in the source's own order, so code
written against the rasterio dataset of the source tile can read the
file instead. Invalid pixels come back as -128 exactly, from the
stored mask. The file's header names the release that decodes it, and
`open` finds that release's bundle by its `bitstream_id`: in a local
cache (`~/.cache/georvq/bundles/<bitstream_id>/`), at `GEORVQ_AE_BUNDLE`,
or by downloading it from the `earth-compress` organisation on Hugging
Face `[pending: hosting]`. A bundle whose ids do not match the file is
refused. The device defaults to CUDA when available, and the backend to
torch when installed, otherwise ONNX Runtime.

```python
import georvq.ae

ds = georvq.ae.open("annual/2021/44N/<stem>.georvq.parquet")
w = ((0, 1024), (0, 1024))                            # rows, cols in source pixels
a = ds.read(window=w, quality="8-8-2")                # (64, 1024, 1024) int8
b = ds.read(3, window=w, quality="p99<=0.02")         # band 3 only, 1-based
m = ds.read_masks(window=w)
```

`window` takes a rasterio `Window` or `((row0, row1), (col0, col1))` in
source pixels. `quality` takes a rung (`"8-8-2"`, `"8,8,2"` or a
tuple), a bound on a quality statistic (`"mean<=X"` or `"p99<=X"`), or
`None` for full depth. A bound is resolved per window to the cheapest
rung whose encode-time statistic meets it, and to full depth where
none does; it costs one extra read of the quality rows. The statistics
are `1 - cos` of the decoded float field against the prepared source,
over the window's valid pixels, measured with the decode precision the
header records.

The command line:

```bash
python -m georvq.ae.v2 info  tile.georvq.parquet          # header, rungs, bytes per column
python -m georvq.ae.v2 fetch tile.georvq.parquet          # download the release it names
python -m georvq.ae.v2 decode tile.georvq.parquet out.npy \
    --window 0,0,1024,1024 --quality 8-8-2                 # or --quality "p99<=0.02"
python -m georvq.ae.v2 reconstruct tile.georvq.parquet recon.tif --quality 8-8-2
python -m georvq.ae.v2 encode src.tif out_dir/ --bundle <bundle dir> --device cuda
```

`decode` writes a `.npy` in source row order and prints the requests
and bytes it used. `reconstruct` writes a tiled GeoTIFF with the
source's profile, block size, compression, tags and band descriptions;
COG layout and overviews are not reproduced. `encode` writes
`{stem}.georvq.parquet` into the output directory (the current one by
default), and `--no-quality` leaves out the quality column, which
otherwise costs one decode per rung at encode time.

## Rate ladder

The 23 rungs fill one tier at a time: coarse from 2 to 8 levels, then
mid from 1 to 8, then fine from 1 to 8. A level costs 11 bits per
window in the coarse tier, 208 in mid and 3,584 in fine, and every rung
also carries the 440-bit mean. Cosine is on the held-out validation
windows through the training run's own validation harness, with the
coded mean at both ends. "Real tiles" is the int8 reconstruction
against the int8 source over the valid pixels of 16 held-out 8192 px
tiles (2018-2025, latitudes -75 to 75, valid fraction 0.51 to 1.00),
pixel-weighted; at full depth the tiles range from 0.9963 to 0.9995.

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
`earth-compress/georvq-ae-v2` `[pending: hosting]`, without pickled
files: model tensors, codebooks, mean-channel tables, the rung table
and a `bundle.json` holding the ids.

```bash
git clone <this repo> && cd georvq
pip install -e .                    # decode, encode and reconstruct
pip install -e ".[onnx]"            # adds torch-free decode
huggingface-cli download earth-compress/georvq-ae-v2 \
    --local-dir georvq-ae-v2        # [pending: hosting]
```

Encoded tiles will be hosted at `<host>/annual/{year}/{zone}/{stem}.georvq.parquet`
`[pending: hosting]`; `georvq.ae.open` reads them by URL.

## Encoding and decoding cost

The data comes with precomputed quality metrics that are calculated by encoding once and then decoding 
at 23 different bitrates. This means that the time taken for encoding is actually dominated by decodes. The encoding takes about 20 s per tile on an A100, but with quality measures, the process takes 113 s per tile. 

For a user, decoding on one A100 takes 7-9 s for a whole tile at any rung, because the decoder runs at full resolution whatever the depth, and fetching the codes is under 1.3 s of that. A single 1024 px block takes about 0.1 s of GPU time, so reading one over HTTP is dominated by the requests: about 1.6 s in all at the latency and throughput measured from our server to Source Cooperative. On CPU through ONNX Runtime, decoding runs at about 45 windows/s
with 16 threads, about 6 minutes for a full tile.
