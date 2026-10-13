Draft for review: wording not yet approved by the maintainer; `[pending: ...]` marks values still being measured or built.

# The AlphaEarth v2 release codec

The second release codec for the AlphaEarth Foundations annual
embedding tiles. Each source tile becomes one GeoParquet file that can
be read in place of the source tile, at any of 23 rates, with windowed
range reads over HTTP. The release policy lives in `georvq/ae/v2.py`;
the container is the source-agnostic `georvq/inference/georvq_parquet.py`
and the mean coder is `RVQPCAMeanChannel` in
`georvq/inference/mean_codec.py`. The v1 release (`ae-beta-v1`,
`docs/AE_RELEASE.md`) is separate and unchanged.

## Who is this for

| you are | you touch |
|---|---|
| AE data user, reading tiles | `georvq.ae.open`, `python -m georvq.ae.v2 decode/reconstruct/info`, the Reading and Rate sections below |
| archive operator, bulk-encoding source tiles | `python -m georvq.ae.v2 encode` on a GPU, the quality column, the Encoding cost section |
| codec researcher | `georvq/{model,train,...}` and `docs/OVERVIEW.md`; `georvq/ae/v2.py` is a worked example of a release layer on the GeoParquet container |

## The release

- Model: a `stats_cond` model with 8-bit (int8) codebooks over the same
  geometry as v1: window 64 px, 64 bands, tiers `1:256:2048:8` /
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
- `bitstream_id`: `[pending: bundle]`. A file records the
  `bitstream_id` and `decoder_id` it was encoded with, and a reader
  holding a different codec refuses to decode it.
- Backends: PyTorch, CPU or CUDA. Decoder passes run in bf16 on CUDA by
  default (`--fp32` or `bf16=False` turns this off) and in fp32 on CPU;
  encoder passes always run in fp32. Torch-free decode through ONNX
  Runtime is planned `[pending: onnx]`.

## What changed from v1

| | ae-beta-v1 | AE v2 |
|---|---|---|
| model | `ae_pyr64_divfull` | `stats_cond` model with int8 codebooks, same tier geometry |
| mean | not sent separately | transmitted, 440 bits per window |
| container | a directory of COG assets per tile | one `{stem}.georvq.parquet` file per tile |
| rate control | `--tuple L0,L1,L2` on the efficient ladder | one of 23 strict fill-order rungs, `2-0-0` to `8-8-8` |
| quality control | `--quality STAT<=X`, STAT in mean, p50, p90, p99, max of pixel dcos | `quality="STAT<=X"`, STAT in mean, p99 of per-pixel `1 - cos` |
| refinement and decoders | `star` sidecar, `plain` and `deblock` decoders | none; one decoder |
| torch-free decode | ONNX | `[pending: onnx]` |
| entry points | `georvq-ae-tile`, `georvq-ae-export-onnx` | `georvq.ae.open`, `python -m georvq.ae.v2` |

v1 users keep everything they have. The `ae-beta-v1` bundle, its COG
assets, `georvq-ae-tile`, the ONNX decoder and `tests/test_compat.py`
are untouched by v2. The two releases have different bitstream ids and
neither decodes the other's files; a tile encoded with v1 stays a v1
tile until it is re-encoded from the source.

## Format: one tile = one GeoParquet file

One file per source tile, named `{source stem}.georvq.parquet` and
placed at the path that mirrors the source tile's path on source.coop
(`annual/{year}/{zone}/{stem}.georvq.parquet`). A tile index matching
the source index is planned `[pending: index]`. The files are
GeoParquet 1.1, so pyarrow and DuckDB read their rows directly; the
georvq header is JSON in the file's key-value metadata, and the source
tile's profile, tags, band descriptions and orientation are kept as
JSON in `georvq_aux`. The full layout, as a draft spec with placeholder
status, is in `docs/GEORVQ_PARQUET.md`.

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

## Reading

`georvq.ae.open(path_or_url, ...)` returns a read-only dataset with the
source tile's georeferencing and shape: `crs`, `transform`, `width`,
`height`, `count` (64), `dtypes` (int8), `nodata` (-128), `profile`,
`tags()`, `descriptions`, `block_shapes` and `block_windows()`. The
AEF source tiles store their rows bottom-up; the file codes them
north-up and `read` returns them in the source's own order, so code
written against the rasterio dataset of the source tile can read the
file instead. Invalid pixels come back as -128 exactly, from the
stored mask. The header and metadata need no model; reading pixels
does.

```python
import georvq.ae

ds = georvq.ae.open("annual/2021/44N/<stem>.georvq.parquet",
                    model="<checkpoint>", channel="<mean channel .npz>",
                    device="cuda")                     # [pending: bundle]
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
header records. `open` also accepts `codec=` (an `AEV2Codec`) in place
of `model` and `channel`, and loading the release bundle through a
loader function is `[pending: bundle]`. A codec whose `bitstream_id`
differs from the file's is refused.

The reader opens a file with one suffix request of 256 KiB, which holds
the footer and header. For each read it works out the column chunks
the windows and depths need, merges ranges separated by at most 16 KiB
(across row groups as well as within one), and over HTTP fetches up to
16 ranges in parallel. Fetched values are cached per (column, chunk) in
an in-memory LRU of 512 MB by default, with an optional directory
cache, so a later read in the same tile fetches only what it lacks.
These are the `RVQParquetReader` arguments `first_read`, `gap`,
`concurrency`, `cache`, `cache_bytes` and `cache_dir`, which `open`
passes through. A file that changes between requests (a different ETag
or Last-Modified) is refused rather than mixed.

The command line:

```bash
python -m georvq.ae.v2 info  tile.georvq.parquet          # header, rungs, bytes per column
python -m georvq.ae.v2 decode tile.georvq.parquet out.npy \
    --model <checkpoint> --channel <mean channel .npz> \
    --window 0,0,1024,1024 --quality 8-8-2                 # or --quality "p99<=0.02"
python -m georvq.ae.v2 reconstruct tile.georvq.parquet recon.tif \
    --model <checkpoint> --channel <mean channel .npz> --quality 8-8-2
python -m georvq.ae.v2 encode src.tif out_dir/ \
    --model <checkpoint> --channel <mean channel .npz> --device cuda
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
coded mean at both ends.

| rung | bits/window | b/px | val cos | real tiles |
|---|---|---|---|---|
| 2-0-0 | 462 | 0.113 | 0.9722 | `[pending: measurements]` |
| 8-0-0 | 528 | 0.129 | 0.9798 | `[pending: measurements]` |
| 8-1-0 | 736 | 0.180 | 0.9858 | `[pending: measurements]` |
| 8-2-0 | 944 | 0.230 | 0.9886 | `[pending: measurements]` |
| 8-4-0 | 1,360 | 0.332 | 0.9912 | `[pending: measurements]` |
| 8-8-0 | 2,192 | 0.535 | 0.9932 | `[pending: measurements]` |
| 8-8-1 | 5,776 | 1.410 | 0.9951 | `[pending: measurements]` |
| 8-8-2 | 9,360 | 2.285 | 0.9962 | `[pending: measurements]` |
| 8-8-4 | 16,528 | 4.035 | 0.9973 | `[pending: measurements]` |
| 8-8-8 | 30,864 | 7.535 | 0.9981 | `[pending: measurements]` |

The header's `rungs` field lists all 23 with their bits per window and
per pixel, and `info` prints them.

## Quality

Against ae-beta-v1 at matched bits, on the same validation windows.
This comparison was run with the mean sent at 6 bits per band (384
bits, charged); the released 440-bit channel adds 0.014 b/px to every
rung and changes cosine by about 0.0001.

| rung | b/px | v2 | ae-beta-v1 |
|---|---|---|---|
| 2-0-0 | 0.099 | 0.9717 | 0.9810 |
| 8-1-0 | 0.166 | 0.9852 | 0.9855 |
| 8-5-0 | 0.369 | 0.9914 | 0.9895 |
| 8-8-0 | 0.521 | 0.9927 | 0.9905 |
| 8-8-8 | 7.52 | 0.9976 | 0.9963 |

Above about 0.2 b/px v2 has 7-36% less error (`1 - cos`) than
ae-beta-v1; below it v2 is worse. The comparison over many real tiles is
`[pending: measurements]`.

On one real tile (2021, zone 44N, 11% nodata), cosine between the
decoded int8 tile and the int8 source was 0.9704 at `8-0-0`, 0.9887 at
`8-8-0`, 0.9944 at `8-8-2` and 0.9973 at `8-8-8`, with the bf16
decoder. The full-depth figure sits below the validation 0.9981, which
was measured on float data with an fp32 decoder; how much of the gap
comes from requantization to int8, from bf16 and from tile-to-tile
spread is being measured `[pending: measurements]`.

## Access costs

Requests and kB (1000 B) for the tile above with a cold cache, read
over HTTP from a local Range server with parallel fetch (a local file
costs the same requests and bytes). The block is one 1024 px fine
chunk and the point a window inside it, both in the first mid chunk.

| read | requests / kB |
|---|---|
| open | 1 / 262 |
| one window (a point) at `8-8-8` | 2 / 2,874 |
| one 1024 px block at `8-8-2` | 2 / 2,186 |
| the same block at `p99<=0.02` | 3 / 3,513 |
| a point at `8-2-0` | 1 / 1,317 |
| whole tile at `8-0-0` | 1 / 1,103 |
| whole tile at `8-4-0` | 4 / 2,809 |
| whole tile at `8-8-0` | 1 / 4,516 |
| whole tile at `8-8-2` | 64 / 19,202 |
| whole tile at `8-8-8` | 1 / 63,317 |

A cold point or block read costs the whole tile's `base` and mask
(1.10 MB), its mid chunk at the requested mid depth and its fine
chunk. Elsewhere in the tile the point and block take 3 requests
rather than 2, because only the first mid chunk directly follows the
base row group. A partially read tier costs one range per chunk, so
`8-4-0` takes one per mid chunk and `8-8-2` one per fine chunk. After
a whole-tile read at `8-0-0`, a point at `8-8-8` costs 2 requests and
1,770 kB, all of it mid and fine, and the same point again costs
nothing.

## Getting the model

Weights are not in git. They will be published on Hugging Face as
`earth-compress/georvq-ae-v2` `[pending: hosting]`, without pickled
files: model tensors, codebooks, mean-channel tables, the rung table
and a `bundle.json` holding the ids `[pending: bundle]`.

```bash
git clone <this repo> && cd georvq
pip install -e ".[eval]"            # rasterio for encode and reconstruct
huggingface-cli download earth-compress/georvq-ae-v2 \
    --local-dir georvq-ae-v2        # [pending: hosting]
```

Encoded tiles will be hosted at `<host>/annual/{year}/{zone}/{stem}.georvq.parquet`
`[pending: hosting]`; `georvq.ae.open` reads them by URL.

## Encoding cost

Encoding the tile above with quality stats took 101 s on one A100,
and a full-tile decode took 8-10 s in the same run. Throughput over many tiles, on GPU and
on CPU, is `[pending: measurements]`.

## Deliberate non-features

The choices below follow a tier ablation at full depth with the exact
mean, quantization removed from one tier at a time: removing it from
coarse closes 0% of the gap to the unquantized ceiling, from mid -22%,
from fine 68%.

- No `star`: with the coarse tier contributing nothing to the gap, a
  coarse-residual sidecar has nothing to recover.
- No ceiling sidecar: a 1-bit fine refinement would cost 53% more rate
  for at most 0.0003 cosine, which is worse per bit than the ladder's
  own last step.
- No deblock decoder: v1's deblocker is tied to v1's bitstream and base
  decoder and cannot be used with v2, and none has been trained for
  v2.
- No entropy coding of the indices, as in v1; the container is
  uncompressed so that every offset follows from the header.

## Open points

- The container is a placeholder (version 1.0, status `placeholder` in
  the header) and may be replaced, for example to entropy-code the
  indices or change the chunk layout once real access patterns are
  measured.
- `georvq_aux` lives in the footer, which every reader fetches on open.
  That suits the AE source profile (about 1 KB) but not a source with
  several MB of product metadata, such as a Sentinel-2 SAFE.
- File bytes depend on the pyarrow version that wrote them (footer and
  statistics columns; the code column chunks and decoded values do not
  change), so a checksum of a release file holds only for one writer
  version.
- GDAL's Parquet driver has not been tried on these files.
- A cold point read costs about three times the bytes it would with
  1024 px chunks for every kind (2.87 MB against 0.99 MB at `8-8-8`);
  the cache recovers this for later reads in the same tile.
