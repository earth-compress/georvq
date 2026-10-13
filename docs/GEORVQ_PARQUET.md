# The georvq GeoParquet layout (draft for review)

**Status: placeholder.** Version 1 is a first, deliberately simple
format, written to ship the AE v2 release. It is expected to be
replaced or substantially revised, for example to add entropy coding of
the code indices or to change the chunk layout once real access
patterns are measured. Files carry their format version in the
`georvq` metadata, so a later reader can refuse or convert a version 1
file, but nothing about this layout should be treated as a long-term
standard.

A georvq GeoParquet file holds one source data product compressed by a
georvq codec: one AlphaEarth tile, one Tessera tile, one Sentinel-2
SAFE. It is named `{source stem}.georvq.parquet`. It is a standard
GeoParquet 1.1 file: pyarrow and DuckDB read its rows directly, and
DuckDB's spatial extension reads `geometry` as a geometry column. For decoding, a reader turns "these windows, at
this rate" into byte ranges using the column-chunk offsets in the
Parquet footer, which one suffix read fetches.

Every code index has a fixed bit width and every per-window record a
fixed byte length, and the code columns are stored uncompressed, so the
size of every value follows from a handful of numbers in the header.

## Rows and columns

One row, and one row group, per square spatial block of the product
(1024 px for AE). Row groups are in row-major block order, so block
`(i, j)` is row group `i * blocks_x + j`. Columns, in file order:

| column | type | content |
|---|---|---|
| `product_id` | string | the product's identifier |
| `block_row`, `block_col` | int32 | the block's position in the block grid |
| `geometry` | binary (WKB) | the block's footprint as a polygon in the product CRS, clipped to the product |
| `bbox` | struct of `xmin`, `ymin`, `xmax`, `ymax` (float64) | the footprint's bounding box (GeoParquet 1.1 covering) |
| `base` | binary | the base record of each window of the block |
| `mask` | binary | the block's pixel validity |
| `<tier>_1` .. `<tier>_L` | binary | for each tier other than the coarsest, one column per level: that level's record of each window |
| `quality` | binary | per-window quality bytes (optional) |

Tier names are `coarse`, `mid`, `fine` for a three-tier codec and
`tier0` .. `tierN` otherwise. The coarsest tier's levels are stored in
`base` together with the codec's base fields (for AE, the transmitted
mean), because every decode needs them. The mask follows `base` so that
a block read at any depth on the fill ladder is one contiguous byte
range: `base`, `mask`, then the level columns in fill order.

Inside a code value, the block's windows are row-major, each window's
record `record_bytes` long. Edge blocks are padded to the full block
size with zero records, so every value of a column has the same length.
Within a record, code indices are unsigned, in cell row-major order,
each `bits` wide, MSB-first, padded with zero bits to a whole byte. A
field with `count` 1 is a single value. A level column holds one level
of every window, so reading a tier at depth `L` means reading its first
`L` columns.

`mask` is empty when every pixel of the block is valid. Otherwise it is
the block's validity bits (row-major, 1 = valid) packed with
`numpy.packbits` and compressed with zlib. Mask blocks are clipped to
the product, not padded. A window with no valid pixel is not encoded:
its records are zero and a reader fills it with nodata.

## Writer settings

Code columns (`base`, the level columns, `quality`) and `mask` are
written with compression NONE, dictionary encoding off and statistics
off. Statistics are on for `product_id`, `block_row`, `block_col` and
the `bbox` fields, so a query engine can prune row groups by position
and extent. The data page size is set above the largest value, so each
column chunk is one data page holding one value. The Arrow schema is
not stored in the footer (`store_schema=False`), which keeps the
header out of the footer a second time. Page indexes and bloom filters
are not written.

Blocks can be encoded independently and in any order; the writer
collects them and writes all row groups in block order on close, so the
file does not depend on encoding order.

## Key-value metadata

- `geo`: GeoParquet 1.1 metadata. `primary_column` is `geometry`,
  encoding WKB, geometry type `Polygon`, `crs` the product CRS as
  PROJJSON (null when unknown), `bbox` the product extent, and
  `covering.bbox` pointing at the `bbox` struct.
- `georvq`*: the header, one JSON object (below).
- `georvq_aux`: the product's own metadata that the codec does not
  code, as a JSON string.

### Header

Fields marked * are required.

- `format`*, `version`*, `status`*: `georvq-parquet`, `1.0`,
  `placeholder`. A reader refuses an unknown major version. Minor
  versions only add fields; a reader ignores fields it does not know.
- `codec`*: `bitstream_id`, `decoder_id`, the tier specs
  (`grid:dim:codes:depth`), the window size in pixels, `min_depths`,
  the mean channel's kind and bits, and the model release the file was
  encoded with. A decoder refuses a `bitstream_id` it does not hold.
- `product`*: the source product's kind (e.g. `aef-annual-tile`), its
  identifier and source path, CRS (WKT and EPSG where one exists),
  north-up geotransform, width and height in pixels, band count, dtype
  and nodata value, and the orientation the source stores rows in
  (`north-up` or `bottom-up`). Coded data is always north-up; the
  orientation is recorded so a reconstruction matches the source.
- `rungs`*: the operating points the file supports, each a depth tuple
  with its bits per window and per pixel.
- `quality`: the descriptor of the quality column's encoding, when
  present.
- `layout`*: `window_px`, `block_px`, `block_windows`, `block_grid`,
  `window_grid`, block and window order (row-major), the mask codec,
  the record layout of each code column (`name`, `record_bytes`,
  `fields` as a list of `{name, count, bits}`), and the tiers (`name`,
  `levels`, and the columns holding them, `["base"]` for the coarsest).

## Reading

A reader fetches the last 256 KiB of the file in one suffix request
(`Range: bytes=-262144`). This holds the footer for every file so far;
a second request is made only when the footer is longer. From the
footer it takes each needed column chunk's offset and length, sorts
the ranges, and merges ranges that touch or are separated by less than
a threshold (16 KiB by default) into one request. The column chunks are
then parsed as ordinary Parquet pages.

For a block read at a rung on the fill ladder, the needed columns are
contiguous and cost one request. A quality-bound read fetches the
`quality` column of the blocks first and then the codes, two requests
per block. A whole-product read at full depth merges into one request,
because the gaps between row groups (the next block's position and
footprint columns, and `quality` when it is not read) are below the
threshold. A whole-product read at the base depth costs one request per
block, since each block's level columns lie between consecutive `base`
values.

### Quality column

Per window, one byte per statistic per rung, stat-major (every rung of
the first statistic, then every rung of the next), rungs in `rungs`
order: the window's measured error under that rung's decode at encode
time. Statistics and their encoding are named in the header's `quality`
field; for AE v2, mean and p99 of per-pixel `1 - cos`, as `code =
round(32 * (log10(e) + 6))` clipped to 0..255 (0 means `e <= 1e-6`). A
reader uses it to pick, per window or per region, the cheapest rung
meeting a quality bound.

## AE v2 instance

For an 8192 x 8192 AlphaEarth tile (8 x 8 blocks of 1024 px, 16 x 16
windows of 64 px per block; tiers `1:256:2048:8`, `4:192:8192:8`,
`16:64:16384:8`):

| column | fields per window | record bytes | value bytes per block | bytes per tile |
|---|---|---|---|---|
| `base` | mean: 2 x 12-bit RVQ + PCA components (416 bits); coarse: 8 levels x 11 bits | 66 | 16,896 | 1,083,072 |
| `mask` | packbits + zlib, empty when fully valid | | 0 to a few KB | 5,560 |
| `mid_1` .. `mid_8` | 16 x 13 bits | 26 | 6,656 | 3,420,672 (all 8) |
| `fine_1` .. `fine_8` | 256 x 14 bits | 448 | 114,688 | 58,734,080 (all 8) |
| `quality` | 2 statistics x 23 rungs x 8 bits | 46 | 11,776 | 755,392 |

Bytes per tile are column-chunk sizes, page headers included, measured
on tile `x21yfwngg902u0lsh-0000008192-0000000000` (2021, zone 44N). The
position and footprint columns add 43,264 B and the footer 125,281 B,
for a file of 64,167,325 B (the same codes in the earlier `.gvq`
container took 64,040,960 B).

The `base` record holds the mean (55 bytes; fields `mean_rvq`, then
`mean_pca_<k>`) and the 8 coarse levels (11 bytes, level 0 first); it is
always read whole. The 23 rungs are the strict fill-order ladder from
`2-0-0` to `8-8-8`. `georvq_aux` holds the source GeoTIFF's tags,
creation profile and orientation (about 1 KB), so reconstruction writes
a tiled GeoTIFF with the source's block size, compression and band
metadata (COG layout and overviews are not reproduced). `codec` also
records the decode precision used for the quality stats.

### Read cost

Requests and bytes for the tile above, read over HTTP from a local
Range server.
"Prototype" is the measurement that chose this layout (mask stored last
and skipped by footer statistics, 64 KiB first read); "reader" is
`georvq.inference.georvq_parquet` on the file as specified here. The
`.gvq` column is the earlier container's reader, for reference.

| read | reader | prototype | `.gvq` |
|---|---|---|---|
| open | 1 / 262,144 B | 2 / 146,076 B | 1 / 65,536 B |
| one 1024 px block at `8-8-2` | 1 / 299,824 B | 1 / 299,867 B | 5 / 713,732 B |
| the same block at `p99<=0.02` | 2 / 885,202 B | | 4 / 1,474,560 B |
| one window (a point) at `8-8-8` | 1 / 988,114 B | 1 / 988,193 B | 5 / 1,401,860 B |
| whole tile at `8-0-0` | 64 / 1,088,632 B | 64 / 1,098,826 B | 14 / 1,086,444 B |
| whole tile at `8-8-0` | 64 / 4,509,304 B | 64 / 4,522,570 B | 30 / 4,494,316 B |
| whole tile at `8-8-8` | 1 / 63,914,846 B | 1 / 64,044,169 B | 94 / 63,214,572 B |

A point costs its whole block at the requested depth, because a block's
level is one value. Reading a local file costs the same reads.

## Open points

- Large aux. `georvq_aux` sits in the footer, which every reader
  fetches on open. That suits the AE source profile (about 1 KB), but a
  Sentinel-2 SAFE's metadata runs to several MB and will need a
  different home, for example a sidecar file or dedicated rows. Not
  built yet.
- File bytes depend on the pyarrow version. In the prototype, files
  written by pyarrow 18.1 to 23.0.1 from the same blocks had identical
  code column chunks, but the statistics columns and the footer
  differed between versions (footer 126,652 B for 18-19, 145,156 B for
  20-21, 146,068 B for 22-23). Decoded values do not change. A file is
  reproducible byte for byte only under one pyarrow version, and a
  checksum of a release file is a checksum of that writer's output.
- Whole-product reads at low depth cost one request per block. A
  layout with all blocks' `base` values adjacent (base row groups first,
  or one row per tier level) would make them one request, at the cost of
  more ranges for a single block.
- GDAL's Parquet driver has not been tried on these files; no GDAL
  build on cirrus includes it.
- Whether to record a checksum per value.
