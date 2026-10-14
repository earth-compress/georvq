# The georvq GeoParquet layout (draft for review)

**Status: placeholder.** This data standard is a deliberately simple
format, written to ship the AE v2 release. It is expected to be
replaced during the Earth Compress project.

A georvq GeoParquet file holds one source data product compressed by a
georvq codec, as defined by the parent source's most convenient or well-used packet/tile/product of data: one AlphaEarth tile, one Tessera tile, one Sentinel-2
SAFE. It is named `{source stem}.georvq.parquet`. It is a standard
GeoParquet 1.1 file, so pyarrow and DuckDB can read and query its rows directly, and DuckDB's spatial extension reads
`geometry` as a geometry column. 

In practice, we have designed a reader that makes requesting and decoding windows from cloud storage more convenient. For decoding, a reader turns "these
windows, at this rate" into byte ranges using the column-chunk offsets
in the Parquet footer, which one suffix read fetches.


## Chunking, kinds, row groups and file order

The windows of the product are grouped into square chunks, with a
chunk size chosen per kind of record:

- `base`: the base record of each window, which holds the codec's base
  fields (for AE, the transmitted mean) and every level of the coarsest
  tier, because every decode needs them;
- one kind per tier other than the coarsest, named after the tier
  (`mid`, `fine` for a three-tier codec, `tier1` .. `tierN` otherwise;
  a tier name is made of `[A-Za-z0-9_]`, and a reader refuses others);
- `quality`: optional per-window quality bytes, chunked as `base`.

Each kind's chunk size is a whole number of windows, not necessarily a
power of two, but something sensible for the data source, aiming in our design for roughly order-of-magnitude 1 MB per chunk. Chunks are
counted from the product's top-left corner; edge chunks are padded to
the full chunk size with zero records, so every value of a column has
the same length.

Row groups are single rows, so that a single chunk can be fetched. Row groups are in kind order, and within a kind, they are
row-major over that kind's chunk grid. Each kind's row groups are
therefore consecutive in the file, so a read of every chunk of a kind
at full depth is one contiguous byte range, and so is `base` followed
by every chunk of the next tier at full depth.

## Columns

All rows share one schema; a row fills only the columns of its kind and
leaves the others null. Columns, in file order:

| column | type | content |
|---|---|---|
| `kind` | string | `base`, a tier name, or `quality` |
| `product_id` | string | the product's identifier |
| `chunk_row`, `chunk_col` | int32 | the chunk's position in its kind's chunk grid |
| `geometry` | binary (WKB) | the chunk's footprint as a polygon in the product CRS, clipped to the product |
| `bbox` | struct of `xmin`, `ymin`, `xmax`, `ymax` (float64) | the footprint's bounding box (GeoParquet 1.1 covering) |
| `base` | binary | the base record of each window of the chunk |
| `mask` | binary | the chunk's pixel validity, in the base rows directly after `base` |
| `<tier>_1` .. `<tier>_L` | binary | for each tier other than the coarsest, one column per level: that level's record of each window |
| `quality` | binary | per-window quality bytes (optional) |

Inside a code value, the chunk's windows are row-major, each window's
record `record_bytes` long. Within a record, code indices are unsigned,
in cell row-major order, each `bits` wide, MSB-first, padded with zero
bits to a whole byte. A field with `count` 1 is a single value. A level
column holds one level of every window of the chunk, so reading a tier
at depth `L` means reading its first `L` columns, which are contiguous
in the row group. Every decode reads `base`, and the mask follows it,
so the mask costs no request of its own. (The generic layout can place
the mask in a tier's chunks instead, as that kind's first column; the
AE files do not.)

`mask` is empty when every pixel of the chunk is valid. Otherwise it is
the chunk's validity bits (row-major, 1 = valid) packed with
`numpy.packbits` and compressed with zlib, clipped to the product
rather than padded. The header lists the chunks whose mask is not
empty, and a reader does not fetch the others. A window with no valid
pixel is not encoded: its records are zero and a reader fills it with
nodata.

From version 1.1, a chunk none of whose windows has a valid pixel is
empty: every data column of its row is null, so it stores no code
bytes, and the header lists it under its kind's `empty_chunks`. A
reader fetches no empty chunk, nor a chunk of another kind whose
windows in the read all lie in empty chunks, and fills those windows
with nodata. A chunk with some valid windows is stored in full, with
zero records for the others.

## Writer settings

Code columns (`base`, the level columns, `quality`) and `mask` are
written with compression NONE, dictionary encoding off and statistics
off. Statistics are on for `kind`, `product_id`, `chunk_row`,
`chunk_col` and the `bbox` fields, so a query engine can prune row
groups by kind, position and extent. The data page size is set above
the largest value, so each column chunk is one data page holding one
value. The Arrow schema is not stored in the footer
(`store_schema=False`), which keeps the header out of the footer a
second time. Page indexes and bloom filters are not written.

Window regions can be encoded independently and in any order; the
writer collects their packed records and writes all row groups in file
order on close, so the file does not depend on encoding order.

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

- `format`*, `version`*, `status`*: `georvq-parquet`, `1.1`,
  `placeholder`. A reader refuses an unknown major version, and
  ignores header fields it does not know. Version 1.1 added empty
  chunks, whose null values a 1.0 reader does not accept, so a 1.0
  reader cannot read every 1.1 file.
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
  present, including `decode_precision`, the precision of the decodes
  the statistics were measured with.
- `layout`*: `window_px`, `window_grid`, `product_px`, chunk and window
  order (row-major); `groups`, the kinds in file order, each with its
  `chunk_px`, `chunk_windows`, `chunk_grid`, its data `columns` in file
  order, its `row_groups` as `[first, count]` and its `empty_chunks`,
  the positions of its empty chunks; `mask` (column, the
  kind holding it, codec, and `chunks`, the positions of non-empty
  masks); the record layout of each code column (`name`,
  `record_bytes`, `fields` as a list of `{name, count, bits}`); and the
  tiers (`name`, `levels`, and the columns holding them, `["base"]` for
  the coarsest). Tier names are made of `[A-Za-z0-9_]`.

## Reading

A reader fetches the last 256 KiB of the file in one suffix request
(`Range: bytes=-262144`). This holds the footer for every file so far;
a second request is made only when the footer is longer. The footer is
kept for the life of the reader.

A read request is a window region, a depth tuple, and whether the mask
and the quality bytes are needed. For each kind the region touches, the
reader lists the chunks it overlaps and, for each, the columns the
depth needs: `base`; the first `L` level columns of a tier read at
depth `L`; the mask of a chunk when the header lists it as
non-empty; `quality`. Values already in the cache, or inside the bytes
the open fetched, are left out. The remaining column-chunk ranges are
sorted and merged when they touch or are separated by at most a
threshold (16 KiB by default). Merging applies across row groups as
well as within one, which is how a whole-kind read collapses to one
range: between two row groups of a kind lie only the next row group's
position and footprint columns and its null chunks, a few hundred
bytes.

Over HTTP the merged ranges are fetched in parallel, up to 16 requests
at once by default; a local file is read range by range. The fetched
column chunks are parsed as ordinary Parquet pages. Each response's
ETag (or Last-Modified) and the file size in its Content-Range are
compared with the ones the open saw, and a reader refuses to combine
bytes from two versions of a file; a response without a tag is
accepted on its size. A response whose Content-Range is not the range
requested is refused, and so is a server that ignores Range. Each
request times out after 60 s by default, and one that fails with a 5xx
or 429 answer, a timeout or a broken connection is retried, up to 3
times by default with a doubling wait.

### Cache

Fetched values are cached per (chunk, column), not per row group or
byte range, so a later read takes the values it shares with earlier
reads from the cache and fetches only the columns and chunks it lacks.
After a whole-tile read of `base`, every window read in that tile
fetches no base bytes. The key is the file's identity and the
(column, chunk_row, chunk_col) of the value. The identity is the
header's `bitstream_id` together with, for a URL, the URL, the ETag
or Last-Modified of the first response and the file size, and for a
local file, its path, size and modification time, so a changed file
does not reuse cached values. The cache is an in-memory LRU with a byte
budget (512 MB by default) and, optionally, a directory on disk
(`cache_dir/<bitstream_id>/<identity hash>/`), off by default. A URL
whose server sends neither ETag nor Last-Modified is cached in memory
only. A cached value of the wrong length is treated as a miss and
fetched again. Readers can share one cache. The reader counts
requests, bytes fetched, retried requests, and values and bytes served
from the cache.

### Quality column

Per window, one byte per statistic per rung, stat-major (every rung of
the first statistic, then every rung of the next), rungs in `rungs`
order: the window's measured error under that rung's decode at encode
time. Statistics and their encoding are named in the header's `quality`
field; for AE v2, mean and p99 of per-pixel `1 - cos`, as `code =
round(32 * (log10(e) + 6))` clipped to 0..255 (0 means `e <= 1e-6`). A
reader uses it to pick, per window or per region, the cheapest rung
meeting a quality bound. The quality rows are only fetched for a read
that resolves such a bound.

## AE v2 instance

For an 8192 x 8192 AlphaEarth tile (128 x 128 windows of 64 px; tiers
`1:256:2048:8`, `4:192:8192:8`, `16:64:16384:8`), chunk sizes are 8192
px for `base` and `quality` (one chunk for the tile), 4096 px for `mid`
(2 x 2 chunks) and 1024 px for `fine` (8 x 8 chunks), 70 row groups in
all.

| column | fields per window | record bytes | value bytes per chunk |
|---|---|---|---|
| `base` | mean: 2 x 12-bit RVQ + PCA components (416 bits); coarse: 8 levels x 11 bits | 66 | 1,081,344 |
| `mask` | packbits + zlib, empty when fully valid | | 21,529 for the tile below |
| `mid_1` .. `mid_8` | 16 x 13 bits | 26 | 106,496 |
| `fine_1` .. `fine_8` | 256 x 14 bits | 448 | 114,688 |
| `quality` | 2 statistics x 23 rungs x 8 bits | 46 | 753,664 |

Measured on tile `x21yfwngg902u0lsh-0000008192-0000000000` (2021, zone
44N), the row groups of each kind take, with page headers and the
position, footprint and null columns of their rows: `base` 1,104,085
B (the mask included), `mid` 3,412,892 B, `fine` 58,800,768 B,
`quality` 754,875 B. In the published sample file the footer is
141,606 B, for a file of 64,214,230 B (64,167,325 B in the earlier
one-block-per-row layout); the footer's size varies by a few hundred
bytes with the header, which records the source path and the release. The tile's
nodata is a strip along its west edge; as 8 separate 1024 px masks it
took about 4 KB, as one 8192 px mask it takes 21,529 B.

The `base` record holds the mean (55 bytes; fields `mean_rvq`, then
`mean_pca_<k>`) and the 8 coarse levels (11 bytes, level 0 first); it is
always read whole. The 23 rungs are the strict fill-order ladder from
`2-0-0` to `8-8-8`. `georvq_aux` holds the source GeoTIFF's tags,
creation profile and orientation (about 1 KB), so reconstruction writes
a tiled GeoTIFF with the source's block size, compression and band
metadata (COG layout and overviews are not reproduced). `codec` also
records the decode precision used for the quality stats.

### Read cost

This section shows the number of separate requests and the total size in kB for the tile above with a cold cache, read over HTTP
from a local Range server with parallel fetch. The block is one fine chunk (1024 pixels across), and
the point is some subset window inside it. "Previous
layout" is one row group per 1024 px block holding every column, as
measured before this layout replaced it.

| read | this layout | previous layout |
|---|---|---|
| open | 1 / 262 kB | 1 / 262 kB |
| one window (a point) at `8-8-8` | 2 / 2,874 kB | 1 / 988 kB |
| one 1024 px block at `8-8-2` | 2 / 2,186 kB | 1 / 300 kB |
| the same block at `p99<=0.02` | 3 / 3,513 kB | 2 / 885 kB |
| a point at `8-2-0` | 1 / 1,317 kB | |
| whole tile at `8-0-0` | 1 / 1,103 kB | 64 / 1,089 kB |
| whole tile at `8-4-0` | 4 / 2,809 kB | |
| whole tile at `8-8-0` | 1 / 4,516 kB | 64 / 4,509 kB |
| whole tile at `8-8-2` | 64 / 19,202 kB | |
| whole tile at `8-8-8` | 1 / 63,317 kB | 1 / 63,915 kB |

A cold point or block read costs the whole tile's `base` and mask
(1.10 MB), its mid chunk at the requested mid depth, and its fine
chunk. The point and block above take 2 requests because the base row
group is followed directly by the first mid chunk's; elsewhere in the
tile they take 3. At a partial depth each chunk of the partially read
tier is its own range, so `8-4-0` costs one range per mid chunk (the
first merged with base) and `8-8-2` one per fine chunk.

With a warm cache, after a whole-tile read at `8-0-0` (1 / 1,103 kB),
a point at `8-8-8` costs 2 / 1,770 kB, all of it mid and fine bytes,
and the same point again costs nothing.

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
  This was measured on the previous layout and has not been repeated on
  this one.
- A cold point read costs about three times the bytes it did with
  1024 px blocks for every column (2.85 MB against 0.99 MB at `8-8-8`),
  because base and mid come in their larger chunks. The cache recovers
  this for later reads in the same tile.
- GDAL's Parquet driver has not been tried on these files; no GDAL
  build on cirrus includes it.
- Whether to record a checksum per value.
