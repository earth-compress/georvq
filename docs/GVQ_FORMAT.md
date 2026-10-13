# The .gvq format (draft for review)

**Status: placeholder.** Version 1 is a first, deliberately simple
format, written to ship the AE v2 release. It is expected to be
replaced or substantially revised, for example to add entropy coding of
the code indices or to change the chunk layout once real access
patterns are measured. Files carry their major version in the preamble,
so a later reader can refuse or convert a version 1 file, but nothing
about this layout should be treated as a long-term standard.

A `.gvq` file holds one source data product compressed by a georvq
codec: one AlphaEarth tile, one Tessera tile, one Sentinel-2 SAFE. It is
built so that a reader can turn "this window, at this rate" into a few
byte ranges by arithmetic on the header, with no index of chunks to
fetch first.

Every code index has a fixed bit width and every per-window record a
fixed byte length, so the coded sections are uncompressed and their
layout follows from a handful of numbers. Content whose size varies (the
validity mask, the product's own metadata) lives in variable sections at
the end, listed in the header with their offsets.

## File layout

```
offset 0      preamble (16 bytes)
16            header (UTF-8 JSON), padded with spaces
H             fixed sections, each starting on a 4096-byte boundary
...           variable sections, each starting on a 4096-byte boundary
```

`H` is the header region's size, a multiple of 4096 stated in the
preamble. A reader fetches the first 64 KiB, which holds the preamble
and, for every source so far, the whole header.

### Preamble

| bytes | content |
|---|---|
| 0-3 | magic `GVQ\0` |
| 4-5 | major version, uint16 little-endian |
| 6-7 | minor version, uint16 little-endian |
| 8-11 | header JSON length in bytes, uint32 little-endian |
| 12-15 | header region size `H` in bytes, uint32 little-endian |

The first version is 1.0. A reader refuses an unknown major version.
Minor versions only add header fields; a reader ignores fields it does
not know.

## Header

The header is one JSON object. Fields marked * are required.

- `codec`*: `bitstream_id`, the tier specs (`grid:dim:codes:depth`), the
  window size in pixels, and the model release the file was encoded
  with. A decoder refuses a `bitstream_id` it does not hold.
- `product`*: the source product's kind (e.g. `aef-annual-tile`), its
  identifier and source path, CRS (WKT and EPSG where one exists),
  geotransform, width and height in pixels, band count, dtype and
  nodata value, and the orientation the source stores rows in
  (`north-up` or `bottom-up`). Coded data is always north-up; the
  orientation is recorded so a reconstruction matches the source.
- `windows`*: the window grid, `[rows, cols]`, covering the product with
  the last row and column padded where the size is not a multiple of the
  window.
- `fixed_sections`*: a list of section descriptors (below).
- `variable_sections`*: a list of `{name, offset, length, codec,
  sha256}`.
- `rungs`*: the operating points the file supports, each a depth tuple
  with its bits per window and per pixel.
- `quality`: the descriptor of the quality section's encoding, when
  present.

### Fixed section descriptor

| field | meaning |
|---|---|
| `name` | e.g. `base`, `mid`, `fine` |
| `offset` | byte offset of the section |
| `chunk_windows` | `[cy, cx]`, windows per chunk |
| `levels` | number of level slabs per chunk (1 for a window-major section) |
| `fields` | the record layout of one window at one level: a list of `{name, count, bits}` |
| `record_bytes` | bytes of one window's record at one level (fields packed MSB-first, padded to a byte) |

Chunks tile the window grid row-major, and edge chunks are full size
with padding windows (zero bytes) beyond the product. Inside a chunk the
data is level-major: all windows' level 0, then all windows' level 1,
and so on; windows inside a slab are row-major. So, for chunk `(i, j)`,
level `l` and window `(y, x)` within the chunk:

```
chunks_x     = ceil(windows.cols / cx)
chunk_index  = i * chunks_x + j
slab_bytes   = cy * cx * record_bytes
chunk_bytes  = levels * slab_bytes
offset       = section.offset + chunk_index * chunk_bytes
               + l * slab_bytes + (y * cx + x) * record_bytes
```

Levels `0..L-1` of a chunk are one contiguous range: reading at depth L
costs one range per chunk per section.

Within a record, code indices are unsigned, in cell row-major order,
each `bits` wide, MSB-first. A field with `count` 1 is a single value.

### Quality section

Optional, fixed-size, window-major (`levels` 1). Per window, one byte
per statistic per rung: for each rung in `rungs` order, the window's
measured error under that rung's decode at encode time. Statistics and
their encoding are named in the header's `quality` field; for AE v2,
mean and p99 of per-pixel `1 - cos`, as `code = round(32 * (log10(e) +
6))` clipped to 0..255 (0 means `e <= 1e-6`). A reader uses it to pick,
per window or per region, the cheapest rung meeting a quality bound.

### Variable sections

- `mask` (codec `packbits-zlib`): validity at pixel resolution. Square
  blocks of `block_px` (1024 for AE) tile the product's pixel array
  row-major, with edge blocks clipped to the product, not padded. Each
  block's bits (row-major, 1 = valid) are packed with `numpy.packbits`
  and compressed with zlib. The section begins with a uint32
  little-endian block count `n` and a table of `(offset, length)` uint64
  little-endian pairs, offsets relative to the section start (the first
  block's data sits at `4 + 16 n` or later). A block with length 0 is
  fully valid and has offset 0. The section's entry in
  `variable_sections` also carries `block_px` and `shape` (`[height,
  width]` in pixels). A product with no invalid pixels keeps the entry
  with length 0; a file with no `mask` entry reads as fully valid.
- `aux` (codec `zip`): a zip archive of everything the product carries
  that the codec does not code, in the product's own paths. Members
  whose names end in an already-compressed suffix (`.jp2`, `.png`,
  `.jpg`, `.jpeg`, `.zip`, `.gz`, `.bz2`, `.xz`, `.zst`, `.7z`) are
  stored, others deflated. Member timestamps are fixed, so the archive's
  bytes depend only on its content. A reader can fetch one member by
  reading the zip's central directory at the end of the section.

`sha256` covers each variable section's bytes (the digest of empty bytes
for an empty section).

## AE v2 instance

For an 8192 x 8192 AlphaEarth tile (128 x 128 windows of 64 px; tiers
`1:256:2048:8`, `4:192:8192:8`, `16:64:16384:8`):

| section | chunk | levels | fields per window per level | record bytes | chunk bytes |
|---|---|---|---|---|---|
| `base` | 64 x 64 windows (4096 px) | 1 | mean: 2 x 12-bit RVQ + PCA components (416 bits); coarse: 8 x 11 bits | 66 | 270,336 |
| `mid` | 32 x 32 windows (2048 px) | 8 | 16 x 13 bits | 26 | 212,992 |
| `fine` | 16 x 16 windows (1024 px) | 8 | 256 x 14 bits | 448 | 917,504 |
| `quality` | 64 x 64 windows | 1 | 2 statistics x 24 rungs x 8 bits | 48 | 196,608 |

The `base` record holds the mean (55 bytes) and the 8 coarse levels (11
bytes); it is always read whole, so the coarse tier needs no level
order. `aux` holds the source GeoTIFF's tags and creation profile, so
reconstruction writes a file with the source's block size, compression
and band metadata.

A window at depth `8-8-L` costs three ranges: its base chunk, the first 8
slabs of its mid chunk, and the first `L` slabs of its fine chunk. A
whole tile at full depth is about 64 MB plus the mask; at `8-8-0` about
4.5 MB.

## Open points

- Header JSON vs a binary header: JSON is chosen for readability and
  additive versioning; the coded sections are binary.
- Whether to record a per-section checksum for the fixed sections.
