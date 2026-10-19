# The georvq Zarr layout

*Draft for review.*

A georvq store holds one source data product compressed by a georvq
codec, such as one AlphaEarth tile, as a Zarr v3 group named
`{source stem}.georvq.zarr`. It is plain Zarr, written by zarr-python
with its own defaults, so zarr-python and xarray open it without
georvq, and every array decodes with the standard `bytes`, `zstd` and
`sharding_indexed` codecs. What those readers get back are packed code
records, which mean nothing until they are unpacked and decoded, so
`georvq.open` (`docs/AE_RELEASE.md`) is the way to turn a store into
pixels. It accepts a path or URL, or a `zarr.Group` that a user has
already opened, for example on another storage backend or as a
rechunked copy. This document specifies the layout for anyone writing
their own reader or writer.

## Arrays and chunks

The windows of the product are grouped into square chunks, with a chunk
size chosen per kind of record, and each kind is one array over the
window grid:

| array | shape | chunks |
|---|---|---|
| `base` | (window rows, window cols, base bytes) | one chunk per base chunk |
| one per tier after the coarsest (`mid`, `fine`) | (window rows, window cols, levels, record bytes) | one shard for the whole array, with an inner chunk per level of each tier chunk |
| `mask` | (pixel rows, pixel cols), bool | one chunk per base chunk, zstd |
| `quality/<decoder_id>` | (window rows, window cols, quality bytes) | as `base` |

`base` holds the base record of each window, which carries the codec's
base fields (for AE, the transmitted mean) and every level of the
coarsest tier, because every decode needs them. Each tier array has a
level dimension, and each level of a tier chunk is its own inner chunk,
so reading a tier to depth `L` reads `L` inner chunks per tier chunk and
leaves the deeper levels untouched. zarr-python writes the inner chunks
of a shard in Morton order and the shard index at its end, and a reader
locates them through the index, so the order does not matter. From
zarr-python 3.3 a partial read of a shard merges neighbouring inner
chunks and fetches them concurrently; earlier versions fetch them one at
a time, which on Hugging Face made a whole-tile read about ten times
slower. The code arrays have no compressor, because the records are
already packed to their bit widths, and the edge chunks are padded with
zero records.

## Records

A record is one window's values for one column, packed MSB-first with
the widths of its fields and padded to a whole byte
(`georvq/inference/bitpack.py`). The fields of each column are listed in
the header (below), so a reader unpacks the uint8 bytes of an array with
those widths. For AE v2, a fine level of one window is 256 codes of 14
bits, 448 bytes.

## Mask and empty chunks

The pixel validity mask is stored once, losslessly, as a bool array
compressed with zstd, and a chunk with every pixel valid is not stored
(the fill value is True). A chunk of any kind none of whose windows has
a valid pixel is empty. The header lists it, its records are zero, and
the tier arrays do not store it, so a partly valid tile is smaller. A
reader does not read an empty chunk, nor a chunk of another kind whose
windows in the read all lie in empty chunks.

## Quality per decoder

The quality bytes give each window's measured error at every rung, as
one byte per statistic per rung, stat-major, in the order of that
decoder's rungs. These numbers belong to the decoder they were measured
with, because a retrained decoder of the same codes has other errors and
possibly other rungs. Each quality array is therefore named by the
`decoder_id` it was measured with (a hash of the decoder's weights,
`georvq/inference/release.py`), and its attributes (`georvq_quality`)
carry that decoder's `decoder_id`, `bitstream_id` and release, its
rungs, the fields of its record and the definition of the error. The
`quality` group's attributes list the decoders it holds under
`decoders`, so a reader finds them without listing the store. A later
decoder of the same bitstream adds its array and its entry in that
list, and leaves the existing arrays as they are. A reader serves the
quality of the decoder the header names until it is asked for another,
and refuses a quality array that names another bitstream.

For AE v2 the statistics are the mean and p99 of per-pixel `1 - cos`,
stored as `code = round(32 * (log10(e) + 6))` clipped to 0..255 (0
means `e <= 1e-6`).

## Header

The group's attributes hold the codec header as JSON under `georvq`:
`format` (`georvq-zarr`), `version`, `status`, `codec` (its format,
`bitstream_id`, `decoder_id`, tiers, window and release), `product`
(the source's id, path, CRS, north-up transform, size and orientation),
`rungs`, `quality` for the header's own decoder, `output` (the data
type, nodata value, band count and band names a reader returns), and
`layout`. The
layout gives the window size, the window grid and product size, each
kind's chunk size, chunk grid and empty chunks, the mask chunks that
hold an invalid pixel, the record fields of every column, and the
tiers. The source's profile, tags and band descriptions are JSON under
`georvq_aux`. The group and its arrays also carry `proj:` and `spatial:`
attributes of the GeoZarr conventions, with the pixel transform on the
mask and the window transform on the code arrays. The writer adds
zarr-python's consolidated metadata, so a reader opens a store with one
request for the group's `zarr.json`.

## Reading

georvq's reader (`georvq/inference/records.py`) is built on an open
`zarr.Group` and leaves all requests to zarr-python. For a window region
and a depth tuple it lists the chunks of each kind the region overlaps,
leaves out the ones in its cache and those ruled out by empty chunks,
and makes one zarr-python read per array over the bounding box of the
rest, every level down to the deepest needed, with the reads of
different arrays running in parallel. The values are cached per
(column, chunk) in memory, so later reads take what they share with
earlier ones from the cache. Over HTTP, `storage_options` go to fsspec,
for example a token for a private repository.

## AE v2 instance

For an 8192 x 8192 AlphaEarth tile (128 x 128 windows of 64 px, with
tiers `1:256:2048:8`, `4:192:8192:8`, `16:64:16384:8`), the chunk sizes
are 8192 px for `base`, `mid`, `quality` and the mask (one chunk for
the tile) and 2048 px for `fine` (4 x 4 chunks). The 2048 px fine chunk
was chosen over the source's 1024 px block because each request to the
Hugging Face CDN costs 0.55 to 0.65 s whatever its size up to a few MB,
so a read covering many fine chunks is limited by the number of
requests rather than by bytes. With 2048 px chunks a whole-tile read at
`8-8-2` took about half as long as at `8-8-8`, where with 1024 px chunks
the two took about the same. A fully valid tile is a 64.0 MB store of 12
objects, of which `fine` takes 58.7 MB.

## Open points

The store is a prototype (`status`). A store per UTM zone, rather than
per tile, would let one open serve a whole zone, and the AEF tiles of a
zone-year share one 81,920 m grid, so such a store could be assembled
from the per-tile records without re-encoding. It has not been built,
because a shard of a whole zone's fine tier would be up to 43 GB and
would need a writer that streams it. Adding a decoder's quality to a
published store also has no tool yet.
