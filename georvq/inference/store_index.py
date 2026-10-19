"""Facts about georvq stores that a release's tile index records.

A release's index has one row per store, joined to the source
collection's own index by the release layer (georvq.ae.index matches
source.coop's `aef_index.parquet`, georvq.tessera.v2 geotessera's
registry). What a row says about the store itself is the same for every
release and is computed here from its metadata and stored sizes:

- `bitstream_id`, `decoder_id`: from the store's codec header;
- `store_bytes`: the bytes stored by the store's arrays, each array's
  zarr.json included (zarr-python's `nbytes_stored`);
- `metadata_bytes`: the group zarr.json, which a reader fetches on open
  and which holds the consolidated metadata of the arrays;
- `quality_bytes`: the stored bytes of the store's quality arrays, 0
  when it has none;
- `rungs`: per rung of the header's rung table, in order, `rung`
  ("8-8-2"), the header's nominal `bits_px`, and `bytes`, the bytes a
  whole-product read at that rung reads: the records of base and of
  each tier's levels up to the rung's depth, outside empty chunks, and
  the mask array's stored bytes; quality excluded.

Only metadata and stored sizes are read, so the stores must be on a
filesystem zarr-python can list, such as a local directory.
"""
from __future__ import annotations

import os

from georvq.inference.records import SUFFIX, RecordReader


def rung_bytes(reader: RecordReader, mask_bytes: int) -> list:
    """Per header rung: {rung, bits_px, bytes}, with bytes the record
    bytes a whole-product read at the rung reads."""
    col = reader.column_bytes()
    out = []
    for r in reader.header["rungs"]:
        t = [int(v) for v in r["tuple"]]
        cols = reader.columns_for(t, mask=False)
        out.append({"rung": "-".join(map(str, t)),
                    "bits_px": float(r["bits_px"]),
                    "bytes": int(sum(col.get(c, 0) for c in cols) + mask_bytes)})
    return out


def tile_record(path, **reader_kw) -> dict:
    """Metadata facts of one coded store: product id, codec ids, sizes and
    rung bytes. `reader_kw` go to RecordReader."""
    rd = RecordReader(path, **reader_kw)
    h = rd.header
    codec = h.get("codec") or {}
    sizes = rd.stored_bytes()
    meta = os.path.join(os.fspath(path), "zarr.json")
    return {"product_id": str(h["product"]["id"]),
            "bitstream_id": codec.get("bitstream_id"),
            "decoder_id": codec.get("decoder_id"),
            "store_bytes": int(sum(sizes.values())),
            "metadata_bytes": int(os.path.getsize(meta)),
            "quality_bytes": int(sum(b for k, b in sizes.items()
                                     if k.startswith("quality/"))),
            "rungs": rung_bytes(rd, sizes.get("mask", 0))}


def find_stores(root) -> list:
    """`*.georvq.zarr` stores under a local directory, sorted."""
    out = []
    for d, names, _ in os.walk(os.fspath(root)):
        hits = [n for n in names if n.endswith(SUFFIX)]
        out += [os.path.join(d, n) for n in hits]
        names[:] = [n for n in names if not n.endswith(SUFFIX)]
    return sorted(out)


def stem(p: str) -> str:
    """The product stem of a store or source path."""
    name = p.rstrip("/").rsplit("/", 1)[-1]
    return name[:-len(SUFFIX)] if name.endswith(SUFFIX) else os.path.splitext(name)[0]
