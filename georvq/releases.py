"""The releases georvq can read, and `georvq.open`.

A store's header names its codec format (`codec.format`). `RELEASES`
maps each format georvq knows to the module that holds its release
layer, whose `SPEC` is a georvq.inference.store_codec.ReleaseSpec. The
table is fixed here: a store only selects one of these entries by its
format, and no module or attribute name is ever taken from a store.

`open(store)` reads the store's consolidated metadata (one request),
picks the release from the table, and returns a
georvq.inference.store_dataset.StoreDataset whose data type, nodata,
band names, rungs, quality statistics and georeferencing come from the
store. The codec is found when pixels are first read, from the decoder
bundle the store's header pins (georvq.inference.bundle_resolve:
environment, local cache, or a download from a trusted repository at
the recorded revision), and cached.
"""
from __future__ import annotations

import importlib
import os

RELEASES = {
    "georvq-ae/v2": "georvq.ae.v2",
    "georvq-tessera/v2": "georvq.tessera.v2",
    "georvq-s2/v1": "georvq.s2.v1",
}


def spec_for(codec_format):
    """The ReleaseSpec of a codec format named in a store header."""
    mod = RELEASES.get(codec_format)
    if mod is None:
        raise ValueError(f"codec format {codec_format!r} is not one georvq "
                         f"reads; it knows {sorted(RELEASES)}")
    return importlib.import_module(mod).SPEC


def open(store, *, codec=None, bundle=None, backend=None, device=None,
         quality=None, bf16=None, threads=None, cache_dir=None,
         trust_remote=False, force=False, storage_options=None, cache=None,
         cache_bytes=None, trace=None, grid=None, **options):
    """Open a georvq store as a rasterio-like read-only dataset.

    `store` is a local path, a URL zarr-python opens through fsspec
    (https://, s3://, hf://, ...), or an open zarr.Group. Nothing else
    is needed: the release, the output data type and nodata, the band
    names, the rungs and quality statistics and the CRS come from the
    store, and the decoder bundle it pins is located and loaded when
    pixels are first read.

    Keywords for special cases:
    - `codec`: a codec to decode with, or `bundle`: a bundle directory to
      load one from. Either must be for the store's bitstream; one whose
      decoder differs from the decoder the store was encoded with is
      refused unless `force=True` (the store's quality values belong to
      its own decoder, see docs/GEORVQ_ZARR.md).
    - `backend`: "torch" (default when it imports) or "onnx"; `device`
      (default cuda when torch sees a GPU, else cpu); `bf16` (decoder
      passes in bf16 on CUDA, default on); `threads` (onnxruntime).
    - `quality`: the default of every `read` (a rung such as "8-8-2", or
      a bound such as "p99<=0.02"; None reads full depth).
    - `cache_dir`, `trust_remote`: where bundles are cached, and whether
      a bundle may be downloaded from a repository outside the trusted
      patterns.
    - `storage_options` (for fsspec), `cache`, `cache_bytes`, `trace`:
      as for georvq.inference.records.RecordReader.
    - `grid`: for a store whose bands lie on several grids (S2's 10, 20
      and 60 m), the grid the dataset shows (default the first); the
      others are reached with `ds.grid(name)`, and `ds.read_grids`
      decodes them all at once.
    - any further keyword is an output option of the store's release
      (its `output_options`, for example Tessera's `units`)."""
    from georvq.inference.records import (HEADER_KEY, _is_group,
                                          check_header, open_store)
    from georvq.inference.store_dataset import StoreDataset
    if _is_group(store):
        if storage_options is not None:
            raise ValueError("storage_options apply to a path or URL, not "
                             "an open zarr.Group")
        group = store
    else:
        group = open_store(os.fspath(store), storage_options)
    attrs = dict(group.attrs)
    if HEADER_KEY not in attrs:
        raise ValueError(f"{store}: no georvq header")
    header = attrs[HEADER_KEY]
    check_header(header, str(store))
    spec = spec_for((header.get("codec") or {}).get("format"))
    want_did = header["codec"].get("decoder_id")

    def checked(c):
        # a decoder-only variant of the store's decoder is accepted
        parent = (getattr(c, "bundle", None) or {}).get("parent_decoder_id")
        if not force and want_did and want_did not in (
                getattr(c, "decoder_id", None), parent):
            raise ValueError(
                f"{store}: encoded with decoder {want_did}, the codec given "
                f"holds decoder {c.decoder_id}; pass force=True to decode "
                "with it (quality bounds then need quality values for that "
                "decoder)")
        return c

    def resolver(h):
        from georvq.inference.bundle_resolve import resolve_codec
        return checked(resolve_codec(
            h, spec, bundle=bundle, backend=backend, device=device,
            bf16=bf16, threads=threads, cache_dir=cache_dir,
            trust_remote=trust_remote))

    reader_kw = {"trace": trace}
    if cache is not None:
        reader_kw["cache"] = cache
    if cache_bytes is not None:
        reader_kw["cache_bytes"] = cache_bytes
    ds = StoreDataset(group, spec, codec=None if codec is None else
                      checked(codec), resolver=None if codec is not None
                      else resolver, quality=quality, options=options,
                      grid=grid, **reader_kw)
    if not _is_group(store):
        ds.reader.source = ds.name = os.fspath(store)
    if bundle is not None and codec is None:
        try:
            ds.codec
        except BaseException:
            ds.close()
            raise
    return ds
