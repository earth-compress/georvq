"""Finding the bundle that decodes an AE v2 store.

georvq.inference.bundle_resolve does the work for every release; this
module binds it to AE v2 (`georvq.ae.v2.AE_V2`): bundles are found in
turn at $GEORVQ_AE_BUNDLE, in the bundle cache
(`<root>/bundles/<bitstream_id>/`, root `cache_dir`, else $GEORVQ_CACHE,
else $XDG_CACHE_HOME/georvq, else ~/.cache/georvq), or by downloading
the release the header names when it is under `hf://earth-compress/` or
the caller passes `trust_remote=True`. A bundle for another bitstream id
is refused. See that module for the rules.

This module imports neither torch nor huggingface_hub until it needs
them.
"""
from __future__ import annotations

from georvq.ae.v2 import (AE_V2, BUNDLE_FORMAT, ENV_BUNDLE,  # noqa: F401
                          RELEASE_NAME, RELEASE_URI, TRUSTED_RELEASES)
from georvq.inference import bundle_resolve as _br
from georvq.inference.bundle_resolve import (ENV_CACHE, ONNX_DIR,  # noqa: F401
                                             BundleError, bundle_path,
                                             cache_root, check_files,
                                             default_backend, default_device,
                                             file_bitstream_id, file_release,
                                             parse_hf_uri, sha256_file)
from georvq.inference.store_codec import release_record as _release_record


def release_record(codec, name=RELEASE_NAME, uri=RELEASE_URI,
                   revision=None) -> dict | None:
    """The header's `codec.release` for a store coded by `codec`; None
    when `name` is None."""
    return _release_record(codec, name, uri, revision)


def read_bundle_meta(bundle_dir) -> dict:
    return _br.read_bundle_meta(bundle_dir, AE_V2)


def check_bundle(bundle_dir, bitstream_id, found_by) -> dict:
    return _br.check_bundle(bundle_dir, AE_V2, bitstream_id, found_by)


def is_trusted(uri, trusted=TRUSTED_RELEASES) -> bool:
    return _br.is_trusted(uri, trusted)


def download_bundle(uri, revision, bitstream_id, cache_dir=None,
                    full=False) -> str:
    return _br.download_bundle(uri, revision, bitstream_id, AE_V2, cache_dir,
                               full=full)


def locate_bundle(header, cache_dir=None, trust_remote=False,
                  download=True, full=False) -> str:
    return _br.locate_bundle(header, AE_V2, cache_dir, trust_remote, download,
                             full=full)


def load_from_bundle(bundle_dir, bitstream_id, backend, device, bf16=None,
                     threads=None, found_by="given as bundle="):
    return _br.load_from_bundle(bundle_dir, AE_V2, bitstream_id, backend,
                                device, bf16=bf16, threads=threads,
                                found_by=found_by)


def resolve_codec(header, bundle=None, backend=None, device=None, bf16=None,
                  threads=None, cache_dir=None, trust_remote=False):
    return _br.resolve_codec(header, AE_V2, bundle=bundle, backend=backend,
                             device=device, bf16=bf16, threads=threads,
                             cache_dir=cache_dir, trust_remote=trust_remote)
