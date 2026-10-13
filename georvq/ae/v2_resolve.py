"""Finding the bundle that decodes an AE v2 file.

The encoder records the release a file was coded with in the header, as
`codec.release` = {"name", "bitstream_id", "decoder_id", "uri",
"revision"}, where `uri` names a published bundle (`hf://org/repo`, or
`hf://org/repo/sub/dir` for a bundle below the repository root) and
`revision` a commit or tag of it (null for the repository's default
branch). Files written before the field existed carry only
`codec.bitstream_id`.

`locate_bundle` looks for a bundle with the file's bitstream id in
turn in

1. the directory named by the environment variable GEORVQ_AE_BUNDLE;
2. the bundle cache, `<root>/bundles/<bitstream_id>/`, where root is
   the `cache_dir` argument, else $GEORVQ_CACHE, else
   $XDG_CACHE_HOME/georvq, else ~/.cache/georvq;
3. the release's `uri` at its `revision`, downloaded into the cache,
   when the repository matches a pattern of TRUSTED_RELEASES or the
   caller passes `trust_remote=True`.

A bundle found at 1 or 2 whose bundle.json names another bitstream id
is refused, not passed over. A download goes to a staging directory
inside the cache and is moved to its cache entry only after its
bundle.json bitstream id and the file hashes it records have been
checked. A file without `codec.release` resolves through 1 and 2 only.

`resolve_codec` loads the located bundle for a backend: the torch
codec through `georvq.ae.v2_bundle.load_bundle(verify=True)`, or the
ONNX codec from the export a bundle holds under `onnx/` (see
`write_bundle(..., onnx=True)`), whose recorded hashes are checked.

This module imports neither torch nor huggingface_hub until it needs
them.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import tempfile

BUNDLE_FORMAT = "georvq-ae/v2-bundle"
RELEASE_NAME = "ae-v2"
RELEASE_URI = "hf://earth-compress/georvq-ae-v2"
# patterns over "hf://<org>/<repo>"; a release outside them is downloaded
# only with trust_remote=True
TRUSTED_RELEASES = ("hf://earth-compress/*",)
ENV_BUNDLE = "GEORVQ_AE_BUNDLE"
ENV_CACHE = "GEORVQ_CACHE"
ONNX_DIR = "onnx"

_ID = re.compile(r"[0-9A-Za-z_-]{1,64}")
_HF_PART = re.compile(r"[A-Za-z0-9][\w.-]*")


class BundleError(ValueError):
    """No bundle for a file could be found, or the one found does not
    belong to it."""


def _how_to_provide(bid, cache_dir=None) -> str:
    return (f"Provide the bundle for bitstream {bid} with open(path, "
            f"bundle=DIR), the environment variable {ENV_BUNDLE}=DIR, or "
            f"by placing it at {bundle_path(bid, cache_dir)}; or pass "
            "model= and channel=, or onnx_dir= with backend='onnx'.")


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------

def release_record(codec, name=RELEASE_NAME, uri=RELEASE_URI,
                   revision=None) -> dict | None:
    """The header's `codec.release` for a file coded by `codec`; None
    when `name` is None."""
    if name is None:
        return None
    return {"name": name, "bitstream_id": codec.bitstream_id,
            "decoder_id": codec.decoder_id, "uri": uri,
            "revision": revision}


def file_release(header) -> dict | None:
    """The header's release record, or None for a file without one (an
    early file may hold a plain string there)."""
    rel = header["codec"].get("release")
    return rel if isinstance(rel, dict) else None


def file_bitstream_id(header) -> str:
    bid = header["codec"]["bitstream_id"]
    rel = file_release(header)
    if rel is not None and rel.get("bitstream_id") not in (None, bid):
        raise BundleError(f"header codec.release names bitstream "
                          f"{rel['bitstream_id']}, codec.bitstream_id {bid}")
    if not isinstance(bid, str) or not _ID.fullmatch(bid):
        raise BundleError(f"bitstream id {bid!r} is not a valid cache key")
    return bid


# --------------------------------------------------------------------------
# cache and bundle checks
# --------------------------------------------------------------------------

def cache_root(cache_dir=None) -> str:
    """The georvq cache directory: `cache_dir`, else $GEORVQ_CACHE, else
    $XDG_CACHE_HOME/georvq, else ~/.cache/georvq."""
    if cache_dir is None:
        cache_dir = os.environ.get(ENV_CACHE)
    if not cache_dir:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache")
        cache_dir = os.path.join(xdg, "georvq")
    return os.path.expanduser(os.fspath(cache_dir))


def bundle_path(bitstream_id, cache_dir=None) -> str:
    """Cache entry of the bundle for `bitstream_id`."""
    return os.path.join(cache_root(cache_dir), "bundles", bitstream_id)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def read_bundle_meta(bundle_dir) -> dict:
    path = os.path.join(bundle_dir, "bundle.json")
    try:
        with open(path) as f:
            meta = json.load(f)
    except FileNotFoundError:
        raise BundleError(f"{bundle_dir}: no bundle.json, not an AE v2 "
                          "bundle") from None
    if meta.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"{bundle_dir}: not an AE v2 bundle (format "
                          f"{meta.get('format')!r})")
    return meta


def check_files(bundle_dir, meta, names=None):
    """Compare the sha256 of the bundle's files (`names`, default all)
    with the hashes bundle.json records."""
    for name in meta["files"] if names is None else names:
        rec = meta["files"][name]
        got = sha256_file(os.path.join(bundle_dir, name))
        if got != rec["sha256"]:
            raise BundleError(f"{bundle_dir}/{name}: sha256 {got}, "
                              f"bundle.json records {rec['sha256']}")


def check_bundle(bundle_dir, bitstream_id, found_by) -> dict:
    """bundle.json of `bundle_dir`, refused unless it is a bundle for
    `bitstream_id`. `found_by` says where the directory came from, for
    the error message."""
    meta = read_bundle_meta(bundle_dir)
    if meta.get("bitstream_id") != bitstream_id:
        raise BundleError(
            f"the bundle at {bundle_dir} ({found_by}) is for bitstream "
            f"{meta.get('bitstream_id')}, the file was coded with "
            f"{bitstream_id}; it cannot decode this file")
    return meta


# --------------------------------------------------------------------------
# remote
# --------------------------------------------------------------------------

def parse_hf_uri(uri) -> tuple:
    """"hf://org/repo[/sub/dir]" -> (repo_id, subdir or None)."""
    if not isinstance(uri, str) or not uri.startswith("hf://"):
        raise BundleError(f"release uri {uri!r}: only hf:// uris can be "
                          "downloaded")
    parts = uri[len("hf://"):].strip("/").split("/")
    if len(parts) < 2 or not all(_HF_PART.fullmatch(p) and ".." not in p
                                 for p in parts):
        raise BundleError(f"release uri {uri!r} is not hf://org/repo[/dir]")
    return "/".join(parts[:2]), "/".join(parts[2:]) or None


def is_trusted(uri, trusted=TRUSTED_RELEASES) -> bool:
    repo_id, _ = parse_hf_uri(uri)
    return any(fnmatch.fnmatchcase(f"hf://{repo_id}", p) for p in trusted)


def download_bundle(uri, revision, bitstream_id, cache_dir=None) -> str:
    """Download the bundle at `uri`/`revision` into the cache entry for
    `bitstream_id` and return the entry. The trust check is the
    caller's."""
    repo_id, sub = parse_hf_uri(uri)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise BundleError(f"downloading {uri} needs huggingface_hub. "
                          + _how_to_provide(bitstream_id, cache_dir)) from None
    entry = bundle_path(bitstream_id, cache_dir)
    parent = os.path.dirname(entry)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f".{bitstream_id}-", dir=parent)
    at = f"{uri}@{revision or 'default branch'}"
    try:
        snapshot_download(repo_id=repo_id, repo_type="model",
                          revision=revision, local_dir=staging,
                          allow_patterns=[f"{sub}/*"] if sub else None)
        src = os.path.join(staging, sub) if sub else staging
        meta = check_bundle(src, bitstream_id, f"downloaded from {at}")
        check_files(src, meta)
        shutil.rmtree(os.path.join(src, ".cache"), ignore_errors=True)
        try:
            os.replace(src, entry)
        except OSError:
            # another process filled the entry first
            if not os.path.isfile(os.path.join(entry, "bundle.json")):
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return entry


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def locate_bundle(header, cache_dir=None, trust_remote=False,
                  download=True) -> str:
    """Directory of the bundle for the file whose header is `header`,
    found through GEORVQ_AE_BUNDLE, the cache, or (with `download`) the
    release's uri. Raises BundleError when none is found or the one
    found is for another bitstream."""
    bid = file_bitstream_id(header)
    env = os.environ.get(ENV_BUNDLE)
    if env:
        check_bundle(env, bid, f"from {ENV_BUNDLE}")
        return env
    entry = bundle_path(bid, cache_dir)
    if os.path.isfile(os.path.join(entry, "bundle.json")):
        check_bundle(entry, bid, "in the bundle cache")
        return entry
    rel = file_release(header)
    if rel is None or not rel.get("uri"):
        raise BundleError(f"no bundle for bitstream {bid} is cached and the "
                          "file names no release to download it from. "
                          + _how_to_provide(bid, cache_dir))
    uri = rel["uri"]
    if not download:
        raise BundleError(f"no bundle for bitstream {bid} is cached; the "
                          f"file's release is {uri}")
    if not trust_remote and not is_trusted(uri):
        raise BundleError(
            f"the file names release {uri}, which matches none of the "
            f"trusted patterns {list(TRUSTED_RELEASES)}. Pass "
            "trust_remote=True to download it from there. "
            + _how_to_provide(bid, cache_dir))
    return download_bundle(uri, rel.get("revision"), bid, cache_dir)


def default_backend() -> str:
    """"torch" when torch imports, else "onnx"."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return "onnx"
    return "torch"


def default_device(backend) -> str:
    """cuda when the backend is torch and torch sees a GPU, else cpu."""
    if backend == "torch":
        import torch
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def load_from_bundle(bundle_dir, bitstream_id, backend, device, bf16=None,
                     threads=None, found_by="given as bundle="):
    """A codec for `backend` from a bundle directory, refused unless the
    bundle is for `bitstream_id`."""
    meta = check_bundle(bundle_dir, bitstream_id, found_by)
    if backend == "onnx":
        sub = meta.get("onnx")
        if not sub:
            raise BundleError(
                f"the bundle at {bundle_dir} holds no ONNX export, so the "
                "onnx backend cannot use it. Use backend='torch', pass "
                "onnx_dir= with an export_onnx directory, or build the "
                "bundle with build_bundle(..., onnx=True).")
        check_files(bundle_dir, meta,
                    [n for n in meta["files"] if n.startswith(sub + "/")])
        from georvq.ae.v2 import AEV2OnnxCodec
        return AEV2OnnxCodec(os.path.join(bundle_dir, sub), device=device,
                             threads=threads)
    if backend != "torch":
        raise ValueError(f"unknown backend {backend!r}")
    from georvq.ae.v2_bundle import load_bundle
    return load_bundle(bundle_dir, device=device, bf16=bf16, verify=True)


def resolve_codec(header, bundle=None, backend=None, device=None, bf16=None,
                  threads=None, cache_dir=None, trust_remote=False):
    """A codec for the file whose header is `header`, from `bundle` when
    given, else as `locate_bundle` finds it."""
    bid = file_bitstream_id(header)
    backend = backend or default_backend()
    device = device or default_device(backend)
    if bundle is not None:
        path, how = os.fspath(bundle), "given as bundle="
    else:
        path = locate_bundle(header, cache_dir, trust_remote)
        how = "located for the file"
    return load_from_bundle(path, bid, backend, device, bf16=bf16,
                            threads=threads, found_by=how)
