"""Finding the bundle that decodes a georvq store.

The encoder records the release a store was coded with in the header, as
`codec.release` = {"name", "bitstream_id", "decoder_id", "uri",
"revision"}, where `uri` names a published bundle (`hf://org/repo`, or
`hf://org/repo/sub/dir` for a bundle below the repository root) and
`revision` a commit or tag of it (null for the repository's default
branch). Stores written before the field existed carry only
`codec.bitstream_id`.

`locate_bundle(header, spec)` looks for a bundle with the store's
bitstream id in turn in

1. the directory named by the release's environment variable
   (`spec.env_bundle`, e.g. GEORVQ_AE_BUNDLE), when it has one;
2. the bundle cache, `<root>/bundles/<bitstream_id>/`, where root is
   the `cache_dir` argument, else $GEORVQ_CACHE, else
   $XDG_CACHE_HOME/georvq, else ~/.cache/georvq;
3. the release's `uri` at its `revision`, downloaded into the cache,
   when the repository matches a pattern of `spec.trusted` or the caller
   passes `trust_remote=True`.

A bundle found at 1 or 2 whose bundle.json names another bitstream id
is refused, not passed over. A download goes to a staging directory
inside the cache and is moved to its cache entry only after its
bundle.json bitstream id and the file hashes it records have been
checked. A store without `codec.release` resolves through 1 and 2 only.

A release publishes its full bundle at `uri` and the decoder-only
bundle, with the same ids, in `uri/decoder` (georvq.inference.bundle).
Reading with torch downloads only the decoder-only bundle, or, from a
revision without one, the full bundle's files other than its ONNX
export. The ONNX backend and encoding (`full=True`) need the full
bundle, which replaces a cache entry that holds less.

`resolve_codec` loads the located bundle for a backend: the torch
codec through `georvq.inference.bundle.load_bundle(verify=True)`, or the
ONNX codec from the export a bundle holds under `onnx/`, whose recorded
hashes are checked.

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

ENV_CACHE = "GEORVQ_CACHE"
ONNX_DIR = "onnx"
DECODER_DIR = "decoder"
FULL, DECODER = "full", "decoder"
# the files of a bundle the torch backend reads, besides bundle.json
FILES = ("weights.npz", "codebooks.npz", "mean_channel.npz", "config.json")

_ID = re.compile(r"[0-9A-Za-z_-]{1,64}")
_HF_PART = re.compile(r"[A-Za-z0-9][\w.-]*")


class BundleError(ValueError):
    """No bundle for a store could be found, or the one found does not
    belong to it."""


def _how_to_provide(spec, bid, cache_dir=None) -> str:
    env = (f" the environment variable {spec.env_bundle}=DIR,"
           if spec.env_bundle else "")
    return (f"Provide the bundle for bitstream {bid} with open(path, "
            f"bundle=DIR),{env} or by placing it at "
            f"{bundle_path(bid, cache_dir)}; or pass model= and channel=, "
            "or onnx_dir= with backend='onnx'.")


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------

def file_release(header) -> dict | None:
    """The header's release record, or None for a store without one (an
    early store may hold a plain string there)."""
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


def read_bundle_meta(bundle_dir, spec) -> dict:
    path = os.path.join(bundle_dir, "bundle.json")
    try:
        with open(path) as f:
            meta = json.load(f)
    except FileNotFoundError:
        raise BundleError(f"{bundle_dir}: no bundle.json, not a "
                          f"{spec.name} bundle") from None
    if meta.get("format") != spec.bundle_format:
        raise BundleError(f"{bundle_dir}: not a {spec.name} bundle (format "
                          f"{meta.get('format')!r})")
    return meta


def check_files(bundle_dir, meta, names=None):
    """Compare the sha256 of the bundle's files (`names`, default all)
    with the hashes bundle.json records."""
    redo = (f"If {bundle_dir} is a bundle cache entry, delete it and it is "
            "downloaded again on the next read.")
    for name in meta["files"] if names is None else names:
        rec = meta["files"][name]
        try:
            got = sha256_file(os.path.join(bundle_dir, name))
        except FileNotFoundError:
            raise BundleError(f"{bundle_dir}/{name} is missing; bundle.json "
                              f"records it. {redo}") from None
        if got != rec["sha256"]:
            raise BundleError(f"{bundle_dir}/{name}: sha256 {got}, "
                              f"bundle.json records {rec['sha256']}. {redo}")


def torch_files(meta) -> list:
    """The files bundle.json records that the torch backend reads: all
    but the ONNX export's."""
    sub = (meta.get("onnx") or ONNX_DIR) + "/"
    return [n for n in meta["files"] if not n.startswith(sub)]


def serves(bundle_dir, meta, full=False) -> bool:
    """Whether the bundle at `bundle_dir` holds what a reader needs: the
    files the torch backend reads, or with `full` the whole full bundle
    (encoder and ONNX export)."""
    if full and meta.get("variant", FULL) != FULL:
        return False
    names = list(meta["files"]) if full else torch_files(meta)
    return all(os.path.isfile(os.path.join(bundle_dir, n)) for n in names)


def check_bundle(bundle_dir, spec, bitstream_id, found_by) -> dict:
    """bundle.json of `bundle_dir`, refused unless it is a bundle of the
    release for `bitstream_id`. `found_by` says where the directory came
    from, for the error message."""
    meta = read_bundle_meta(bundle_dir, spec)
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


def is_trusted(uri, trusted) -> bool:
    repo_id, _ = parse_hf_uri(uri)
    return any(fnmatch.fnmatchcase(f"hf://{repo_id}", p) for p in trusted)


def download_bundle(uri, revision, bitstream_id, spec, cache_dir=None,
                    full=False) -> str:
    """Download the bundle at `uri`/`revision` into the cache entry for
    `bitstream_id` and return the entry: the decoder-only bundle in
    `uri/decoder`, or the full bundle's files without its ONNX export
    when the revision has none, or with `full` the whole full bundle. An
    existing entry is replaced. The trust check is the caller's."""
    repo_id, sub = parse_hf_uri(uri)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise BundleError(f"downloading {uri} needs huggingface_hub. "
                          + _how_to_provide(spec, bitstream_id, cache_dir)
                          ) from None
    entry = bundle_path(bitstream_id, cache_dir)
    parent = os.path.dirname(entry)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f".{bitstream_id}-", dir=parent)
    at = f"{uri}@{revision or 'default branch'}"
    pre = f"{sub}/" if sub else ""
    root = os.path.join(staging, sub) if sub else staging

    def fetch(patterns):
        try:
            snapshot_download(repo_id=repo_id, repo_type="model",
                              revision=revision, local_dir=staging,
                              allow_patterns=patterns)
        except Exception as err:
            # offline, 401/404 from a private or missing repository, 5xx,
            # timeouts: huggingface_hub, httpx and OSError types alike
            raise BundleError(
                f"could not download the bundle for bitstream {bitstream_id} "
                f"from {uri} at revision {revision or '(default branch)'}: "
                f"{type(err).__name__}: {err}. A private repository needs a "
                "Hugging Face token: run `hf auth login` or set HF_TOKEN. "
                + _how_to_provide(spec, bitstream_id, cache_dir)) from err

    top = [pre + n for n in ("bundle.json",) + FILES]
    try:
        src = None
        if not full:
            fetch([f"{pre}{DECODER_DIR}/*"])
            dec = os.path.join(root, DECODER_DIR)
            if os.path.isfile(os.path.join(dec, "bundle.json")):
                src = dec
        if src is None:
            fetch(top + ([f"{pre}{ONNX_DIR}/*"] if full else []))
            src = root
        meta = check_bundle(src, spec, bitstream_id, f"downloaded from {at}")
        if full and meta.get("variant", FULL) != FULL:
            raise BundleError(f"the bundle at {at} is not a full bundle")
        check_files(src, meta, None if full or meta.get("variant") == DECODER
                    else torch_files(meta))
        shutil.rmtree(os.path.join(src, ".cache"), ignore_errors=True)
        if src != root:
            shutil.rmtree(os.path.join(staging, ".cache"), ignore_errors=True)
        if os.path.isdir(entry):
            # an entry that holds less than this download is replaced
            os.replace(entry, os.path.join(staging, ".old"))
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

def locate_bundle(header, spec, cache_dir=None, trust_remote=False,
                  download=True, full=False) -> str:
    """Directory of the bundle for the store whose header is `header`,
    found through the release's environment variable, the cache, or
    (with `download`) the release's uri. `full` asks for the full bundle
    (encoder and ONNX export); otherwise any bundle the torch backend can
    read will do. Raises BundleError when none is found or the one found
    is for another bitstream."""
    bid = file_bitstream_id(header)
    env = os.environ.get(spec.env_bundle) if spec.env_bundle else None
    if env:
        check_bundle(env, spec, bid, f"from {spec.env_bundle}")
        return env
    entry = bundle_path(bid, cache_dir)
    if os.path.isfile(os.path.join(entry, "bundle.json")):
        meta = check_bundle(entry, spec, bid, "in the bundle cache")
        # a cached entry missing a file it records is reported when
        # loaded, except that an entry holding less than the full bundle
        # asked for is replaced by a download of it
        if not full or not download or serves(entry, meta, full=True):
            return entry
    rel = file_release(header)
    if rel is None or not rel.get("uri"):
        raise BundleError(f"no bundle for bitstream {bid} is cached and the "
                          "file names no release to download it from. "
                          + _how_to_provide(spec, bid, cache_dir))
    uri = rel["uri"]
    if not download:
        raise BundleError(f"no bundle for bitstream {bid} is cached; the "
                          f"file's release is {uri}")
    if not trust_remote and not is_trusted(uri, spec.trusted):
        raise BundleError(
            f"the file names release {uri}, which matches none of the "
            f"trusted patterns {list(spec.trusted)}. Pass "
            "trust_remote=True to download it from there. "
            + _how_to_provide(spec, bid, cache_dir))
    return download_bundle(uri, rel.get("revision"), bid, spec, cache_dir,
                           full=full)


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


def load_from_bundle(bundle_dir, spec, bitstream_id, backend, device,
                     bf16=None, threads=None, found_by="given as bundle="):
    """A codec for `backend` from a bundle directory, refused unless the
    bundle is for `bitstream_id`."""
    meta = check_bundle(bundle_dir, spec, bitstream_id, found_by)
    if backend == "onnx":
        sub = meta.get("onnx")
        if not sub:
            raise BundleError(
                f"the bundle at {bundle_dir} holds no ONNX export, so the "
                "onnx backend cannot use it. Use backend='torch', pass "
                "onnx_dir= with an export_onnx directory, or build the "
                "bundle with build_bundle(..., onnx=True).")
        # the export and, for one inside the bundle, the bundle's books
        # and channel, which its meta.json names as ../<file>
        check_files(bundle_dir, meta,
                    [n for n in meta["files"] if n.startswith(sub + "/")
                     or n in ("codebooks.npz", "mean_channel.npz")])
        from georvq.inference.store_codec import make_onnx_codec
        return make_onnx_codec(spec, os.path.join(bundle_dir, sub),
                               device=device, threads=threads)
    if backend != "torch":
        raise ValueError(f"unknown backend {backend!r}")
    from georvq.inference.bundle import load_bundle
    return load_bundle(bundle_dir, spec, device=device, bf16=bf16, verify=True)


def resolve_codec(header, spec, bundle=None, backend=None, device=None,
                  bf16=None, threads=None, cache_dir=None, trust_remote=False):
    """A codec for the store whose header is `header`, from `bundle` when
    given, else as `locate_bundle` finds it."""
    bid = file_bitstream_id(header)
    backend = backend or default_backend()
    device = device or default_device(backend)
    if bundle is not None:
        path, how = os.fspath(bundle), "given as bundle="
    else:
        path = locate_bundle(header, spec, cache_dir, trust_remote,
                             full=backend == "onnx")
        how = "located for the file"
    with _quiet():
        return load_from_bundle(path, spec, bid, backend, device, bf16=bf16,
                                threads=threads, found_by=how)


def _quiet():
    from georvq.inference.store_codec import quiet_build
    return quiet_build()
