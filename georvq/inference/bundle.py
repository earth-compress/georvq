"""Release bundles: the files a decoder of a release's stores needs.

Layout of a bundle directory:

    weights.npz       every model tensor except the codebooks and the
                      VQ training state, at the dtype the model holds
    codebooks.npz     per RVQ level, `<prefix>.q` (K, D) int8 and
                      `<prefix>.scale` (K,) float32, with
                      codebook = float32(q) * scale exactly, and
                      q = -128 standing for -0.0; for a model trained
                      with float32 books (vq.book_bits unset),
                      `<prefix>.weight` (K, D) float32 instead
    mean_channel.npz  the mean channel's tables (georvq.inference.
                      mean_codec; its kind is recorded in bundle.json)
    config.json       the model's resolved Config
    bundle.json       ids, tiers, rungs with their bits, file hashes
    README.md
    onnx/             optional (`onnx=True`): the `export_onnx`
                      directory, for the torch-free backend

bundle.json records the sha256 of every file but itself and the README,
those under onnx/ included, and names the ONNX directory under "onnx"
(null without one). Its `format` and `codec_format` are the release's
(`ReleaseSpec.bundle_format`, `codec_format`).

Everything is stored as plain numpy arrays and JSON, and the loader
reads the arrays with `allow_pickle=False`, so loading a bundle runs no
pickled code.

The VQ training state (`ema_count`, `ema_sum`, `step`, `initialized`)
is left out: it only drives the EMA update during training. The loader
leaves those buffers at their fresh values and marks every level
initialized.

A QuantizedBookVQ book is stored as `q * scale` per codeword. The
training projection computes the scale as max|w| / qmax, but after the
fp32 product the scale recomputed from a saved book can differ from the
one used by an ulp, and then `q * scale` misses the stored weight. The
writer searches the neighbouring floats for a scale that reproduces the
row bit for bit and refuses to write a book where none does. The
projection rounds small negative weights to -0.0, and the sign of zero
is part of the bytes `bitstream_id` hashes, so the int8 code -128
(outside the symmetric range) stores -0.0.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from georvq.inference.bundle_resolve import (ONNX_DIR, check_files,
                                             read_bundle_meta, sha256_file)
from georvq.inference.store_codec import export_onnx, make_codec

BUNDLE_VERSION = 1
FILES = ("weights.npz", "codebooks.npz", "mean_channel.npz", "config.json")
TRAINING_STATE = ("ema_count", "ema_sum", "ema_sse", "step", "initialized")
BOOK_SUFFIX = ".codebook.weight"
# a scale found by the search is at most this many ulps from max|w| / qmax
SCALE_SEARCH_ULPS = 4
NEG_ZERO = -128


def _is_training_state(key) -> bool:
    return key.startswith("rvqs.") and key.rsplit(".", 1)[-1] in TRAINING_STATE


def _book_keys(model) -> list:
    return [f"rvqs.{t}.levels.{i}"
            for t, rvq in enumerate(model.rvqs)
            for i in range(len(rvq.levels))]


def book_to_int(W, bits=8):
    """(K, D) float32 on-grid book -> (q int8 (K, D), scale float32 (K,))
    with book_from_int(q, scale) equal to W bit for bit, -0.0 coded as
    NEG_ZERO. Raises ValueError if some row has no such representation."""
    W = torch.as_tensor(W, dtype=torch.float32).cpu()
    qmax = float(2 ** (bits - 1) - 1)
    s0 = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
    wi = W.view(torch.int32)
    scale = s0.clone()
    q_out = torch.zeros_like(W)
    found = torch.zeros(W.shape[0], dtype=torch.bool)
    for d in [0] + [s * k for k in range(1, SCALE_SEARCH_ULPS + 1)
                    for s in (1, -1)]:
        s = (s0.view(torch.int32) + d).view(torch.float32)
        q = (W / s).round().clamp(-qmax, qmax)
        ok = ((q * s).view(torch.int32) == wi).all(1) & ~found
        scale[ok] = s[ok]
        q_out[ok] = q[ok]
        found |= ok
        if bool(found.all()):
            break
    if not bool(found.all()):
        raise ValueError(f"{int((~found).sum())} codewords are not on the "
                         f"{bits}-bit grid")
    q = q_out.to(torch.int8)
    q[(W == 0) & torch.signbit(W)] = NEG_ZERO
    q, scale = q.numpy(), scale[:, 0].numpy()
    if not torch.equal(book_from_int(q, scale).view(torch.int32), wi):
        raise ValueError("int8 book does not reproduce the weights")
    return q, scale


def book_from_int(q, scale) -> torch.Tensor:
    """Inverse of book_to_int: float32(q) * scale per codeword."""
    q = torch.from_numpy(np.asarray(q))
    f = torch.where(q == NEG_ZERO, torch.tensor(-0.0), q.float())
    return f * torch.from_numpy(np.asarray(scale, dtype=np.float32))[:, None]


def _numpy(t):
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        raise TypeError("bfloat16 tensors have no numpy dtype")
    return t.numpy()


def shippable_config(d):
    """A config dump without the paths of the machine it was trained on:
    absolute path strings become "" and list entries that start with one
    are dropped. Building the model reads none of these fields."""
    if isinstance(d, dict):
        return {k: shippable_config(v) for k, v in d.items()}
    if isinstance(d, list):
        return [shippable_config(v) for v in d
                if not (isinstance(v, str) and v.startswith("/"))]
    if isinstance(d, str) and d.startswith("/"):
        return ""
    return d


def write_bundle(out_dir, model, cfg, channel, spec, source=None, notes="",
                 onnx=False, extra=None) -> dict:
    """Write a bundle of the release `spec` for a loaded model and its
    mean channel. Returns the bundle.json payload. `source` is
    descriptive metadata about the checkpoint the weights came from;
    `extra` further JSON entries of bundle.json. `onnx` also writes the
    model's ONNX export under onnx/, so that one bundle serves both
    backends."""
    bits = cfg.model.vq.book_bits
    codec = make_codec(spec, model, channel)   # checks the model and bands
    os.makedirs(out_dir, exist_ok=True)
    sd = model.state_dict()
    prefixes = _book_keys(model)
    book_keys = {p + BOOK_SUFFIX for p in prefixes}
    weights = {k: _numpy(v) for k, v in sd.items()
               if k not in book_keys and not _is_training_state(k)}
    books = {}
    for p in prefixes:
        if bits:
            q, s = book_to_int(sd[p + BOOK_SUFFIX], bits)
            books[p + ".q"], books[p + ".scale"] = q, s
        else:
            books[p + ".weight"] = _numpy(sd[p + BOOK_SUFFIX].float())
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)
    np.savez(os.path.join(out_dir, "codebooks.npz"), **books)
    channel.save(os.path.join(out_dir, "mean_channel.npz"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(shippable_config(cfg.model_dump(mode="json")), f, indent=1)
    files = list(FILES)
    if onnx:
        export_onnx(codec, os.path.join(out_dir, ONNX_DIR))
        files += [f"{ONNX_DIR}/{n}" for n in
                  sorted(os.listdir(os.path.join(out_dir, ONNX_DIR)))]

    payload = {
        "format": spec.bundle_format, "version": BUNDLE_VERSION,
        "codec_format": spec.codec_format,
        "bitstream_id": codec.bitstream_id,
        "decoder_id": codec.decoder_id,
        "window": codec.window, "bands": codec.bands,
        "tiers": [str(sp) for sp in codec.specs],
        "min_depths": list(cfg.model.min_depths),
        "book_bits": int(bits) if bits else None,
        "mean_channel": {"kind": channel.kind,
                         "bits": int(channel.cost_bits())},
        "rungs": codec.header_rungs(),
        "onnx": ONNX_DIR if onnx else None,
        "files": {n: {"bytes": os.path.getsize(os.path.join(out_dir, n)),
                      "sha256": sha256_file(os.path.join(out_dir, n))}
                  for n in files},
        "source": source or {},
        "notes": notes,
        **(extra or {}),
    }
    with open(os.path.join(out_dir, "bundle.json"), "w") as f:
        json.dump(payload, f, indent=1)
    if spec.bundle_readme is not None:
        with open(os.path.join(out_dir, "README.md"), "w") as f:
            f.write(spec.bundle_readme(payload))
    return payload


def build_bundle(ckpt, mean_npz, out_dir, spec, notes="", onnx=False,
                 extra=None) -> dict:
    """Bundle a training checkpoint and its mean channel tables; `onnx`
    and `extra` as for write_bundle."""
    from georvq.factory import from_checkpoint
    from georvq.inference.mean_codec import load_channel
    from georvq.repro import load_checkpoint
    ck = load_checkpoint(ckpt)
    if ck.get("prep"):
        raise NotImplementedError("checkpoints with fitted prep constants "
                                  "are not bundled")
    model = from_checkpoint(ckpt, "cpu")
    channel = load_channel(os.fspath(mean_npz))
    source = {"checkpoint": os.path.basename(os.fspath(ckpt)),
              "step": ck.get("step"), "config_hash": ck.get("config_hash"),
              "git": ck.get("git"), "metrics": ck.get("metrics")}
    return write_bundle(out_dir, model, model.full_config, channel, spec,
                        source=source, notes=notes, onnx=onnx, extra=extra)


def load_model(bundle_dir, spec, device="cpu", verify=True):
    """(model, bundle.json payload) of a bundle, the model in eval mode.
    `verify` checks the file hashes recorded in bundle.json."""
    from georvq.config import Config, migrate
    from georvq.factory import build_model
    meta = read_bundle_meta(bundle_dir, spec)
    if verify:
        check_files(bundle_dir, meta)
    with open(os.path.join(bundle_dir, "config.json")) as f:
        cfg = Config.model_validate(migrate(json.load(f)))
    model = build_model(cfg, "cpu")
    sd = {}
    with np.load(os.path.join(bundle_dir, "weights.npz"),
                 allow_pickle=False) as z:
        for k in z.files:
            sd[k] = torch.from_numpy(z[k])
    with np.load(os.path.join(bundle_dir, "codebooks.npz"),
                 allow_pickle=False) as z:
        for p in _book_keys(model):
            if p + ".weight" in z.files:
                sd[p + BOOK_SUFFIX] = torch.from_numpy(z[p + ".weight"])
            else:
                sd[p + BOOK_SUFFIX] = book_from_int(z[p + ".q"],
                                                    z[p + ".scale"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected or any(not _is_training_state(k) for k in missing):
        raise KeyError(f"{bundle_dir}: unexpected {sorted(unexpected)}, "
                       f"missing {sorted(k for k in missing if not _is_training_state(k))}")
    for rvq in model.rvqs:
        for lev in rvq.levels:
            lev.initialized.fill_(True)
    model.eval()
    return model.to(device), meta


def load_bundle(bundle_dir, spec, device="cpu", bf16=None, verify=True):
    """The release's codec from a bundle directory. Raises if the ids
    recomputed from the loaded tensors differ from those in bundle.json."""
    from georvq.inference.mean_codec import load_channel
    model, meta = load_model(bundle_dir, spec, device, verify=verify)
    channel = load_channel(os.path.join(bundle_dir, "mean_channel.npz"),
                           (meta.get("mean_channel") or {}).get("kind"))
    codec = make_codec(spec, model, channel, bf16=bf16)
    if verify:
        for key in ("bitstream_id", "decoder_id"):
            if getattr(codec, key) != meta[key]:
                raise ValueError(f"{bundle_dir}: loaded tensors give {key} "
                                 f"{getattr(codec, key)}, bundle.json "
                                 f"records {meta[key]}")
    codec.bundle = meta
    return codec
