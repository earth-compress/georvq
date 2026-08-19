"""Release bundles: a self-contained directory a collaborator can decode
from with nothing but `pip install georvq` (+extras).

Layout:
    model.pt        georvq checkpoint (weights + embedded config)
    cdfs.npz        calibrated rANS tables (optional, archive decoding)
    stats_pca.npz   stats-codec basis (optional, normalized sources)
    bundle.json     bitstream_id, decoder_version, notes
    README.md

`bitstream_id` hashes everything that DEFINES the bitstream (tier specs +
codebook tensors + stats basis + CDF tables) — decoder weights are
excluded, so improved decoders can ship against the same id and decode
existing archives (see docs/BITSTREAM.md).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from georvq.inference.api import Codec
from georvq.inference.stats_codec import StatsPacker
from georvq.repro import save_checkpoint


def bitstream_id(model, stats_pca=None, cdfs_path=None, extra=()) -> str:
    """Hash of everything that DEFINES the bitstream.

    Decoder weights are deliberately excluded, so a better decoder ships
    against the same id and decodes existing archives. Everything a
    decoder must agree on to reconstruct the same field is included, and
    `extra` is how a source's release layer adds its own such state --
    a transmitted-mean channel's constants, a refinement's table and
    rotation seed. Leaving those out would let two incompatible bundles
    claim one id, and mismatched-but-same-id state decodes to plausible
    garbage rather than to an error."""
    h = hashlib.sha256()
    for sp in model.specs:
        h.update(str((sp.grid, sp.dim, sp.codes, sp.depth)).encode())
    for rvq in model.rvqs:
        for lev in rvq.levels:
            h.update(lev.codebook.weight.detach().cpu().numpy().tobytes())
    if model.conditioner is not None:      # priors shape the coded residuals
        for p in model.conditioner.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
    if stats_pca is not None:
        h.update(np.asarray(stats_pca["components"]).tobytes())
    if cdfs_path and os.path.exists(cdfs_path):
        h.update(open(cdfs_path, "rb").read())
    for blob in extra:
        h.update(blob if isinstance(blob, bytes) else np.asarray(blob).tobytes())
    return h.hexdigest()[:16]


def decoder_id(model) -> str:
    """Identity of the decoder half (bitstream_id deliberately excludes
    it): hashes the decoder's parameters. Decoder-specific artifacts
    (e.g. per-window quality rasters) are keyed by this, so upgraded or
    variant decoders publish alongside rather than overwriting."""
    h = hashlib.sha256()
    sd = model.decoder.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def save_release(out_dir, model, cfg, step=0, cdfs_path=None,
                 stats_pca=None, notes=""):
    """Write a self-contained bundle: weights, optional side tables, and
    the ids that define what they decode.

    The `bitstream_id` returned here hashes everything that DEFINES the
    bitstream -- tier specs, codebooks, conditioning priors, the stats
    basis and the CDF tables -- but deliberately NOT the decoder
    weights, so a better decoder can ship against the same id and decode
    existing archives without re-encoding. Source-specific release state
    (containers, refinements, mean channels) belongs in that source's
    release layer, not here.
    """
    os.makedirs(out_dir, exist_ok=True)
    save_checkpoint(os.path.join(out_dir, "model.pt"), model, cfg, step)
    if cdfs_path:
        import shutil
        shutil.copy(cdfs_path, os.path.join(out_dir, "cdfs.npz"))
    if stats_pca is not None:
        np.savez(os.path.join(out_dir, "stats_pca.npz"), **stats_pca)
    bid = bitstream_id(model, stats_pca,
                       os.path.join(out_dir, "cdfs.npz") if cdfs_path else None)
    with open(os.path.join(out_dir, "bundle.json"), "w") as f:
        json.dump({"bitstream_id": bid,
                   "decoder_version": 1,
                   "window": model.window,
                   "tiers": [str(sp) for sp in model.specs],
                   "notes": notes}, f, indent=1)
    _write_readme(out_dir, model)
    return bid


def load_release(release_dir, device="cpu") -> Codec:
    """A ready Codec from a bundle directory (fresh-process safe)."""
    from georvq.factory import from_checkpoint
    model = from_checkpoint(os.path.join(release_dir, "model.pt"), device)
    packer = None
    spath = os.path.join(release_dir, "stats_pca.npz")
    if os.path.exists(spath):
        packer = StatsPacker(spath)
    codec = Codec(model, stats_packer=packer)
    bpath = os.path.join(release_dir, "bundle.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            codec.bundle = json.load(f)
    return codec


def _write_readme(out_dir, model):
    tiers = "\n".join(f"  - {sp}" for sp in model.specs)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(f"""# georvq release bundle

Pyramidal RVQ codec, window {model.window}px, tiers (grid:dim:codes:depth):
{tiers}

```python
from georvq.inference.release import load_release
codec = load_release("{os.path.basename(out_dir)}", device="cuda")
packets = codec.encode(x)              # x: prepared (B, C, W, W)
xhat = codec.decode(packets)           # full quality
xlow = codec.decode(packets, depths=[6, 1, 0])   # any monotone tuple
```

`bundle.json` carries the `bitstream_id`: archives and packets encoded
against this id decode with ANY future decoder that declares the same id.
""")
