"""Location encoders and location->latent priors (optional capability).

SatCLIP (Microsoft, MIT) provides a pretrained (lon, lat) -> 256-d embedding
that carries strong geographic context. Its repo is not pip-packaged and its
analytic spherical-harmonics tables are large, so instead of vendoring we
clone a pinned commit into a local cache on first use (override with
GEORVQ_SATCLIP_DIR to point at an existing checkout). Weights come from
HuggingFace (`huggingface_hub`, install extra `georvq[loc]`).
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

SATCLIP_GIT = "https://github.com/microsoft/satclip.git"
SATCLIP_COMMIT = "1ead1067e9a00d44b132403b6bb2ce061e48d48e"
SATCLIP_HF_REPO = "microsoft/SatCLIP-ResNet18-L40"
SATCLIP_HF_FILE = "satclip-resnet18-l40.ckpt"


def _satclip_src() -> str:
    """Path to the satclip package source (cached clone at a pinned commit)."""
    override = os.environ.get("GEORVQ_SATCLIP_DIR")
    if override:
        return os.path.join(override, "satclip")
    cache = os.path.join(os.path.expanduser("~"), ".cache", "georvq",
                         "satclip")
    if not os.path.isdir(os.path.join(cache, "satclip")):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        try:
            subprocess.run(["git", "clone", "--quiet", SATCLIP_GIT, cache],
                           check=True)
            subprocess.run(["git", "-C", cache, "checkout", "--quiet",
                            SATCLIP_COMMIT], check=True)
        except Exception as e:
            raise RuntimeError(
                f"could not fetch the SatCLIP source ({e}); clone "
                f"{SATCLIP_GIT} yourself and set GEORVQ_SATCLIP_DIR") from e
    return os.path.join(cache, "satclip")


def get_satclip(device="cpu"):
    """The pretrained SatCLIP location encoder: (N, 2) lon/lat degrees
    (float64) -> (N, 256) embeddings."""
    src = _satclip_src()
    if src not in sys.path:
        sys.path.insert(0, src)
    from huggingface_hub import hf_hub_download
    from load_lightweight import get_satclip_loc_encoder
    ck = hf_hub_download(SATCLIP_HF_REPO, SATCLIP_HF_FILE)
    return get_satclip_loc_encoder(ck, "cpu").to(device)


def get_spherical_harmonics(degree=40):
    """Raw analytic spherical-harmonics location features (degree^2 dims);
    an untrained, band-limited alternative to SatCLIP."""
    src = _satclip_src()
    if src not in sys.path:
        sys.path.insert(0, src)
    from location_encoder import get_positional_encoding
    return get_positional_encoding("sphericalharmonics",
                                   legendre_polys=degree,
                                   harmonics_calculation="analytic").double()


def get_location_encoder(encoder="satclip", device="cpu", degree=40):
    """Location-encoder factory: "satclip" (pretrained 256-d context) or
    "sphharm" (raw analytic SH features, degree^2 dims — the band-limited
    positional basis for a learn-your-own location prior)."""
    if encoder == "sphharm":
        return get_spherical_harmonics(degree).to(device)
    return get_satclip(device)


def build_location_table(rows, device, encoder=None, standardize=True):
    """Per-row standardized location embeddings (len(rows), D) from each
    chip's lon/lat."""
    enc = encoder or get_satclip(device)
    p = next(enc.parameters(), None)
    if p is None:   # analytic SH is parameterless; evaluate on `device`
        p = next(enc.buffers(), None)
    enc_device = p.device if p is not None else torch.device(device)
    ll = torch.tensor(np.stack([rows.lon.values, rows.lat.values], 1),
                      dtype=torch.float64, device=enc_device)
    with torch.no_grad():
        s = enc(ll).float()
    if standardize:
        s = (s - s.mean(0)) / (s.std(0) + 1e-6)
    return s.to(device)


def build_location_tables(cfg, store, train_rows, val_rows, device):
    """(train_table, val_tokens, val_table) for a conditioned training run.
    Val tokens are rebuilt WITH per-token side-info so each token carries its
    chip's embedding."""
    from georvq.data.pool import fixed_val_tokens
    loc_cfg = cfg.model.conditioning.location
    enc = get_location_encoder(loc_cfg.encoder, device, loc_cfg.degree)
    standardize = loc_cfg.standardize
    train_t = build_location_table(train_rows, device, enc, standardize)
    val_t_chip = build_location_table(val_rows, device, enc, standardize)
    val_s, val_loc = fixed_val_tokens(store, val_rows,
                                      token=cfg.model.window,
                                      sat_table=val_t_chip.cpu())
    return train_t, val_s.to(device), val_loc.to(device)


def build_base(d_in, d_out, hidden=512, layers=1, bottleneck=0, pre_layers=0,
               pre_hidden=256, dropout=0.0):
    """Location-features -> latent-vector prior MLP. Structure:
        d_in -> [pre_hidden]*pre_layers -> [bottleneck] -> [hidden]*layers -> d_out
    bottleneck=0 -> flat MLP. A narrow bottleneck band-limits the
    location->latent map so it can't memorise individual chips; dropout>0
    regularizes the high-capacity flat variant."""
    def blk(i, o):
        return [nn.Linear(i, o), nn.GELU()] + \
            ([nn.Dropout(dropout)] if dropout else [])
    seq = []
    w = d_in
    for _ in range(pre_layers):
        seq += blk(w, pre_hidden); w = pre_hidden
    if bottleneck:
        seq += blk(w, bottleneck); w = bottleneck
    for _ in range(layers):
        seq += blk(w, hidden); w = hidden
    seq += [nn.Linear(w, d_out)]
    return nn.Sequential(*seq)


def load_coarse_prior(ckpt_path, out_dim, device):
    """A pretrained frozen location->coarse-latent prior (see
    scripts/pretrain_location_prior.py). Returns (module, feature_dim)."""
    bk = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = build_base(bk["sdim"], out_dim, bk["hidden"], bk["layers"],
                   bottleneck=bk.get("bottleneck", 0),
                   pre_layers=bk.get("pre_layers", 0),
                   pre_hidden=bk.get("pre_hidden", 256),
                   dropout=bk.get("dropout", 0.0)).to(device)
    m.load_state_dict(bk["base"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, bk["sdim"]
