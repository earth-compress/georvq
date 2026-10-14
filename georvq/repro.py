"""Run provenance: seeds, environment capture, and checkpoint writing.

Every checkpoint embeds the fully-resolved config (plus its hash, the git
state and library versions), so any model can be rebuilt exactly with
`georvq.from_checkpoint` and any two runs can be compared with
`georvq.config.config_diff`.
"""
from __future__ import annotations

import os
import random
import subprocess

import numpy as np
import torch

from georvq.config import Config, config_hash


def seed_all(seed: int):
    """Seed torch, numpy's global generator and Python's `random`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _checkpoint_safe_globals():
    """Non-tensor types that existing checkpoints contain, all of them
    plain values: `TorchVersion` (the `torch` metadata field before it
    was stored as a str) and numpy float scalars (the lr in optimizer
    and scheduler state, from the numpy cosine schedule)."""
    from torch.torch_version import TorchVersion
    core = getattr(np, "_core", None) or np.core
    return [TorchVersion, core.multiarray.scalar, np.dtype,
            type(np.dtype(np.float64)), type(np.dtype(np.float32)),
            type(np.dtype(np.int64))]


def load_checkpoint(path, map_location="cpu"):
    """`torch.load` with `weights_only=True`, so loading a file cannot run
    pickled code. Accepts every georvq checkpoint, including those
    written before the torch version was stored as a string."""
    with torch.serialization.safe_globals(_checkpoint_safe_globals()):
        return torch.load(path, map_location=map_location, weights_only=True)


def git_state(repo_dir=None) -> dict:
    """The repo's commit and whether the tree was dirty, or Nones.

    Never raises: provenance is metadata, and a checkpoint that cannot
    be saved because git is missing (or the code was installed as a
    wheel, or the run is in a container) would be a worse outcome than
    one whose provenance is unknown.
    """
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir,
            stderr=subprocess.DEVNULL).decode().strip())
        return {"sha": sha, "dirty": dirty}
    except Exception:
        return {"sha": None, "dirty": None}


def run_metadata(cfg: Config) -> dict:
    meta = {"config_hash": config_hash(cfg), "git": git_state(),
            "torch": str(torch.__version__)}
    if torch.cuda.is_available():
        meta["cuda"] = torch.version.cuda
        meta["gpu"] = torch.cuda.get_device_name(0)
    return meta


def save_checkpoint(path, model, cfg: Config, step: int, metrics=None,
                    opt=None, sched=None, prep_state=None):
    """Write a checkpoint that can rebuild its own model.

    The resolved config travels with the weights, which is what makes
    `georvq.from_checkpoint` exact and `config_diff` possible between
    any two runs. Nothing here is optional: a checkpoint without its
    config is a tensor dump.

    `opt`/`sched` add optimizer and scheduler state for resume. A resume
    without them builds a fresh AdamW whose zero second moments give
    every parameter full-lr updates for thousands of steps -- the
    measured trigger of the tessera gradient storms -- so the periodic
    `model.pt` should carry them. Additive keys: older checkpoints and
    readers are unaffected.
    """
    d = {
        "model": model.state_dict(),
        "config": cfg.model_dump(mode="json"),
        "step": step,
        "metrics": metrics or {},
        **run_metadata(cfg),
    }
    if opt is not None:
        d["opt"] = opt.state_dict()
    if sched is not None:
        d["sched"] = sched.state_dict()
    if prep_state:
        # the fitted prep constants (model-space definition) travel with
        # the weights, so decode never depends on the artifact file
        d["prep"] = prep_state
    torch.save(d, path)


def write_resolved_config(run_dir, cfg: Config):
    import yaml
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
