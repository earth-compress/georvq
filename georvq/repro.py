"""Run provenance: seeds, environment capture, and checkpoint writing.

Every checkpoint embeds the fully-resolved config (plus its hash, the git
state and library versions), so any model can be rebuilt exactly with
`georvq.from_checkpoint` and any two runs can be compared with
`georvq.config.config_diff`.
"""
from __future__ import annotations

import os
import subprocess

import torch

from georvq.config import Config, config_hash


def seed_all(seed: int):
    torch.manual_seed(seed)


def git_state(repo_dir=None) -> dict:
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
            "torch": torch.__version__}
    if torch.cuda.is_available():
        meta["cuda"] = torch.version.cuda
        meta["gpu"] = torch.cuda.get_device_name(0)
    return meta


def save_checkpoint(path, model, cfg: Config, step: int, metrics=None):
    torch.save({
        "model": model.state_dict(),
        "config": cfg.model_dump(mode="json"),
        "step": step,
        "metrics": metrics or {},
        **run_metadata(cfg),
    }, path)


def write_resolved_config(run_dir, cfg: Config):
    import yaml
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
