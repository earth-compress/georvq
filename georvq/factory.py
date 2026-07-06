"""Build models from configs; rebuild them exactly from checkpoints."""
from __future__ import annotations

import torch

from georvq.config import Config
from georvq.model.codec import PyramidCodec
from georvq.sources import get_source


def build_model(cfg: Config, device="cpu") -> PyramidCodec:
    source = get_source(cfg.data.source, cfg.model)
    model = PyramidCodec(cfg.model, source).to(device)
    model.tokenizer.grad_ckpt = cfg.train.grad_ckpt
    model.source = source
    model.full_config = cfg
    return model


def from_checkpoint(path, device="cpu") -> PyramidCodec:
    """Rebuild the exact model from a checkpoint's embedded config and load
    its weights (strict). The returned model is in eval mode and carries the
    resolved config as `.full_config`."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ck:
        raise KeyError(
            f"{path}: no embedded config — for legacy research checkpoints "
            "use georvq.compat.from_legacy_checkpoint")
    cfg = Config.model_validate(ck["config"])
    model = build_model(cfg, device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model
