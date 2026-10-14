"""Build models from configs; rebuild them exactly from checkpoints."""
from __future__ import annotations

import torch

from georvq.config import Config
from georvq.model.codec import PyramidCodec
from georvq.sources import get_source


def build_model(cfg: Config, device="cpu") -> PyramidCodec:
    """The one way to construct a model from a config.

    Also attaches `source` and `full_config` to the model, which is what
    lets a checkpoint rebuild itself and what every eval and inference
    path reads. Construct models through here rather than calling
    `PyramidCodec` directly, or those attributes go missing.
    """
    source = get_source(cfg.data.source, cfg.model)
    model = PyramidCodec(cfg.model, source).to(device)
    model.tokenizer.grad_ckpt = cfg.train.grad_ckpt
    model.decoder.grad_ckpt = cfg.train.grad_ckpt
    model.source = source
    model.full_config = cfg
    if cfg.train.compile and device != "cpu":
        compile_trunks(model)
    return model


def _decoder_blocks(decoder):
    """The per-layer trunk blocks of any decoder kind: a plain or tiersum
    decoder's `blocks`, every `DecoderBank` member's, and a ladder's stem
    plus each stage's blocks."""
    if hasattr(decoder, "members"):
        return [b for m in decoder.members for b in _decoder_blocks(m)]
    if hasattr(decoder, "stages"):
        return list(decoder.stem) + [b for st in decoder.stages for b in st]
    return list(decoder.blocks)


def compile_trunks(model):
    """torch.compile the per-layer forwards of both trunks. Compiling bound
    forwards (not the modules) keeps state-dict names untouched, so
    checkpoints stay loadable everywhere."""
    for layer in model.tokenizer.vit.layers:
        layer.forward = torch.compile(layer.forward)
    for blk in _decoder_blocks(model.decoder):
        blk.forward = torch.compile(blk.forward)


def from_checkpoint(path, device="cpu") -> PyramidCodec:
    """Rebuild the exact model from a checkpoint's embedded config and load
    its weights (strict). The returned model is in eval mode and carries the
    resolved config as `.full_config`."""
    from georvq.repro import load_checkpoint
    ck = load_checkpoint(path, map_location=device)
    if "config" not in ck:
        raise KeyError(
            f"{path}: no embedded config — for legacy research checkpoints "
            "use georvq.compat.from_legacy_checkpoint")
    from georvq.config import migrate
    cfg = Config.model_validate(migrate(ck["config"]))
    model = build_model(cfg, device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model
