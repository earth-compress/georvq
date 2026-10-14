"""Build models from configs; rebuild them exactly from checkpoints."""
from __future__ import annotations

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
    return model


def from_checkpoint(path, device="cpu") -> PyramidCodec:
    """Rebuild the exact model from a checkpoint's embedded config and load
    its weights (strict). The returned model is in eval mode and carries the
    resolved config as `.full_config` and the run's prep as `.prep`."""
    from georvq.repro import load_checkpoint
    ck = load_checkpoint(path, map_location=device)
    if "config" not in ck:
        raise KeyError(
            f"{path}: no embedded config — for legacy research checkpoints "
            "use georvq.compat.from_legacy_checkpoint")
    from georvq.config import migrate
    raw = migrate(ck["config"])
    cfg = Config.model_validate(raw)
    cfg = bank_heads_as_built(cfg, raw, ck["model"], path)
    model = build_model(cfg, device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    attach_prep(model, ck.get("prep"))
    return model


def bank_heads_as_built(cfg, raw, state_dict, path=""):
    """Before 2026-10-14 every decoder-bank member's pixel head was built
    with `decoder.head_feat`, whatever its own `head_feat` said. For a
    member whose checkpointed head shapes match that width and not the
    configured one, return a config with the member's head_feat set to
    the width it was built with, so the checkpoint loads strictly."""
    bank = cfg.model.decoder_bank
    want = cfg.model.decoder.head_feat
    if not bank or all(st.head_feat in (None, want) for st in bank):
        return cfg
    import copy
    import warnings
    old_raw = copy.deepcopy(raw)
    for st in old_raw["model"]["decoder_bank"]:
        st["head_feat"] = want
    built = {"now": build_model(cfg, "cpu").state_dict(),
             "old": build_model(Config.model_validate(old_raw),
                                "cpu").state_dict()}

    def heads(sd, i):
        pre = f"decoder.members.{i}.head."
        return {k: tuple(v.shape) for k, v in sd.items() if k.startswith(pre)}
    fixed = []
    for i, st in enumerate(bank):
        ck = heads(state_dict, i)
        if (st.head_feat not in (None, want)
                and ck != heads(built["now"], i)
                and ck == heads(built["old"], i)):
            raw["model"]["decoder_bank"][i]["head_feat"] = want
            fixed.append((i, st.head_feat))
    if not fixed:
        return cfg
    warnings.warn(
        f"{path}: decoder_bank members {[i for i, _ in fixed]} were built "
        f"with decoder.head_feat={want}, not their configured widths "
        f"{[h for _, h in fixed]}; loading them as built", stacklevel=3)
    return Config.model_validate(raw)


def attach_prep(model, state=None):
    """Rebuild the run's named prep (`cfg.data.prep`) as `model.prep`, and
    restore any constants the checkpoint embedded for it. Eval paths prep
    through this, so a checkpoint trained under a named prep is scored in
    the model space it was trained in."""
    from georvq.sources.prep import build_prep
    cfg = model.full_config
    model.prep = build_prep(cfg.data.prep.name, model.source, cfg.model,
                            params=cfg.data.prep.params)
    model.prep.load_state(state or {})
    return model.prep
