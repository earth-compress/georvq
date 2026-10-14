"""Load legacy research-format checkpoints (argv-style, pre-Config).

Those checkpoints store `{"model": state_dict, "args": <flat argv dict>,
"step": int}`. This module translates the argv dict into a georvq `Config`
and remaps state-dict key prefixes onto the georvq module tree, so existing
trained models load strict — no retraining needed.
"""
from __future__ import annotations

from georvq.config import (Config, RateConfig, SigmaConfig)
from georvq.factory import build_model

# ordered: first matching prefix wins
_RENAMES = [
    ("vit_patches.", "tokenizer.patches."),
    ("vit_cls", "tokenizer.cls"),
    ("vit_pos", "tokenizer.pos"),
    ("vit_out.", "tokenizer.out."),
    ("vit.", "tokenizer.vit."),
    ("attn_decoder.to_patch.", "decoder.head.to_patch."),
    ("attn_decoder.head.", "decoder.head.conv."),
    ("attn_decoder.", "decoder."),
    ("rate_models.", "rate.models."),
    ("rate_ctx_proj.", "rate.ctx_proj."),
    ("rvqs.", "rvqs."),
]
# legacy keys with no georvq counterpart (dropped paths)
_DROP = ("swin_mask",)


def config_from_legacy_args(a: dict) -> Config:
    """Translate a legacy flat argv dict into a validated georvq Config."""
    if not a.get("vit_enc") or not a.get("attn_dec_dim"):
        raise ValueError("legacy conv-trunk/FPN checkpoints are not "
                        "supported (only ViT + attention-decoder models)")
    if a.get("cascade") or a.get("satclip_select"):
        raise NotImplementedError(
            "legacy cascade/select checkpoints are not supported: their "
            "conditioning arguments have no translation to georvq configs")

    stage1 = "sigma_max" in a
    model = {
        "window": a["window"],
        "bands": a.get("bands", 64),
        "tiers": list(a["levels"]),
        "min_depths": a.get("min_depths"),
        "full_frac": a.get("full_frac", 0.0),
        "ff_ratio": a.get("ff_ratio", 4.0),
        "tokenizer": {"dim": a.get("vit_dim", 256), "depth": a["vit_enc"]},
        "decoder": {"dim": a["attn_dec_dim"],
                    "depth": a.get("attn_dec_depth", 8),
                    "heads": a.get("attn_dec_heads", 8),
                    "head_feat": a.get("attn_dec_head_feat", 64),
                    "cls_bridge": a.get("cls_bridge", False)},
    }
    if stage1:
        model["vq"] = None
        model["sigma"] = SigmaConfig(max=a["sigma_max"],
                                     freqs=a.get("sigma_freqs", 8)).model_dump()
    else:
        model["vq"] = {
            "commit_weight": a.get("commit_weight", 0.25),
            "dead_threshold": a.get("dead_threshold", 0.01),
            "ema_decay": a.get("ema_decay", 0.99),
            "ema_decay_final": a.get("ema_decay_final"),
            "decay_ramp": tuple(a.get("decay_ramp", (2000, 6000))),
            "reset_freeze_frac": a.get("reset_freeze_frac", 0.9),
            "warmup_steps": a.get("vq_warmup", 0),
            "kmeans_sample": a.get("kmeans_sample", 131072),
            "kmeans_iters": a.get("kmeans_iters", 6),
            "bf16_scores": a.get("vq_bf16", False),
            "skip_masked": a.get("vq_skip_masked", False),
            "grad_codebook": a.get("grad_codebook", False),
        }
    if a.get("rate_model"):
        model["rate"] = RateConfig(
            model_dim=a.get("rate_model_dim", 256),
            blocks=a.get("rate_blocks", 4),
            ctx_dim=a.get("rate_ctx_dim", 96),
            weight=a.get("rate_weight", 0.005),
            warmup=a.get("rate_warmup", 4000),
            batch=a.get("rate_batch", 64),
            tau=a.get("rate_tau", 0.5),
            lr=a.get("rate_lr", 5e-4)).model_dump()

    raw = {
        "data": {
            "source": "ae",
            "store_dir": a.get("chips_dir", "/data/AE-samples/chips_1024"),
            "val_frac": a.get("val_frac", 0.04),
            "split_seed": a.get("seed", 0),
            "pool": {"size": a.get("pool_size", 192),
                     "workers": a.get("pool_workers", 2),
                     "slot_rows": a.get("pool_slot_rows") or None,
                     "refresh_per_s": a.get("pool_refresh", 2.0),
                     "device": a.get("pool_device", "cuda"),
                     "cycle": a.get("pool_cycle", False)},
        },
        "model": model,
        "train": {
            "steps": a.get("steps", 20000),
            "batch_windows": a.get("batch_tokens", 512),
            "amp": a.get("amp", True),
            "grad_ckpt": a.get("grad_ckpt", False),
            "tf32": a.get("tf32", True),
            "init_from": a.get("init_from") or None,
            "freeze_encoder_steps": a.get("freeze_encoder_steps", 0),
            "optim": {"lr": a.get("lr", 5e-4), "wd": a.get("wd", 0.01),
                      "warmup": a.get("warmup", 500)},
            "seed": a.get("seed", 0),
            "val_interval": a.get("val_interval", 1000),
            "fig_interval": a.get("fig_interval", 2000),
            "save_interval": a.get("save_interval", 2500),
            "val_tuples": a.get("val_tuples", 16),
        },
        "eval": {"recon_tuples": a.get("recon_tuples"),
                 "rd_refs": a.get("rd_ref", [])},
        "log": {"run_tag": a.get("run_tag", "legacy"),
                "wandb": not a.get("no_wandb", False)},
    }
    return Config.model_validate(raw)


def remap_state_dict(sd: dict, strict=True) -> dict:
    """Rename a legacy research-repo state dict onto today's modules.

    Strict by default, and deliberately so: an unrecognized key usually
    means the checkpoint has a module this code no longer builds, and
    silently dropping it would load a partially-initialized model that
    reports plausible-looking garbage. Pass `strict=False` only when
    you intend to discard the unmapped half.
    """
    out = {}
    dropped = []
    for k, v in sd.items():
        if k in _DROP:
            continue
        for old, new in _RENAMES:
            if k.startswith(old):
                out[new + k[len(old):]] = v
                break
        else:
            if strict:
                raise KeyError(f"unrecognized legacy state-dict key: {k}")
            dropped.append(k)
    if dropped:
        print(f"legacy remap: dropped {len(dropped)} unmapped keys "
              f"(e.g. {dropped[0]})", flush=True)
    return out


def from_legacy_checkpoint(path, device="cpu"):
    """Legacy checkpoint -> (georvq model in eval mode, Config)."""
    from georvq.repro import load_checkpoint
    ck = load_checkpoint(path, map_location=device)
    cfg = config_from_legacy_args(ck["args"])
    model = build_model(cfg, device)
    model.load_state_dict(remap_state_dict(ck["model"]), strict=True)
    model.eval()
    return model, cfg
