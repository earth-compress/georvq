"""Warm-starting and freeze schedules.

`load_partial` copies every matching-shape tensor from a source checkpoint
(georvq or legacy format) into a freshly-built model — so a checkpoint with
different codebook sizes, extra modules (rate model), or a missing sigma
branch still warm-starts everything that lines up. Warm-started codebooks
are "matured": dead-code resets off, slower EMA.
"""
from __future__ import annotations

import torch

ENCODER_PREFIXES = ("tokenizer.", "stem.")


def load_partial(model, path, device):
    """Shape-reconciled warm start. Returns (n_copied, codebooks_loaded)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"]
    if "config" not in ck:                      # legacy checkpoint naming
        from georvq.compat import remap_state_dict
        sd = remap_state_dict(sd, strict=False)
    own = model.state_dict()
    copied = 0
    with torch.no_grad():
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v); copied += 1
            elif (k in own and k.endswith("to_patch.weight")
                  and own[k].shape[0] == v.shape[0]):
                # same pixel-patch output, wider ancestor input: zero-pad;
                # any other head geometry change starts fresh
                own[k].zero_()
                own[k][:, :v.shape[1]].copy_(v)
                copied += 1
    model.load_state_dict(own)
    loaded_cb = any(k.startswith("rvqs") and k in own
                    and own[k].shape == sd[k].shape for k in sd)
    if loaded_cb and getattr(model, "rvqs", None) is not None:
        mature_codebooks(model)
    print(f"warm-started from {path}: copied {copied}/{len(own)} tensors"
          f"{'' if loaded_cb else ' (codebooks FRESH -> resets stay ON)'}",
          flush=True)
    return copied, loaded_cb


def mature_codebooks(model, decay=0.9995):
    """Warm-started codebooks: freeze dead-code resets, slow the EMA."""
    for rvq in model.rvqs:
        for lev in rvq.levels:
            lev.reset_interval = 1 << 30
            lev.decay = decay


def encoder_param_names(model):
    return set(n for n, _ in model.named_parameters()
               if n.startswith(ENCODER_PREFIXES))


def set_encoder_frozen(model, frozen: bool):
    names = encoder_param_names(model)
    for n, p in model.named_parameters():
        if n in names:
            p.requires_grad_(not frozen)
    return names
