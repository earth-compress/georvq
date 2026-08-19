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
    # Test the CODEBOOK WEIGHTS specifically. `k.startswith("rvqs")` also
    # matches each level's `step` and `initialized`, which are 0-dim
    # scalars and therefore shape-match ANY checkpoint -- so a model whose
    # books share nothing with the source (different K, different depth)
    # still reported "codebooks loaded" and got matured: resets disabled
    # and decay forced to 0.9995, silently overriding the config.
    loaded_cb = any(k.endswith("codebook.weight") and k in own
                    and own[k].shape == sd[k].shape for k in sd)
    if loaded_cb and getattr(model, "rvqs", None) is not None:
        mature_codebooks(model)
    print(f"warm-started from {path}: copied {copied}/{len(own)} tensors"
          f"{'' if loaded_cb else ' (codebooks FRESH -> resets stay ON)'}",
          flush=True)
    return copied, loaded_cb


def mature_codebooks(model, decay=0.9995):
    """Warm-started codebooks: freeze dead-code resets, slow the EMA.

    This OVERRIDES `vq.ema_decay` and disables `vq.reset_interval` /
    `vq.reset_mode` for the run. That is right when the books arrive
    already fitted and should barely move, and wrong when the point of
    the run is to keep them live -- set `vq.mature_warmstart: false`
    then. It is on by default, so any run with `train.init_from` gets it
    unless it opts out; the caller logs when it contradicts an explicit
    setting."""
    for rvq in model.rvqs:
        for lev in rvq.levels:
            lev.reset_interval = 1 << 30
            lev.decay = decay


def freeze_books(model):
    """Hard-freeze the codebooks: no EMA, no resets, no gradient.

    The EMA update in `VectorQuantizer.forward` is guarded by
    `self.training`, so putting the RVQ modules in eval() is a complete
    freeze without touching `vq.py` (they hold no dropout/norm state, so
    eval() changes nothing else). Codebook weights are also detached from
    the optimizer — an `nn.Embedding` weight would otherwise pick up
    weight decay and decay toward zero even with no loss gradient.

    Must be re-asserted after every `validate()` call: validate flips the
    whole model back to train() on its way out.
    """
    for rvq in model.rvqs:
        rvq.eval()
        for lev in rvq.levels:
            lev.reset_interval = 1 << 30
            lev.codebook.weight.requires_grad_(False)
    return sum(l.codebook.weight.numel() for rvq in model.rvqs
               for l in rvq.levels)


def decoder_tail_names(model, n_blocks):
    """Names of the last `n_blocks` decoder blocks plus the output head."""
    import re
    idx = {int(m.group(1)) for n, _ in model.named_parameters()
           if (m := re.match(r"decoder\.blocks\.(\d+)\.", n))}
    tail = {i for i in sorted(idx)[-n_blocks:]} if n_blocks > 0 else set()
    keep = set()
    for n, _ in model.named_parameters():
        m = re.match(r"decoder\.blocks\.(\d+)\.", n)
        if (m and int(m.group(1)) in tail) or \
                n.startswith(("decoder.head.", "decoder.norm_out.")):
            keep.add(n)
    return keep


def set_params_frozen(model, names, frozen: bool):
    """Freeze/unfreeze by name. Safe to call after the optimizer is built:
    the optimizer holds every trainable param from the start (see
    `trainable_params`), and AdamW skips params whose grad is None — so a
    param frozen at construction still trains once released."""
    for n, p in model.named_parameters():
        if n in names:
            p.requires_grad_(not frozen)
    return names


def trainable_params(model, predicate=None):
    """Params the optimizer should own for the WHOLE run.

    Deliberately ignores the current `requires_grad`: a param frozen only
    for the first N steps must already be in the optimizer, or unfreezing
    it later silently does nothing (it would never be stepped). Params
    that are frozen forever — the codebooks under `vq.freeze_books` — are
    excluded via `predicate`, so they never pick up weight decay."""
    out = []
    for n, p in model.named_parameters():
        if predicate is not None and not predicate(n, p):
            continue
        out.append((n, p))
    return out


def encoder_param_names(model):
    return set(n for n, _ in model.named_parameters()
               if n.startswith(ENCODER_PREFIXES))


def set_encoder_frozen(model, frozen: bool):
    names = encoder_param_names(model)
    for n, p in model.named_parameters():
        if n in names:
            p.requires_grad_(not frozen)
    return names
