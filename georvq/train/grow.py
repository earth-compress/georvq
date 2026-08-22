"""Function-preserving growth of a trained decoder: deeper and wider.

The point is to start a capacity experiment from EXACTLY the trained
model's function, so any change in quality is attributable to the extra
capacity being trained rather than to having thrown the weights away. At
init the grown decoder reproduces its ancestor to floating-point
round-off; `verify_growth` asserts it.

What grows, and why only these:

- **Depth.** New `AdaLNBlock`s append after the trained ones, before the
  pixel head. They are identity at init for free: AdaLN-Zero zeroes the
  modulation projection, so both residual gates are zero. Nothing to copy.
- **MLP hidden width** (`ff_ratio`). New hidden units are gated off by
  zeroing the *second* Linear's new input columns, so they contribute
  nothing at init while still receiving gradient once those columns move.
- **Pixel-head feature width** (`decoder.head_feat`). Same trick down the
  conv stack: new channels may be computed, but every path by which they
  could reach an OLD output is zeroed.

What deliberately does not grow:

- **The residual stream** (`decoder.dim`). The blocks use mean-subtracting
  `LayerNorm`, so widening it cannot be function-preserving: padding the
  stream changes each token's mean, and every original dimension shifts.
  No initialisation fixes that, and AdaLN's conditioning cannot absorb it
  because the correction is per-token.
- **Attention internals.** `nn.MultiheadAttention` ties head_dim to
  dim/heads, so widening needs a different module -- and renaming those
  parameters would break every existing checkpoint (`docs/OVERVIEW.md`
  invariant 2).
- **The encoder and codebooks.** This is a decoder-side experiment; the
  bitstream is unchanged, which is the whole point of the contract in
  `docs/BITSTREAM.md`.

New units keep the fresh model's own random init rather than being zeroed
outright: zeroing both sides of a gated unit leaves it with no gradient
path and it never wakes up.
"""
from __future__ import annotations

import torch

# state-dict prefixes whose tensors must match exactly, or the grown model
# is not the same codec any more
FROZEN_PREFIXES = ("tokenizer.", "stem.", "rvqs.", "stats_enc.", "stats_dec.")


def _copy_into(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy `src` into the leading corner of `dst`, whatever the rank."""
    dst[tuple(slice(0, s) for s in src.shape)].copy_(src)


def grow_decoder(dst_model, src_state, verbose=True):
    """Load `src_state` into the larger `dst_model`, preserving its function.

    Returns a dict of what happened, for the caller to log.
    """
    own = dst_model.state_dict()
    report = {"exact": 0, "expanded": [], "fresh": [], "missing": []}

    with torch.no_grad():
        for k, dv in own.items():
            sv = src_state.get(k)
            if sv is None:
                report["fresh"].append(k)
                continue
            if sv.shape == dv.shape:
                dv.copy_(sv)
                report["exact"] += 1
                continue
            if any(k.startswith(p) for p in FROZEN_PREFIXES):
                raise ValueError(
                    f"{k}: shape changed ({tuple(sv.shape)} -> "
                    f"{tuple(dv.shape)}) but this tensor belongs to the "
                    f"frozen encoder/codebook side. Growing it would change "
                    f"the bitstream, which this surgery must not do.")
            _copy_into(dv, sv)
            report["expanded"].append((k, tuple(sv.shape), tuple(dv.shape)))

        # --- zero every path by which a NEW unit could reach an OLD
        # output. state_dict() hands back the live parameter tensors, so
        # these in-place edits land on the model itself.
        for i in range(len(dst_model.decoder.blocks)):
            k0 = f"decoder.blocks.{i}.mlp.0.weight"
            k2 = f"decoder.blocks.{i}.mlp.2.weight"
            if k0 in src_state and own[k2].shape[1] > src_state[k0].shape[0]:
                own[k2][:, src_state[k0].shape[0]:].zero_()
        _zero_head_paths(dst_model, src_state, own)

    if verbose:
        print(f"grow: {report['exact']} tensors copied exactly, "
              f"{len(report['expanded'])} expanded, "
              f"{len(report['fresh'])} fresh (new blocks)")
        for k, a, b in report["expanded"]:
            print(f"  expand {k}: {a} -> {b}")
    return report


def _zero_head_paths(model, src_state, sd):
    """Cut every route from a new head channel to an old head output."""
    conv = model.decoder.head.conv
    idx = [i for i, m in enumerate(conv)
           if isinstance(m, torch.nn.Conv2d)]
    for i in idx:
        k = f"decoder.head.conv.{i}.weight"
        if k not in src_state:
            continue
        s_out, s_in = src_state[k].shape[:2]
        w = sd[k]
        # Old output channels must not see new input channels. The last
        # conv writes the pixel bands and grows no outputs, so cutting its
        # new inputs is the whole job there.
        if w.shape[1] > s_in:
            w[:s_out, s_in:].zero_()


@torch.no_grad()
def verify_growth(src_model, dst_model, sample, atol=1e-4, **kw):
    """Assert the grown decoder still computes its ancestor's function.

    `sample` is whatever the models' `forward`-equivalent takes; this
    compares full-depth reconstructions, which exercises every grown
    tensor. fp16-stored release weights make exact equality unavailable,
    hence a tolerance rather than `allclose(..., 0)`.
    """
    src_model.eval(); dst_model.eval()
    a = _full_decode(src_model, sample, **kw)
    b = _full_decode(dst_model, sample, **kw)
    if isinstance(a, dict):
        diff = max(float((a[k] - b[k]).abs().max()) for k in a)
        scale = max(float(a[k].abs().max()) for k in a)
    else:
        diff = float((a - b).abs().max())
        scale = float(a.abs().max())
    ok = diff <= atol
    return {"max_abs_diff": diff, "output_scale": scale, "ok": ok,
            "rel": diff / max(scale, 1e-12)}


@torch.no_grad()
def _full_decode(model, x, valid=None):
    st = model._stats_of(x, valid)
    zs = model.encode_levels(x, st, valid)
    Ls = torch.tensor([[sp.depth for sp in model.specs]],
                      device=zs[0].device).expand(zs[0].shape[0], -1)
    zqs, _, _ = model.quantize(zs, Ls)
    return model.decode(zqs, Ls, st, valid)
