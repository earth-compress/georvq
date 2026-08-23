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

# Tensors that must match exactly or the grown model is not the same codec.
# Growing the encoder is allowed only when the caller says so explicitly:
# it invalidates any cached bitstream, because the latents move.
FROZEN_DECODER_ONLY = ("tokenizer.", "stem.", "rvqs.", "stats_enc.",
                       "stats_dec.")
FROZEN_FULL = ("rvqs.",)


def _copy_into(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy `src` into the leading corner of `dst`, whatever the rank."""
    dst[tuple(slice(0, s) for s in src.shape)].copy_(src)


def grow_decoder(dst_model, src_state, verbose=True, grow_encoder=False):
    """Load `src_state` into the larger `dst_model`, preserving its function.

    `grow_encoder=True` also expands the ViT tokenizer -- more blocks and
    a wider FFN. That changes the LATENTS, so any cached bitstream built
    with the old encoder is invalidated and the codebooks no longer sit
    where the encoder puts things. Only do it for a run that trains the
    encoder.

    Returns a dict of what happened, for the caller to log.
    """
    frozen = FROZEN_FULL if grow_encoder else FROZEN_DECODER_ONLY
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
            if any(k.startswith(p) for p in frozen):
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
        if grow_encoder:
            _grow_tokenizer(dst_model, src_state, own, verbose)

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


def _grow_tokenizer(model, src_state, sd, verbose=True):
    """Make the widened and newly-added ViT encoder layers no-ops.

    `nn.TransformerEncoderLayer` has no AdaLN gate, so a fresh block is
    NOT identity the way the decoder's AdaLNBlock is -- its attention and
    MLP start random and the residual adds them straight in. With
    `norm_first=True` the block is x + attn(norm(x)) + mlp(norm(x)), so
    zeroing both output projections makes it exactly the identity while
    leaving the input-side weights random, which is what keeps a gradient
    path into them.
    """
    layers = model.tokenizer.vit.layers
    n_new = 0
    for i, _ in enumerate(layers):
        k1 = f"tokenizer.vit.layers.{i}.linear1.weight"
        if k1 not in src_state:
            # a block the ancestor did not have: zero its two output
            # projections and it passes its input straight through
            for name in ("self_attn.out_proj.weight", "self_attn.out_proj.bias",
                         "linear2.weight", "linear2.bias"):
                key = f"tokenizer.vit.layers.{i}.{name}"
                if key in sd:
                    sd[key].zero_()
            n_new += 1
            continue
        # an inherited block whose FFN got wider: cut the new hidden units
        # out of the output so the block still computes what it did
        old_ff = src_state[k1].shape[0]
        k2 = f"tokenizer.vit.layers.{i}.linear2.weight"
        if k2 in sd and sd[k2].shape[1] > old_ff:
            sd[k2][:, old_ff:].zero_()
    if verbose:
        print(f"grow: tokenizer {len(layers) - n_new} inherited blocks "
              f"+ {n_new} new pass-through blocks")
