"""Shared codec load + reconstruct helpers for the A/B and rate-sweep evals.

A target pixel is reconstructed by sub-tiling its stored context crop into the
checkpoint's native window, encoding once, and decoding at one (or many) RVQ
rate tuples. For the rate sweep we encode each window once and decode every
tuple from the cached cumulative codes -- so sweeping the Pareto front costs one
encode pass plus a cheap decode per rate.
"""
from __future__ import annotations

import numpy as np
import torch

from georvq.eval.extract import dequant_np, l2norm_np
from georvq.model.codec import monotone_tuples, tuple_bits
from georvq.sources.ae import dequant


def load_codec(ckpt_path, device):
    """Model from a checkpoint (embedded-config or legacy format), in eval
    mode. Returns (model, window, specs)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" in ck:
        from georvq.factory import from_checkpoint
        model = from_checkpoint(ckpt_path, device)
    else:
        from georvq.compat import from_legacy_checkpoint
        model, _ = from_legacy_checkpoint(ckpt_path, device)
    return model, model.window, model.specs


def orig_embeddings(windows, t, source="ae"):
    """Published embedding at each target pixel (L2-normed), (N, bands).
    `source` picks the raw-window dequant: AE = signed-square int8; tessera
    = q * fp16-scale (scale bytes in the 2 trailing bands)."""
    wi, r, c = t.window_idx.to_numpy(), t.r.to_numpy(), t.c.to_numpy()
    px = windows[wi, :, r, c]                        # (N, raw_bands)
    if source == "tessera":
        q = px[:, :-2].astype(np.float32)
        scale = np.ascontiguousarray(px[:, -2:]).view(np.uint8) \
            .view(np.float16).astype(np.float32)
        return l2norm_np(q * scale)
    return l2norm_np(dequant_np(px))


def unique_windows(windows, t, W):
    """Sub-tile each target's crop to the codec window. Returns the unique
    (raw_bands,W,W) int8 windows, the per-target window index `inv`, and
    per-target local pixel offsets (lr, lc) inside that window."""
    crop = windows.shape[-1]
    assert crop % W == 0 and crop >= W, f"crop {crop} not a multiple of W {W}"
    wi, r, c = t.window_idx.to_numpy(), t.r.to_numpy(), t.c.to_numpy()
    sr, sc, lr, lc = r // W, c // W, r % W, c % W
    key = np.stack([wi, sr, sc], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    win = np.empty((len(uniq), windows.shape[1], W, W), np.int8)
    for k in range(len(uniq)):
        w, a, b = uniq[k]
        win[k] = windows[w][:, a * W:(a + 1) * W, b * W:(b + 1) * W]
    return win, inv, lr, lc


@torch.no_grad()
def reconstruct_tuples(model, win_int8, inv, lr, lc, tuples_ls, device,
                       chunk=256, source="ae"):
    """Reconstruct every target pixel at each rate tuple. Encodes each window
    once; decodes all tuples from the cached cumulative codes. Returns a list
    (aligned to tuples_ls) of (N, bands) L2-normed reconstructions."""
    N, bands = len(inv), model.cfg.bands
    out = [np.empty((N, bands), np.float32) for _ in tuples_ls]
    for a in range(0, len(win_int8), chunk):
        b = min(a + chunk, len(win_int8))
        raw = torch.from_numpy(win_int8[a:b])
        if source == "ae":
            # historical AE eval path: plain dequant, no re-norm/mask
            x = dequant(raw.to(device))
        else:
            x, _ = model.source.prep(raw, device)    # exact training prep
        _, cums = model.level_cumsums(x)
        mask = (inv >= a) & (inv < b)
        idx = np.nonzero(mask)[0]
        loc, lrm, lcm = inv[mask] - a, lr[mask], lc[mask]
        for j, ls in enumerate(tuples_ls):
            o = model.eval_tuple(cums, ls, x).float().cpu().numpy()
            out[j][idx] = l2norm_np(o[loc, :, lrm, lcm])
    return out


def select_front_tuples(specs, min_depths, n):
    """Pick ~n monotone rate tuples spread (geometrically in bits) across the
    operating range the model was trained for (>= per-level min depths)."""
    md = min_depths or [1] + [0] * (len(specs) - 1)
    ts = [t for t in monotone_tuples(specs)
          if all(t[i] >= md[i] for i in range(len(specs)))]
    ts.sort(key=lambda t: tuple_bits(specs, t))
    if len(ts) <= n:
        return ts
    bits = np.array([tuple_bits(specs, t) for t in ts])
    chosen, used = [], set()
    for tb in np.geomspace(bits[0], bits[-1], n):
        j = int(np.argmin(np.abs(bits - tb)))
        if j not in used:
            used.add(j)
            chosen.append(ts[j])
    return chosen
