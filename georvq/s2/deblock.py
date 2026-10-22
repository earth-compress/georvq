"""Optional client-side seam deblocking of S2 v1 reads.

The deblocker (georvq.sources.s2_deblock.S2BoundaryDeblocker) is a small
network run after the decoder on the decoded 10, 20 and 60 m grids. It
is not part of the codec: stores, their quality arrays and the rung
tables describe the base decoder only, and nothing in a store changes
when a reader deblocks. Its weights are published as a separate file
(`DEBLOCKER_FILE`) in the S2 model repository next to the bundle,
together with the recipe it was trained with and the `bitstream_id`
and `decoder_id` of the codec whose decodes it was trained on.

The network standardises each frame over its own extent, so its output
depends on where the frames lie. `read_deblocked` therefore expands the
requested window to the frames of the product's frame grid (`frame` px
squares from the product's origin on the 10 m grid) that it touches,
reads them with a `halo` px margin clipped at the product's edge,
deblocks each frame on its core plus halo and crops back. A window read
this way equals the same window cut from a deblock of the whole
product. Only fixed rungs are supported, since the network is
conditioned on one fine-tier depth per frame.
"""
from __future__ import annotations

import numpy as np

from georvq.inference.store_codec import parse_rung

DEBLOCKER_FILE = "deblocker/deblocker.pt"
#: the 10 m grid's DN per unit of the network's input
SCALE = 10000.0


def load(path, device="cpu"):
    """(eval-mode net, entry) of a saved S2 deblocker; the entry holds
    `recipe`, `trained_against` and `provenance`."""
    from georvq.sources.s2_deblock import load_s2_deblocker
    return load_s2_deblocker(path, device)


def check(entry, header, force=False):
    """Refuse a deblocker trained against another codec than the one the
    store's header names, unless `force`."""
    want = entry.get("trained_against") or {}
    codec = header["codec"]
    diff = [k for k in ("bitstream_id", "decoder_id")
            if want.get(k) != codec.get(k)]
    if diff and not force:
        raise ValueError(
            "the deblocker was trained against "
            + ", ".join(f"{k} {want.get(k)}" for k in diff)
            + "; the store has "
            + ", ".join(f"{k} {codec.get(k)}" for k in diff)
            + ". Pass force=True to apply it anyway")


def read_deblocked(ds, deblocker, window=None, quality=None, force=False,
                   device="cpu"):
    """Deblocked pixels of every grid of a region of an S2 v1 store:
    {grid name: uint16 (count, rows, cols)}, as `ds.read_grids` returns.

    `ds` is a dataset from `georvq.open`; `deblocker` is (net, entry)
    from `load`, or a path to the saved file. `window` is
    ((row0, row1), (col0, col1)) on the 10 m grid, with every bound a
    multiple of 6 (one 60 m pixel), or None for the whole product.
    `quality` is a fixed rung such as "8-4" (None reads full depth);
    quality bounds are refused. Nodata pixels stay 0 and valid pixels
    are rounded and clipped to 1..65535."""
    import torch

    from georvq.sources.s2_deblock import SCALES
    if isinstance(deblocker, (str, bytes)) or hasattr(deblocker,
                                                      "__fspath__"):
        deblocker = load(deblocker, device)
    net, entry = deblocker
    check(entry, ds.header, force)
    frame, halo = entry["recipe"]["frame"], entry["recipe"]["halo"]
    if isinstance(quality, str) and "<" in quality:
        raise ValueError(f"quality {quality!r}: the deblocker needs a fixed "
                         "rung such as '8-4'")
    rungs = [tuple(r["tuple"]) for r in ds.header["rungs"]]
    rung = max(rungs) if quality is None else parse_rung(quality)
    if rung not in rungs:
        raise ValueError(f"rung {rung} is not one of the store's {rungs}")
    H, W = ds.product_height, ds.product_width
    if window is None:
        r0, r1, c0, c1 = 0, H, 0, W
    else:
        (r0, r1), (c0, c1) = window
    if any(v % 6 for v in (r0, r1, c0, c1)):
        raise ValueError("window bounds must be multiples of 6 px (one "
                         "60 m pixel) for a deblocked read")
    if not (0 <= r0 < r1 <= H and 0 <= c0 < c1 <= W):
        raise ValueError(f"window rows {r0}:{r1} cols {c0}:{c1} outside "
                         f"the {H}x{W} product")
    # the frames the window touches, and the region they read
    fr0, fc0 = r0 // frame * frame, c0 // frame * frame
    fr1, fc1 = min(H, -(-r1 // frame) * frame), min(W, -(-c1 // frame) * frame)
    gr0, gc0 = max(0, fr0 - halo), max(0, fc0 - halo)
    gr1, gc1 = min(H, fr1 + halo), min(W, fc1 + halo)
    dn = ds.read_grids(((gr0, gr1), (gc0, gc1)), quality=rung)
    valid = {g: (a != 0).any(0) for g, a in dn.items()}
    x = {g: torch.from_numpy(a.astype(np.float32) / SCALE)[None].to(device)
         for g, a in dn.items()}
    Ls = torch.tensor([rung], dtype=torch.long, device=device)
    out = {g: v.clone() for g, v in x.items()}

    def cut(t, a0, a1, b0, b1):
        return {g: v[:, :, a0 // SCALES[g]:a1 // SCALES[g],
                     b0 // SCALES[g]:b1 // SCALES[g]] for g, v in t.items()}

    with torch.no_grad():
        for a0 in range(fr0, fr1, frame):
            for b0 in range(fc0, fc1, frame):
                a1, b1 = min(a0 + frame, H), min(b0 + frame, W)
                i0, j0 = max(0, a0 - halo), max(0, b0 - halo)
                i1, j1 = min(H, a1 + halo), min(W, b1 + halo)
                y = net(cut(x, i0 - gr0, i1 - gr0, j0 - gc0, j1 - gc0), Ls)
                core = cut(y, a0 - i0, a1 - i0, b0 - j0, b1 - j0)
                for g in out:
                    s = SCALES[g]
                    out[g][:, :, (a0 - gr0) // s:(a1 - gr0) // s,
                           (b0 - gc0) // s:(b1 - gc0) // s] = core[g]
    res = {}
    for g, t in out.items():
        s = SCALES[g]
        a = t[0].cpu().numpy() * SCALE
        a = np.where(valid[g], np.clip(np.rint(a), 1, 65535), 0)
        sl = (slice(None), slice((r0 - gr0) // s, (r1 - gr0) // s),
              slice((c0 - gc0) // s, (c1 - gc0) // s))
        res[g] = a.astype(np.uint16)[sl]
    return res
