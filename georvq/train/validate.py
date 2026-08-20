"""Validation: reconstruction quality across the R-D front, codebook health,
and reference baselines."""
from __future__ import annotations

import json

import numpy as np
import torch

from georvq.config import TierSpec, parse_tier
from georvq.metrics import centered_cos
from georvq.model.codec import tuple_bits


def tname(ls):
    return "-".join(map(str, ls))


def canonical_tuples(specs, mins=None):
    """Named operating points for scalar logging: minimal / coarse-only /
    half-ladder / full. With min-depths training, every point is clamped
    into the TRAINED tuple space (untrained tuples mislead)."""
    mins = mins or [1] + [0] * (len(specs) - 1)

    def cascade(start):
        out, prev = [], start
        for sp in specs:
            L = min(prev, sp.depth); out.append(L); prev = L
        return out
    d0 = specs[0].depth
    half = [max(L // 2, m) for L, m in zip(cascade(d0), mins)]
    pts = {"min": list(mins),
           "coarse_only": [d0] + list(mins[1:]),
           "half": half, "full": cascade(d0)}
    return {k: v for k, v in pts.items()}


@torch.no_grad()
def validate(model, source, val_s, tuples, canon_names=(), val_loc=None,
             quantized=True):
    """val_s: raw val tokens resident on the eval device (converted per
    chunk). val_loc: per-token location embeddings when the model carries a
    conditioner/selector. Returns (scalars dict, cos-by-tuple dict)."""
    was_training = model.training
    model.eval()
    specs = model.specs
    out, cos_by_tuple, ccos_by_tuple = {}, {}, {}
    vchunk = max(64, 65536 // specs[-1].cells)
    chunks = []
    for i in range(0, val_s.shape[0], vchunk):
        x, v = source.prep(val_s[i:i + vchunk], val_s.device)
        chunks.append((x, v, val_loc[i:i + vchunk] if val_loc is not None else None))
    if not quantized:
        # VQ-free warmup: the codebooks hold whatever nn.Embedding was
        # initialised with, so every quantized tuple would be measuring the
        # model against random vectors -- 16 tuples x 4096 tokens of
        # quantization per validation, producing numbers that look like
        # results and are not. Only the unquantized ceiling means anything
        # until the books are fitted.
        ceil_sum = [0.0, 0, 0.0]
        bm_sum, bm_n = 0.0, 0
        for cx, cv, _ in chunks:
            o = model.eval_ceiling(cx, valid=cv)
            n = int(cv.sum())
            ceil_sum[0] += float(source.recon_stats(o, cx, cv)[1]["cos"]) * n
            ceil_sum[1] += n
            ceil_sum[2] += float(centered_cos(o, cx, cv)) * n
            vm = cv.unsqueeze(1).float()
            mean = (cx * vm).sum((2, 3)) / vm.sum((2, 3)).clamp_min(1)
            mean = torch.nn.functional.normalize(mean, dim=1)
            bm_sum += float((mean[:, :, None, None].expand_as(cx)
                             * cx).sum(1)[cv].sum()); bm_n += n
        out["cos_ceiling"] = ceil_sum[0] / ceil_sum[1]
        out["ccos_ceiling"] = ceil_sum[2] / ceil_sum[1]
        out["cos_blockmean"] = bm_sum / bm_n
        out["_ccos"] = {}
        if was_training:
            model.train()
        return out, {}

    sums = {tname(t): [0.0, 0, 0.0] for t in tuples}
    ceil_sum = [0.0, 0, 0.0]
    track_cd = model.conditioner is not None
    cd_sum = [np.zeros(sp.depth + 1) for sp in specs] if track_cd else None
    cd_n = 0
    for cx, cv, cloc in chunks:
        zs, cums = model.level_cumsums(cx, loc=cloc, valid=cv)
        if track_cd:       # per-tier cos-dist by cumulative level (0 = prior)
            for ti, sp in enumerate(specs):
                z = zs[ti].float(); b = z.shape[0]
                for L in range(sp.depth + 1):
                    cd_sum[ti][L] += (1 - torch.cosine_similarity(
                        z, cums[ti][:, :, L], -1).mean()).item() * b
            cd_n += zs[0].shape[0]
        n = int(cv.sum())
        for t in tuples:
            o = model.eval_tuple(cums, t, cx, valid=cv)
            c = source.recon_stats(o, cx, cv)[1]["cos"]
            s = sums[tname(t)]; s[0] += float(c) * n; s[1] += n
            if tname(t) in canon_names:
                s[2] += float(centered_cos(o, cx, cv)) * n
        o = model.eval_ceiling(cx, valid=cv)
        c = source.recon_stats(o, cx, cv)[1]["cos"]
        ceil_sum[0] += float(c) * n; ceil_sum[1] += n
        ceil_sum[2] += float(centered_cos(o, cx, cv)) * n
    for t in tuples:
        s = sums[tname(t)]
        cos_by_tuple[tname(t)] = s[0] / s[1]
        if tname(t) in canon_names:
            ccos_by_tuple[tname(t)] = s[2] / s[1]
    out["cos_ceiling"] = ceil_sum[0] / ceil_sum[1]
    out["ccos_ceiling"] = ceil_sum[2] / ceil_sum[1]
    out["_ccos"] = ccos_by_tuple

    # codebook health per tier/level (on one chunk)
    cx, cv, cloc = chunks[0]
    out.update(model.codebook_health(cx, loc=cloc))
    if model.selector is not None:
        out.update(model.selector.diagnostics(model, cx, cloc, cv))

    # block-mean baseline (accumulated per chunk)
    bm_sum, bm_n = 0.0, 0
    for x, v, _ in chunks:
        vm = v.unsqueeze(1).float()
        mean = (x * vm).sum((2, 3)) / vm.sum((2, 3)).clamp_min(1)
        mean = torch.nn.functional.normalize(mean, dim=1)
        bm = mean[:, :, None, None].expand_as(x)
        bm_sum += float((bm * x).sum(1)[v].sum()); bm_n += int(v.sum())
    out["cos_blockmean"] = bm_sum / bm_n
    if track_cd and cd_n:
        out["cosdist"] = [(cd_sum[ti] / cd_n).tolist()
                          for ti in range(len(specs))]
    if was_training:
        model.train()
    return out, cos_by_tuple


def load_rd_ref(spec_str):
    """'path|lvl,lvl|scale|label' -> (label, [(bits, cos) Pareto front]).
    scale is the reference run's window size (area-normalizes its bits onto
    a bits/pixel axis). The 3-part form 'points.json|method|label' reads a
    {method: [(bits, cos), ...]} points file."""
    parts = spec_str.split("|")
    if len(parts) == 3:
        path, method, label = parts
        pts = sorted(map(tuple, json.load(open(path))[method]))
    else:
        path, lvls, ref_window, label = parts
        ref_specs = [TierSpec(**parse_tier(t)) for t in lvls.split(",")]
        cbt = json.load(open(path))["cos_by_tuple"]
        px = float(ref_window) ** 2
        pts = sorted((tuple_bits(ref_specs, [int(v) for v in k.split("-")])
                      / px, c) for k, c in cbt.items())
    front, best = [], -1
    for b, c in pts:
        if c > best:
            front.append((b, c)); best = c
    return label, front
