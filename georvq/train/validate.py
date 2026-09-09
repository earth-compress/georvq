"""Validation: reconstruction quality across the R-D front, codebook health,
and reference baselines."""
from __future__ import annotations

import json

import numpy as np
import torch

from georvq.config import TierSpec, parse_tier
from georvq.model.codec import tuple_bits


def tname(ls):
    return "-".join(map(str, ls))


def canonical_tuples(specs, mins=None, path=False):
    """Named operating points for scalar logging: minimal / coarse-only /
    half-ladder / full. With min-depths training, every point is clamped
    into the TRAINED tuple space (untrained tuples mislead). With `path`
    (strict fill-order training), every named point must lie ON the path:
    half becomes the path's midpoint rather than the per-tier half
    cascade, which the path never visits."""
    mins = mins or [1] + [0] * (len(specs) - 1)

    def cascade(start):
        out, prev = [], start
        for sp in specs:
            L = min(prev, sp.depth); out.append(L); prev = L
        return out
    d0 = specs[0].depth
    if path:
        from georvq.model.codec import path_tuples
        p = path_tuples(specs, mins)
        pts = {"min": list(p[0]),
               "coarse_only": [d0] + [0] * (len(specs) - 1),
               "half": list(p[len(p) // 2]),
               "full": cascade(d0)}
    else:
        half = [max(L // 2, m) for L, m in zip(cascade(d0), mins)]
        pts = {"min": list(mins),
               "coarse_only": [d0] + list(mins[1:]),
               "half": half, "full": cascade(d0)}
    return {k: v for k, v in pts.items()}


@torch.no_grad()
def validate(model, source, val_s, tuples, canon_names=(), val_loc=None,
             quantized=True, spec=None):
    """val_s: raw val tokens resident on the eval device (converted per
    chunk). val_loc: per-token location embeddings when the model carries a
    conditioner/selector. `spec`: the run's resolved `ReportSpec`, which
    names the headline quality metric, the extra diagnostics and the
    zero-rate baselines -- nothing here knows what a cosine is.

    Returns (scalars dict, quality-by-tuple dict). Scalar keys are built
    from the metric keys, so an embedding field still publishes
    `cos_ceiling` / `cos_blockmean` while a scalar field publishes
    `r2_ceiling` and no block-mean at all.
    """
    if spec is None:        # ad-hoc callers (tests, scripts): unnarrowed
        spec = source.report(getattr(model, "cfg", None))
    q = spec.quality
    was_training = model.training
    model.eval()
    # sources with sampled prep (era5 @single draws a level per chip)
    # must serve the same slices every validation, or the val metric
    # jitters with the draw; the flag is a no-op for other sources
    source._prep_deterministic = True
    specs = model.specs
    out, q_by_tuple = {}, {}
    extra_by_tuple = {m.key: {} for m in spec.extra}
    vchunk = max(64, 65536 // specs[-1].cells)
    chunks = []
    for i in range(0, val_s.shape[0], vchunk):
        x, v = source.prep(val_s[i:i + vchunk], val_s.device)
        chunks.append((x, v, val_loc[i:i + vchunk] if val_loc is not None else None))

    def _baselines():
        """Zero-rate reference points, scored with the run's own metric."""
        for b in spec.baselines:
            tot, n_tot = 0.0, 0
            for x, v, _ in chunks:
                n = int(v.sum())
                tot += float(q(b.fn(x, v), x, v)) * n
                n_tot += n
            out[f"{q.key}_{b.key}"] = tot / max(n_tot, 1)

    if not quantized:
        # VQ-free warmup: the codebooks hold whatever nn.Embedding was
        # initialised with, so every quantized tuple would be measuring the
        # model against random vectors -- 16 tuples x 4096 tokens of
        # quantization per validation, producing numbers that look like
        # results and are not. Only the unquantized ceiling means anything
        # until the books are fitted.
        ceil = {m.key: [0.0, 0] for m in spec.metrics}
        for cx, cv, _ in chunks:
            o = model.eval_ceiling(cx, valid=cv)
            n = int(cv.sum())
            for m in spec.metrics:
                ceil[m.key][0] += float(m(o, cx, cv)) * n
                ceil[m.key][1] += n
        for k, (tot, n) in ceil.items():
            out[f"{k}_ceiling"] = tot / max(n, 1)
        _baselines()
        out["_extra"] = {}
        if was_training:
            model.train()
        source._prep_deterministic = False
        return out, {}

    sums = {tname(t): [0.0, 0] for t in tuples}
    esums = {m.key: {tname(t): [0.0, 0] for t in tuples} for m in spec.extra}
    ceil = {m.key: [0.0, 0] for m in spec.metrics}
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
            s = sums[tname(t)]; s[0] += float(q(o, cx, cv)) * n; s[1] += n
            # the extras are the expensive ones and only the canonical
            # operating points are ever read, so skip the rest
            if tname(t) in canon_names:
                for m in spec.extra:
                    e = esums[m.key][tname(t)]
                    e[0] += float(m(o, cx, cv)) * n; e[1] += n
        o = model.eval_ceiling(cx, valid=cv)
        for m in spec.metrics:
            ceil[m.key][0] += float(m(o, cx, cv)) * n
            ceil[m.key][1] += n
    for t in tuples:
        s = sums[tname(t)]
        q_by_tuple[tname(t)] = s[0] / s[1]
        for m in spec.extra:
            e = esums[m.key][tname(t)]
            if e[1]:
                extra_by_tuple[m.key][tname(t)] = e[0] / e[1]
    for k, (tot, n) in ceil.items():
        out[f"{k}_ceiling"] = tot / max(n, 1)
    out["_extra"] = extra_by_tuple

    # codebook health per tier/level (on one chunk)
    cx, cv, cloc = chunks[0]
    out.update(model.codebook_health(cx, loc=cloc))
    if model.selector is not None:
        out.update(model.selector.diagnostics(model, cx, cloc, cv))

    _baselines()
    if track_cd and cd_n:
        out["cosdist"] = [(cd_sum[ti] / cd_n).tolist()
                          for ti in range(len(specs))]
    if was_training:
        model.train()
    source._prep_deterministic = False
    return out, q_by_tuple


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
        blob = json.load(open(path))
        # runs before the per-source report layer wrote "cos_by_tuple"
        cbt = blob.get("quality_by_tuple") or blob["cos_by_tuple"]
        px = float(ref_window) ** 2
        pts = sorted((tuple_bits(ref_specs, [int(v) for v in k.split("-")])
                      / px, c) for k, c in cbt.items())
    front, best = [], -1
    for b, c in pts:
        if c > best:
            front.append((b, c)); best = c
    return label, front
