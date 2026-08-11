"""FIXED evaluation module for the codebook challenge. DO NOT MODIFY.

Any edit to this file invalidates the run (it is hash-checked afterwards).

Usage from your own script:

    from challenge_eval import score_holdout, score_local

    stats = score_holdout([C0, C1, C2], note="my method")   # SUBMISSION
    stats = score_local([C0, C1, C2], vectors)              # free, local

Each Ci must be a float32/float64 array of shape (K, D) with K and D
exactly as reported by `spec()`. Codebooks are applied as a 3-level
residual quantizer: level 0 quantizes the vector, level 1 quantizes what
level 0 left behind, level 2 quantizes what level 1 left behind.

Reported metrics (both computed the same way in both functions):
  per-level index entropy (bits) and perplexity = exp(H_nats)
  sum_perplexity  -- THE OBJECTIVE, summed over the 3 levels
  residual        -- remaining energy / input energy (reported, not the
                     objective; recorded so the trade-off is visible)
"""
import hashlib
import json
import os
import time

import numpy as np

_HOLDOUT = "/data/tessera-samples/cb_private/holdout.npy"
_LOG = "/data/tessera-samples/cb_private/submission_log.jsonl"
K = 4096
DIM = 256
LEVELS = 3
MAX_SUBMISSIONS = 10


def spec():
    """Fixed problem parameters. K is set by the organiser and may not be
    changed; submissions with a different K or DIM are rejected."""
    return {"K": K, "DIM": DIM, "LEVELS": LEVELS,
            "objective": "maximise sum of per-level perplexity (max 3*K)",
            "target": "mean per-level entropy >= log2(K) - 1 = %.1f bits"
                      % (np.log2(K) - 1.0),
            "max_submissions": MAX_SUBMISSIONS}


def _check(codebooks):
    if len(codebooks) != LEVELS:
        raise ValueError(f"need exactly {LEVELS} codebooks, got {len(codebooks)}")
    out = []
    for i, C in enumerate(codebooks):
        C = np.asarray(C, dtype=np.float32)
        if C.shape != (K, DIM):
            raise ValueError(f"codebook {i}: shape {C.shape}, required "
                             f"{(K, DIM)} (K is fixed and may not change)")
        if not np.isfinite(C).all():
            raise ValueError(f"codebook {i}: contains non-finite values")
        out.append(C)
    return out


def _quantize(V, codebooks):
    """Sequential residual quantization; returns (per-level indices, resid)."""
    try:
        import torch
        dev = "cuda:1" if torch.cuda.is_available() else "cpu"
        r = torch.from_numpy(V).to(dev)
        idxs = []
        for C in codebooks:
            E = torch.from_numpy(C).to(dev)
            cn = E.pow(2).sum(1)
            parts = []
            for chunk in r.split(16384):
                d = (chunk.pow(2).sum(1, keepdim=True) - 2 * chunk @ E.t()
                     + cn[None])
                parts.append(d.argmin(1))
            a = torch.cat(parts)
            idxs.append(a.cpu().numpy())
            r = r - E[a]
        return idxs, float(r.pow(2).sum(1).mean()) / float(
            torch.from_numpy(V).to(dev).pow(2).sum(1).mean())
    except ImportError:
        r = V.copy()
        idxs = []
        for C in codebooks:
            d = ((r ** 2).sum(1)[:, None] - 2 * r @ C.T + (C ** 2).sum(1)[None])
            a = d.argmin(1)
            idxs.append(a)
            r = r - C[a]
        return idxs, float((r ** 2).sum(1).mean() / (V ** 2).sum(1).mean())


def _metrics(V, codebooks):
    codebooks = _check(codebooks)
    idxs, resid = _quantize(V, codebooks)
    per = []
    for a in idxs:
        c = np.bincount(a, minlength=K).astype(np.float64)
        p = c / c.sum()
        nz = p[p > 0]
        H = float(-(nz * np.log2(nz)).sum())
        per.append({"entropy_bits": H, "perplexity": float(2.0 ** H),
                    "used": int((c > 0).sum())})
    sum_perp = float(sum(l["perplexity"] for l in per))
    mean_H = float(np.mean([l["entropy_bits"] for l in per]))
    return {"per_level": per, "sum_perplexity": sum_perp,
            "max_sum_perplexity": float(LEVELS * K),
            "mean_entropy_bits": mean_H, "bits_from_full": float(np.log2(K) - mean_H),
            "residual": resid, "n_vectors": int(V.shape[0])}


def score_local(codebooks, vectors):
    """Free, unlimited scoring on WHATEVER vectors you pass (e.g. your own
    train/val split). Does not touch the holdout and is not logged."""
    return _metrics(np.asarray(vectors, dtype=np.float32), codebooks)


def score_holdout(codebooks, note=""):
    """SUBMISSION: score on the private holdout set. Logged and limited to
    MAX_SUBMISSIONS calls."""
    n = 0
    if os.path.exists(_LOG):
        with open(_LOG) as f:
            n = sum(1 for _ in f)
    if n >= MAX_SUBMISSIONS:
        raise RuntimeError(f"submission limit reached ({MAX_SUBMISSIONS})")
    V = np.load(_HOLDOUT)
    st = _metrics(V, codebooks)
    st["submission"] = n + 1
    st["note"] = str(note)[:200]
    st["codebook_sha"] = hashlib.sha256(
        b"".join(np.asarray(c, dtype=np.float32).tobytes()
                 for c in codebooks)).hexdigest()[:16]
    st["t"] = time.strftime("%H:%M:%S")
    with open(_LOG, "a") as f:
        f.write(json.dumps(st) + "\n")
    return st
