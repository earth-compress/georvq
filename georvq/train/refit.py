"""Periodic offline refit phases: pause training, re-fit the codebooks on a
large fresh sample of encoder latents, resume.

Replaces the online EMA + dead-code-reset scheme, which had three defects
we measured rather than guessed at:

  * EMA usage is a lagging, batch-sized estimate. A code's fate was decided
    by ~2 assignments per step at K=8192, smoothed over ~100 steps.
  * All 24 (tier, level) books were reset against a SINGLE stale snapshot,
    so the residual each level was judged on had already been invalidated
    by the level above being repaired in the same instant.
  * Revived codes were teleported to the source cell's worst-fitting
    point -- an outlier by construction. The instantaneous residual drop
    was large (91% on tier1 L8) and ~100% of it evaporated before the next
    reset event, every time, for 23000 steps.

This phase fixes all three. Usage and distortion come from ~100 vectors
per codeword rather than an EMA; levels are updated STRICTLY IN ORDER so
level l+1 is fitted on the residual that level l's new book actually
produces; and splits place the pair at centroid +/- eps along the cell's
principal direction (classical LBG), which is a viable pair rather than
one codeword sitting on an outlier.

Books are otherwise static -- `vq.freeze_books` holds them still between
phases, so all book movement is deliberate and happens here.
"""
from __future__ import annotations

import torch

from georvq.model.vq import _chunked_assign


@torch.no_grad()
def collect_latents(model, source, pool, cfg, device, per_code=100,
                    max_batches=4096):
    """Encoder-only pass: gather ~`per_code` vectors per codeword, per tier.

    Per-tier budgets, because the tiers are coupled through batch count:
    the coarse tier yields 1 vector per window and the finest 256, so a
    batch budget that feeds the coarse tier would materialise tens of GiB
    for the finest. Fine tiers are subsampled per batch instead.
    """
    specs = cfg.model.tiers
    want = [per_code * sp.codes for sp in specs]
    got = [0] * len(specs)
    bufs = [[] for _ in specs]
    batch = cfg.train.batch_windows
    align = cfg.data.pool.align_crops
    nb = 0
    while any(g < w for g, w in zip(got, want)) and nb < max_batches:
        s = pool.sample_batch(batch, cfg.model.window, align=align)
        x, _ = source.prep(s, device)
        for i, z in enumerate(model.encode_levels(x)):
            if got[i] >= want[i]:
                continue
            zf = z.reshape(-1, z.shape[-1]).float()
            zf = zf[:want[i] - got[i]]
            bufs[i].append(zf)
            got[i] += zf.shape[0]
        nb += 1
    return [torch.cat(b) for b in bufs], nb


@torch.no_grad()
def _cell_stats(r, C, chunk=1 << 15):
    """Assignment, per-code count and per-code total squared error."""
    K = C.shape[0]
    a = _chunked_assign(r, C)
    se = (r - C[a]).pow(2).sum(1)
    cnt = torch.bincount(a, minlength=K).float()
    sse = torch.zeros(K, device=r.device).index_add_(0, a, se)
    return a, cnt, sse


def _perp(cnt):
    p = cnt / cnt.sum().clamp_min(1)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


@torch.no_grad()
def _principal_dir(r, mask, iters=8):
    """Top principal direction of the points in `mask` (power iteration)."""
    X = r[mask]
    if X.shape[0] < 2:
        return None, 0.0
    mu = X.mean(0)
    Xc = X - mu
    v = torch.randn(X.shape[1], device=r.device)
    v /= v.norm().clamp_min(1e-12)
    for _ in range(iters):
        v = Xc.t() @ (Xc @ v)
        n = v.norm()
        if n < 1e-12:
            return None, 0.0
        v = v / n
    lam = float((Xc @ v).pow(2).mean())
    return v, lam ** 0.5


@torch.no_grad()
def refit_level(r, C, dead_frac=0.02, recycle_frac=0.05, lloyd_iters=3,
                eps=0.5):
    """Refit one RVQ level on residual `r`; returns (new_C, stats).

    Victims are the underused (count below `dead_frac` of the mean),
    topped up to `recycle_frac` of K by the LOWEST distortion-contribution
    codes -- a code with a small count but a large error contribution is
    doing useful work and should not be recycled, whereas one carrying
    negligible error is a good victim regardless of its count.

    Donors are the highest distortion-contribution cells, and each donates
    once per phase. The pair is placed at centroid +/- eps*sigma along the
    donor's principal direction, so both members are viable; teleporting
    the revived code to the donor's worst-fitting point produces a
    codeword that serves an outlier and dies again.
    """
    K, N = C.shape[0], r.shape[0]
    C = C.clone()
    a, cnt, sse = _cell_stats(r, C)
    st = {"perp_before": _perp(cnt), "K": K,
          "resid_before": float((r - C[a]).pow(2).sum(1).mean()),
          "n_zero_before": int((cnt == 0).sum())}

    mean_cnt = N / K
    victim = cnt < dead_frac * mean_cnt
    n_min = int(recycle_frac * K)
    if int(victim.sum()) < n_min:                 # top up by lowest SSE
        order = torch.argsort(sse)                # ascending: least useful
        for idx in order.tolist():
            if int(victim.sum()) >= n_min:
                break
            victim[idx] = True
    vict = torch.nonzero(victim).flatten()
    n_rec = int(vict.numel())

    # donors: highest distortion contribution among the survivors
    score = sse.clone()
    score[victim] = -1.0
    donors = torch.argsort(score, descending=True)[:n_rec]
    n_split = 0
    for j in range(n_rec):
        d = int(donors[j])
        if score[d] <= 0:
            break
        v, sig = _principal_dir(r, a == d)
        if v is None or sig == 0.0:
            continue
        C[d] = C[d] + eps * sig * v               # classical LBG pair
        C[int(vict[j])] = C[d] - 2 * eps * sig * v
        n_split += 1

    for _ in range(lloyd_iters):
        a2 = _chunked_assign(r, C)
        sums = torch.zeros_like(C).index_add_(0, a2, r)
        c2 = torch.zeros(K, device=r.device).index_add_(
            0, a2, torch.ones(N, device=r.device))
        live = c2 > 0
        C[live] = sums[live] / c2[live, None]

    a3, cnt3, _ = _cell_stats(r, C)
    st.update({
        "n_recycled": float(n_rec), "frac_recycled": n_rec / K,
        "n_split": float(n_split),
        "perp_after": _perp(cnt3),
        "resid_after": float((r - C[a3]).pow(2).sum(1).mean()),
        "n_zero_after": int((cnt3 == 0).sum()),
    })
    return C, a3, st


@torch.no_grad()
def refit_phase(model, samples, cfg, lloyd_iters=3, dead_frac=0.02,
                recycle_frac=0.05):
    """Refit every tier, LEVEL BY LEVEL in cascade order.

    The whole point of the ordering: level l+1 is fitted on the residual
    that level l's UPDATED book produces, so a repair upstream is already
    accounted for downstream instead of invalidating it. A `before` chain
    is computed first with the original books so every level has an
    honest baseline including the cascade effect.
    """
    stats = {}
    for i, (rvq, sp, Z) in enumerate(zip(model.rvqs, cfg.model.tiers,
                                         samples)):
        # baseline chain: the whole cascade with the books as they stand
        r0 = Z.clone()
        base_in = []
        for lev in rvq.levels:
            base_in.append(float(r0.pow(2).sum(1).mean()))
            r0 = r0 - lev.codebook.weight.data[
                _chunked_assign(r0, lev.codebook.weight.data.float())]
        del r0

        r = Z.clone()
        for l, lev in enumerate(rvq.levels):
            C_old = lev.codebook.weight.data.float().clone()
            C_new, a, st = refit_level(r, C_old, dead_frac=dead_frac,
                                       recycle_frac=recycle_frac,
                                       lloyd_iters=lloyd_iters)
            st["centroid_shift"] = float((C_new - C_old).norm(dim=1).mean())
            st["input_energy_before"] = base_in[l]
            st["input_energy_after"] = float(r.pow(2).sum(1).mean())
            lev.codebook.weight.data.copy_(
                C_new.to(lev.codebook.weight.dtype))
            # keep the EMA buffers consistent so any downstream diagnostic
            # that reads them is not looking at stale state
            _, cnt, _ = _cell_stats(r, C_new)
            lev.ema_count.copy_((cnt / max(r.shape[0], 1) * 4096)
                                .clamp_min(2 * lev.dead_threshold))
            lev.ema_sum.copy_(C_new * lev.ema_count[:, None])
            lev.initialized.fill_(True)
            r = r - C_new[a]
            for k, v in st.items():
                stats[f"refit/pyr{i}_L{l+1}_{k}"] = v
        del r
    return stats
