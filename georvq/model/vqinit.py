"""Codebook initialisation: balanced PCA-tree seeding, and the residual
cascade that applies it level by level.

Kept out of `vq.py` on purpose -- nothing here touches a decode path, so
the golden-checkpoint gate is unaffected by anything in this file.

Why a tree rather than random data points: the draws are i.i.d. from the
data, so dense regions get many near-duplicate seeds that immediately
compete and die, while sparse-but-high-error regions get none. A balanced
median tree spends log2(K) passes instead, each estimating one direction
and one median from an enormous number of points. Heavily duplicated
rows are the exception: a node holding only copies of one row splits
every copy to the same side, so its other child is empty and comes back
as a zero vector (and for K not a power of two the padding draws random
rows). The Lloyd iterations in `residual_init_rvq` reseed those cells.

NOT because the fit is sample-starved: fitting from the full latent store
gives 4965 / 1953 / 977 vectors per codeword across the three tiers, all
comfortably above the ~D=384 needed to determine a cell mean. Tier 2's
perplexity falloff at deep RVQ levels (95% of K at level 2 down to 20% at
level 7) is therefore most likely real -- deep residuals carry less
structure, so fewer distinct directions earn a codeword -- and not a
seeding artifact. Starvation does apply to the in-training init, where
coupling through batch count leaves the coarse tier at ~16 vectors per
codeword; that is the case the offline store replaces.

This is an INIT criterion, not a distortion criterion: every node splits,
every round, at its own median. Which node to split is never a choice.
The distortion-ranked splitting (n_j * lambda_1) lives in
VectorQuantizer._split_targets and serves the opposite purpose -- placing
a code that has already died where the error actually is.

It is also near-optimal at this dimensionality for a specific reason:
Zador's optimal fixed-rate codeword density goes as p(x)^(d/(d+2)), whose
exponent at d=384 is 0.995 -- essentially p(x) itself, i.e. equal
probability mass per cell. Balanced cells are close to what the optimal
quantizer looks like here. At d=3 they would not be.

The tree's cells are intersections of half-spaces, not Voronoi cells, so
this is an INITIALISER: Lloyd iterations follow and are free to unbalance
it wherever the data says they should.
"""
from __future__ import annotations

import math

import torch

from georvq.model.vq import _chunked_assign


@torch.no_grad()
def tree_init(X, K, iters=8, chunk=1 << 20):
    """Balanced PCA tree -> (K, D) leaf means.

    A node with no rows yields an all-zero leaf; see the module docstring.
    Never gathers a node's rows: one gather at the root would copy the
    whole fit set. All per-node statistics come from chunked
    scatter-reduction, and every node advances its power iteration in the
    same pass, so cost is O(log2(K)) sweeps regardless of K.
    """
    N, D = X.shape
    dev = X.device
    depth = int(math.ceil(math.log2(K)))
    node = torch.zeros(N, dtype=torch.long, device=dev)
    for _ in range(depth):
        n_nodes = int(node.max().item()) + 1
        cnt = torch.bincount(node, minlength=n_nodes).clamp_min(1).float()
        MU = torch.zeros(n_nodes, D, device=dev)
        for i in range(0, N, chunk):
            MU.index_add_(0, node[i:i + chunk], X[i:i + chunk])
        MU /= cnt[:, None]
        V = torch.randn(n_nodes, D, device=dev)
        V /= V.norm(dim=1, keepdim=True)
        for _ in range(iters):                    # power iteration, all nodes
            A = torch.zeros(n_nodes, D, device=dev)
            for i in range(0, N, chunk):
                nb = node[i:i + chunk]
                xc = X[i:i + chunk] - MU[nb]
                A.index_add_(0, nb, xc * (xc * V[nb]).sum(1, keepdim=True))
            V = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
        p = torch.empty(N, device=dev)
        for i in range(0, N, chunk):
            nb = node[i:i + chunk]
            p[i:i + chunk] = ((X[i:i + chunk] - MU[nb]) * V[nb]).sum(1)
        # each node splits at ITS OWN median, so the halves stay balanced
        order = torch.argsort(node)
        ps = p[order]
        offs = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                          torch.bincount(node, minlength=n_nodes).cumsum(0)])
        thr = torch.zeros(n_nodes, device=dev)
        for nid in range(n_nodes):
            seg = ps[offs[nid]:offs[nid + 1]]
            if seg.numel():
                thr[nid] = seg.median()
        node = node + (p > thr[node]).long() * n_nodes
        del p, order, ps
    n_nodes = int(node.max().item()) + 1
    cnt = torch.bincount(node, minlength=n_nodes).clamp_min(1).float()
    C = torch.zeros(n_nodes, D, device=dev)
    for i in range(0, N, chunk):
        C.index_add_(0, node[i:i + chunk], X[i:i + chunk])
    C /= cnt[:, None]
    if C.shape[0] < K:                            # K not a power of two
        pad = X[torch.randint(0, N, (K - C.shape[0],), device=dev)]
        C = torch.cat([C, pad])
    return C[:K].contiguous()


@torch.no_grad()
def residual_init_rvq(rvq, z, mode="kmeans", iters=6, expected_batch=4096,
                      max_fit=0):
    """Residual init of a whole RVQ stack: fit level 1 on the latents,
    level 2 on its residuals, and so on.

    Writes codebooks and EMA buffers (counts at steady-state assignment
    rates, so nothing is instantly dead) and marks levels initialized so
    the lazy data-init never fires. The fit set is not capped unless
    `max_fit` is given: a silent 1e6 cap would leave 61 vectors per
    centroid at K=16384.
    """
    residual = z.reshape(-1, z.shape[-1]).float()
    if max_fit and residual.shape[0] > max_fit:
        sel = torch.randperm(residual.shape[0], device=residual.device)[:max_fit]
        residual = residual[sel]
    N = residual.shape[0]
    dev = residual.device
    ones = torch.ones(N, device=dev)
    for lev in rvq.levels:
        K = lev.num_codes
        if mode == "tree":
            C = tree_init(residual, K)
        else:
            C = residual[torch.randperm(N, device=dev)[:K]].clone()
        for _ in range(iters):
            assign = _chunked_assign(residual, C)
            sums = torch.zeros_like(C).index_add_(0, assign, residual)
            cnt = torch.zeros(K, device=dev).index_add_(0, assign, ones)
            live = cnt > 0
            C[live] = sums[live] / cnt[live, None]
            n_dead = int((~live).sum())
            if n_dead:
                C[~live] = residual[torch.randint(0, N, (n_dead,), device=dev)] \
                    + 1e-4 * torch.randn(n_dead, C.shape[1], device=dev)
        assign = _chunked_assign(residual, C)
        cnt = torch.zeros(K, device=dev).index_add_(0, assign, ones)
        lev.codebook.weight.data.copy_(C)
        lev.ema_count.copy_((cnt / N * expected_batch)
                            .clamp_min(2 * lev.dead_threshold))
        lev.ema_sum.copy_(C * lev.ema_count[:, None])
        lev.initialized.fill_(True)
        if getattr(lev, "reset_mode", "random") == "split":
            # seed the distortion tracker so the first reset round has a
            # real ranking instead of an all-zero one
            se = (residual - C[assign]).pow(2).sum(1).float()
            lev.ema_sse.copy_(torch.zeros_like(lev.ema_sse)
                              .index_add_(0, assign, se))
        residual = residual - C[assign]
