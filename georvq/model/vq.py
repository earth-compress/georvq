"""Vector quantization: EMA/gradient codebooks, residual stacks, k-means init.

Codebook-health machinery (all deliberate; VQ state must stay fp32 even
when the surrounding network runs bf16):
- data-driven init on first training forward
- periodic dead-code reset from current encoder outputs
- Laplace-smoothed EMA updates (fp32 buffers even under bf16 autocast)
- optional bf16 assignment scores (`bf16_scores`): tensor-core distance
  matrix with boundary flips only; z_q/residuals/EMA stay fp32
- optional gradient codebooks (`grad_codebook`): VQ-VAE codebook loss
  instead of EMA, so a rate model's dR/dC gradient is live
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector quantization with EMA codebook updates and straight-through
    estimator. See module docstring for the health machinery."""

    def __init__(self, num_codes=512, dim=128, decay=0.99, eps=1e-5,
                 kmeans_init=True, reset_interval=2500, dead_threshold=0.01,
                 grad_codebook=False, bf16_scores=False,
                 reset_mode="random"):
        super().__init__()
        self.reset_mode = reset_mode
        self.grad_codebook = grad_codebook
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.kmeans_init = kmeans_init
        # period, in trainer steps, of this level's dead-code resets; 0
        # disables them. Read by the trainer's schedule (train.loop
        # mark_resets), not by forward()
        self.reset_interval = reset_interval
        self.dead_threshold = dead_threshold
        self.bf16_scores = bf16_scores

        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_sum", torch.zeros(num_codes, dim))
        self.register_buffer("initialized", torch.tensor(not kmeans_init, dtype=torch.bool))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        # Set by the trainer at each scheduled reset point; the level then
        # resets at its next training forward that sees data (with
        # skip_masked a level can sit out a batch) and clears it. The
        # schedule belongs to the trainer: `step` counts this level's
        # updates for the reset stats and decides nothing. Plain
        # attribute, no state-dict impact.
        self.reset_pending = False
        if reset_mode == "split":
            # EMA of each code's total squared error: the distortion
            # CONTRIBUTION (count x within-cell spread), which is what a
            # split should be targeted at. Non-persistent, so the
            # state-dict layout -- and therefore every existing
            # checkpoint's strict load -- is untouched; it re-accumulates
            # within a few hundred steps after a resume.
            self.register_buffer("ema_sse", torch.zeros(num_codes),
                                 persistent=False)

    @torch.no_grad()
    def _sample_from(self, z_flat, k):
        """Return k vectors sampled (with replacement if needed) from z_flat."""
        n = z_flat.shape[0]
        idx = torch.randint(0, n, (k,), device=z_flat.device) if n < k \
            else torch.randperm(n, device=z_flat.device)[:k]
        return z_flat[idx]

    @torch.no_grad()
    def _data_init(self, z_flat):
        """Initialize codebook from a sample of input vectors. Called once.

        ema_count starts at 0.1 (an order of magnitude above dead_threshold)
        so unused codes become eligible for reset quickly and ema_usage is an
        honest signal from the start."""
        sampled = self._sample_from(z_flat, self.num_codes)
        # Tiny noise so identical samples don't become permanently tied.
        sampled = sampled + 1e-4 * torch.randn_like(sampled)
        self.codebook.weight.data.copy_(sampled)
        self.ema_sum.copy_(sampled * 0.1)
        self.ema_count.fill_(0.1)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _reset_dead(self, z_flat, indices=None, z_q=None):
        """Revive codes whose EMA mass has fallen below threshold.

        reset_mode="random" (default, unchanged): reseed from random rows
        of z_flat. Cheap, but a uniformly drawn point lands in a DENSE
        region by construction -- usually one already well covered -- so
        the revived code steals a few points from a healthy neighbour,
        contributes little, and often dies again at the next reset.

        reset_mode="split": send each dead code into the live cell with
        the largest distortion CONTRIBUTION (EMA of summed squared
        error) and split that cell in two -- see `_split_targets` for
        where the pair is placed. This is LBG splitting: the new
        codeword lands where the error actually is. Point count alone
        would be the wrong target, since a large but tight cluster gains
        nothing from a second codeword; what matters is count x spread,
        which is what the SSE EMA measures.

        `replacements` is cast to the codebook dtype: z_flat may be bf16 under
        autocast while the codebook is fp32."""
        dead = self.ema_count < self.dead_threshold
        n_dead = int(dead.sum().item())
        # Before-state, measured on this batch with the pre-reset book.
        # Recorded even when NOTHING is dead: otherwise a level that never
        # loses a code emits no diagnostic at all and is indistinguishable
        # in the figure from one that is broken -- which is exactly
        # backwards, since it is the healthiest level in the model.
        stats = None
        if indices is not None and z_q is not None:
            p_ = torch.bincount(indices, minlength=self.num_codes).float()
            p_ = p_ / p_.sum().clamp_min(1)
            stats = {
                "n_reset": float(n_dead),
                "frac_reset": n_dead / self.num_codes,
                "perp_before": float(torch.exp(
                    -(p_ * p_.clamp_min(1e-12).log()).sum())),
                "resid_before": float((z_flat - z_q).pow(2).sum(1).mean()),
            }
        if n_dead == 0:
            if stats is not None:              # nothing changed: after == before
                stats["perp_after"] = stats["perp_before"]
                stats["resid_after"] = stats["resid_before"]
                stats["perp_gain"] = 0.0
                stats["resid_gain"] = 0.0
                stats["step"] = int(self.step.item())
                self.last_reset = stats
            return 0
        if self.reset_mode == "split" and indices is not None:
            reps = self._split_targets(z_flat, indices, z_q, dead, n_dead)
        else:
            reps = self._sample_from(z_flat, n_dead)
        replacements = reps.to(self.codebook.weight.data.dtype)
        self.codebook.weight.data[dead] = replacements
        self.ema_sum[dead] = replacements
        self.ema_count[dead] = 1.0
        if stats is not None:
            # after-state: re-assign the SAME batch to the revived book, so
            # before/after differ only by the reset. One chunked pass every
            # reset_interval steps -- negligible against 2500 training steps.
            idx2 = _chunked_assign(z_flat.float(),
                                   self.codebook.weight.data.float())
            q2 = self.codebook.weight.data[idx2]
            p2 = torch.bincount(idx2, minlength=self.num_codes).float()
            p2 = p2 / p2.sum().clamp_min(1)
            stats["perp_after"] = float(torch.exp(
                -(p2 * p2.clamp_min(1e-12).log()).sum()))
            stats["resid_after"] = float(
                (z_flat.float() - q2).pow(2).sum(1).mean())
            stats["perp_gain"] = stats["perp_after"] - stats["perp_before"]
            stats["resid_gain"] = stats["resid_before"] - stats["resid_after"]
            stats["step"] = int(self.step.item())
            self.last_reset = stats           # plain attr: no state-dict impact
        return n_dead

    @torch.no_grad()
    def _split_targets(self, z_flat, indices, z_q, dead, n_dead):
        """LBG split of the top-SSE live cells: the pair straddles the
        donor centroid, offset along a data-driven direction.

        The previous version put the revived code on the donor's
        WORST-FITTING point. That is an outlier by construction, so the
        new codeword served almost nobody and died again -- measured on
        this model, the instantaneous residual drop was 91% at tier1 L8
        and ~100% of it evaporated before the next reset, every reset, for
        23000 steps (resid_before 86.1 -> 85.1). Classical LBG splits the
        CENTROID into c +/- eps*delta so both members are viable.

        `delta` is (random member - centroid): a TYPICAL offset for that
        cell, which gives both a direction and a scale for free and stays
        fully vectorized. A per-cell principal direction would be better
        in principle, but at K=16384 a batch gives a donor ~16 points in
        256 dims, so that estimate is mostly noise.

        Any dead code that cannot be paired with a live cell holding a
        point in this batch falls back to a random row."""
        eps = 0.5
        se = (z_flat - z_q).pow(2).sum(1).float()
        # one RANDOM member per code: shuffle, then scatter row ids so the
        # last write per code is an arbitrary member (not the worst one)
        order = torch.randperm(z_flat.shape[0], device=z_flat.device)
        worst = torch.full((self.num_codes,), -1, dtype=torch.long,
                           device=z_flat.device)
        worst[indices[order]] = order
        score = self.ema_sse.clone()
        score[dead] = -1.0                    # never split a dead cell
        score[worst < 0] = -1.0               # nothing in this batch to use
        srcs = torch.argsort(score, descending=True)[:n_dead]
        ok = score[srcs] > 0
        reps = torch.empty(n_dead, self.dim, device=z_flat.device,
                           dtype=z_flat.dtype)
        if bool(ok.any()):
            src_ok = srcs[ok]
            C = self.codebook.weight.data
            delta = z_flat[worst[src_ok]].to(C.dtype) - C[src_ok]
            # pair straddles the old centroid: donor moves +eps*delta, the
            # revived code takes -eps*delta
            reps[ok] = (C[src_ok] - eps * delta).to(reps.dtype)
            C[src_ok] = C[src_ok] + eps * delta
            # halve the source's mass and error: the pair now shares the
            # cell, and one hot cell must not win every slot in a round
            self.ema_count[src_ok] *= 0.5
            self.ema_sum[src_ok] = C[src_ok] * self.ema_count[src_ok, None]
            self.ema_sse[src_ok] *= 0.5
        if bool((~ok).any()):
            reps[~ok] = self._sample_from(z_flat, int((~ok).sum()))
        return reps

    def forward(self, z):
        """
        Args:
            z: (..., D) continuous vectors
        Returns:
            z_q: (..., D) quantized vectors (straight-through)
            indices: (...,) codebook indices
            commit_loss: scalar commitment loss
        """
        flat_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.dim)  # (N, D)

        # Lazy data-driven init
        if self.training and not self.initialized.item():
            self._data_init(z_flat.detach())

        if self.bf16_scores:
            # argmin ||z-e||^2 == argmax (z.e - ||e||^2/2); scores in bf16
            # (tensor cores + half the matrix traffic), boundary flips only.
            # Everything downstream (z_q, residuals, EMA) stays fp32.
            E = self.codebook.weight
            s = z_flat.bfloat16() @ E.t().bfloat16()
            indices = (s - (E.pow(2).sum(1) * 0.5).bfloat16()[None]) \
                .argmax(dim=-1)
        else:
            # Distances: ||z - e||^2 = ||z||^2 - 2*z@e^T + ||e||^2
            d = (z_flat.pow(2).sum(1, keepdim=True)
                 - 2 * z_flat @ self.codebook.weight.t()
                 + self.codebook.weight.pow(2).sum(1, keepdim=True).t())
            indices = d.argmin(dim=-1)  # (N,)
        z_q = self.codebook(indices)  # (N, D)

        # EMA codebook update (training only; skipped under grad_codebook)
        if self.training and not self.grad_codebook:
            with torch.no_grad():
                # Match z_flat's dtype (bf16 under autocast, fp32 otherwise) so
                # the dense one_hot @ z_flat matmul runs at the same precision
                # as the encoder. EMA buffers themselves stay fp32 — values
                # get upcast on add_ — so the running state is preserved.
                one_hot = F.one_hot(indices, self.num_codes).to(z_flat.dtype)  # (N, K)
                count = one_hot.sum(0)  # (K,)
                vec_sum = one_hot.t() @ z_flat  # (K, D)

                self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(vec_sum, alpha=1 - self.decay)

                # Laplace smoothing
                n = self.ema_count.sum()
                count_stable = (self.ema_count + self.eps) / (n + self.num_codes * self.eps) * n
                self.codebook.weight.data.copy_(self.ema_sum / count_stable.unsqueeze(1))

                if self.reset_mode == "split":
                    # per-code EMA of TOTAL squared error = the distortion
                    # contribution a split would recover
                    se = (z_flat - z_q).pow(2).sum(1).float()
                    self.ema_sse.mul_(self.decay).add_(
                        torch.zeros_like(self.ema_sse).index_add_(
                            0, indices, se), alpha=1 - self.decay)

                self.step += 1
                if self.reset_pending:
                    self.reset_pending = False
                    self._reset_dead(z_flat.detach(), indices,
                                     z_q.detach())

        # Commitment loss: encoder outputs should be close to codebook entries
        commit_loss = F.mse_loss(z_flat, z_q.detach())
        if self.grad_codebook:
            # codebook loss pulls codes toward encoder outputs (replaces EMA)
            commit_loss = commit_loss + F.mse_loss(z_q, z_flat.detach())

        # Straight-through estimator
        z_q_st = z_flat + (z_q - z_flat).detach()

        return z_q_st.reshape(*flat_shape, self.dim), indices.reshape(flat_shape), commit_loss

    def forward_gated(self, z, scores, subset_k, sel_n=4096, temp=0.5, tau=0.4):
        """Location-gated VQ (codebook subsetting). Encode using ONLY the
        top-`subset_k` codes that `scores` (per-sample, over the codebook)
        selects — a hard subset for the actual code/EMA/index, plus a soft
        top-k mask that gives the selector a reconstruction gradient.
        Returns (z_q_st, indices, commit, select_loss).

        Rate is unchanged vs the original codebook: 4x codes (+2 bits) minus
        the 1/4 subset (-2 bits). The selector is trained so the *set* of
        allowed codes covers the patch distribution at that location."""
        flat = z.shape[:-1]                                          # (B, cells)
        z_flat = z.reshape(-1, self.dim)
        B = flat[0]
        if self.training and not self.initialized.item():
            self._data_init(z_flat.detach())
        E = self.codebook.weight
        # hard assignment via the PER-WINDOW subset: take each window's
        # top-`subset_k` codes once, gather just those code vectors, and bmm
        # the window's cells against them. Cost == the plain path (subset_k
        # codes, not 4*K). Non-differentiable (the selector learns from the
        # soft path).
        with torch.no_grad():
            allowed = scores.topk(subset_k, dim=-1).indices         # (B, subk) per window
        E_sub = E[allowed]                                          # (B, subk, D)
        zc = z if z.dim() == 3 else z_flat.unsqueeze(1)             # (B, cells, D)
        sco = torch.bmm(zc.bfloat16(), E_sub.transpose(1, 2).bfloat16()) \
            - (E_sub.pow(2).sum(-1) * 0.5).bfloat16()[:, None, :]   # (B, cells, subk) ~ -||z-e||^2
        local = sco.argmax(dim=-1)                                  # (B, cells) index into subset
        indices = allowed.gather(1, local).reshape(-1)              # (N,) global code index
        z_q = self.codebook(indices)

        if self.training and not self.grad_codebook:
            with torch.no_grad():
                # index_add avoids the (N,K) one-hot (4+ GiB at K=65536)
                count = torch.bincount(indices, minlength=self.num_codes).to(z_flat.dtype)
                vec_sum = torch.zeros_like(self.ema_sum, dtype=z_flat.dtype) \
                    .index_add_(0, indices, z_flat)
                self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(vec_sum, alpha=1 - self.decay)
                n = self.ema_count.sum()
                cs = (self.ema_count + self.eps) / (n + self.num_codes * self.eps) * n
                self.codebook.weight.data.copy_(self.ema_sum / cs.unsqueeze(1))
                if self.reset_mode == "split":
                    # per-code EMA of TOTAL squared error = the distortion
                    # contribution a split would recover
                    se = (z_flat - z_q).pow(2).sum(1).float()
                    self.ema_sse.mul_(self.decay).add_(
                        torch.zeros_like(self.ema_sse).index_add_(
                            0, indices, se), alpha=1 - self.decay)

                self.step += 1
                if self.reset_pending:
                    self.reset_pending = False
                    self._reset_dead(z_flat.detach(), indices,
                                     z_q.detach())

        # selector gradient: one cell per window (B rows, all locations). The
        # soft subset mask is folded into the softmax LOGITS (log space) so the
        # assignment is always normalized.
        zf = z[:, 0, :] if z.dim() == 3 else z_flat
        sf = scores                                                 # (B,K) per-window, carries grad
        d = (zf.pow(2).sum(1, keepdim=True) - 2 * zf @ E.t()
             + E.pow(2).sum(1).unsqueeze(0))                        # (B,K) fp32
        thr2 = torch.kthvalue(sf.float(), self.num_codes - subset_k + 1,
                              dim=-1).values.unsqueeze(-1)
        logm = torch.log(torch.sigmoid((sf - thr2) / temp) + 1e-6)   # soft subset, log space
        # Temperature = the LOCAL (nearest-neighbourhood) distance scale, not
        # the global mean over all K codes — sharpens soft_q toward the nearest
        # ALLOWED code, aligning the selector's gradient with the hard
        # assignment (global scale trains the selector to bracket z instead).
        loc = d.detach().topk(min(32, d.shape[-1]), dim=-1, largest=False).values \
            .mean(-1, keepdim=True)
        T = (loc * tau).clamp_min(1e-6)
        p = torch.softmax(-d / T + logm, dim=-1)                     # always sums to 1
        soft_q = p @ E
        select_loss = F.mse_loss(soft_q, zf.detach())

        commit_loss = F.mse_loss(z_flat, z_q.detach())
        if self.grad_codebook:
            commit_loss = commit_loss + F.mse_loss(z_q, z_flat.detach())
        z_q_st = z_flat + (z_q - z_flat).detach()
        return (z_q_st.reshape(*flat, self.dim), indices.reshape(flat),
                commit_loss, select_loss)

    def distances(self, z):
        """(N, K) squared distances z<->codebook (same d as the argmin).
        Carries a live gradient to the codebook under grad_codebook — the
        RDVQ rate branch consumes this as softmax(-d/tau)."""
        zf = z.reshape(-1, self.dim)
        return (zf.pow(2).sum(1, keepdim=True)
                - 2 * zf @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1, keepdim=True).t())

    @torch.no_grad()
    def usage(self, indices):
        """Fraction of codebook entries used in this batch."""
        return indices.unique().numel() / self.num_codes

    @torch.no_grad()
    def ema_usage(self):
        """Fraction of codebook entries above the dead-code threshold."""
        return (self.ema_count > self.dead_threshold).float().mean().item()

    @torch.no_grad()
    def perplexity(self, indices):
        """Effective codebook size — exp(entropy of code distribution).

        Healthy quantization has perplexity climbing toward a meaningful
        fraction of K. Collapse shows up as perplexity << K."""
        flat = indices.flatten()
        counts = torch.bincount(flat, minlength=self.num_codes).float()
        probs = counts / counts.sum().clamp_min(1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum()
        return torch.exp(ent).item()


class ResidualVQ(nn.Module):
    """Residual Vector Quantization: stack of VQ levels.
    Level 1 quantizes z, level 2 quantizes the residual, etc."""

    def __init__(self, num_levels=8, num_codes=256, dim=128, decay=0.99,
                 kmeans_init=True, reset_interval=2500, dead_threshold=0.01,
                 grad_codebook=False, bf16_scores=False,
                 reset_mode="random"):
        super().__init__()
        self.num_levels = num_levels
        self.levels = nn.ModuleList([
            VectorQuantizer(num_codes, dim, decay,
                            kmeans_init=kmeans_init,
                            reset_interval=reset_interval,
                            dead_threshold=dead_threshold,
                            grad_codebook=grad_codebook,
                            bf16_scores=bf16_scores,
                            reset_mode=reset_mode)
            for _ in range(num_levels)
        ])

    def forward(self, z, n_levels=None):
        """
        Args:
            z: (..., D) continuous vectors
            n_levels: how many levels to use (None = all).
        Returns:
            z_q: (..., D) quantized (sum of all levels)
            all_indices: list of (...,) indices per level
            commit_loss: scalar total commitment loss
        """
        if n_levels is None:
            n_levels = self.num_levels

        residual = z
        z_q = torch.zeros_like(z)
        all_indices = []
        commit_loss = torch.tensor(0.0, device=z.device)

        for i in range(n_levels):
            level_q, level_idx, level_commit = self.levels[i](residual)
            residual = residual - level_q.detach()  # detach so each level sees clean residual
            z_q = z_q + level_q
            all_indices.append(level_idx)
            commit_loss = commit_loss + level_commit

        return z_q, all_indices, commit_loss / n_levels

    def all_levels(self, z):
        """Run all L levels and return per-level contributions (unsummed).

        Caller can apply a per-sample mask to the level dimension to realize
        any truncation without re-running quantization. Costs the same as
        forward() — we run every level anyway.

        Returns:
            per_level_q: (..., L, D) — each level's quantized contribution
            all_indices: list of L (...,) tensors
            commit_loss: scalar (mean over levels)
        """
        residual = z
        per_level = []
        all_indices = []
        commit_loss = torch.tensor(0.0, device=z.device)
        for lvl in self.levels:
            q, idx, c = lvl(residual)
            residual = residual - q.detach()
            per_level.append(q)
            all_indices.append(idx)
            commit_loss = commit_loss + c
        per_level_q = torch.stack(per_level, dim=-2)  # (..., L, D)
        return per_level_q, all_indices, commit_loss / self.num_levels

    def forward_gated(self, z, scores, subset_k, n_levels=None, sel_n=4096,
                      temp=0.5, tau=0.4):
        """Location-gated residual VQ. `scores`: list of L tensors (..., K) —
        the per-depth selector scores for this pyramid tier. `subset_k`: codes
        kept per book (= original codebook size).
        Returns (per_level_q, indices, commit, select)."""
        n = n_levels or self.num_levels
        residual = z
        per = []
        idxs = []
        commit = z.new_zeros(())
        select = z.new_zeros(())
        for i in range(n):
            q, idx, c, s = self.levels[i].forward_gated(
                residual, scores[i], subset_k, sel_n=sel_n, temp=temp, tau=tau)
            residual = residual - q.detach()
            per.append(q)
            idxs.append(idx)
            commit = commit + c
            select = select + s
        # per-level contributions (..., L, D) so the thermometer matryoshka
        # mask applies exactly as in all_levels()
        return torch.stack(per, dim=-2), idxs, commit / n, select / n

    @torch.no_grad()
    def all_levels_gated(self, z, scores, subset_k):
        """Eval: per-level gated quantization (hard subset), unsummed. Mirrors
        all_levels so the matryoshka truncation mask works unchanged."""
        residual = z
        per = []
        idxs = []
        for i, lvl in enumerate(self.levels):
            E = lvl.codebook.weight
            allowed = scores[i].topk(subset_k, dim=-1).indices          # (B, subk)
            E_sub = E[allowed]                                          # (B, subk, D)
            zc = residual if residual.dim() == 3 else residual.unsqueeze(1)
            sc = torch.bmm(zc.bfloat16(), E_sub.transpose(1, 2).bfloat16()) \
                - (E_sub.pow(2).sum(-1) * 0.5).bfloat16()[:, None, :]   # (B, cells, subk)
            gidx = allowed.gather(1, sc.argmax(dim=-1))                 # (B, cells) global
            q = lvl.codebook(gidx.reshape(-1)).reshape_as(residual)
            residual = residual - q
            per.append(q)
            idxs.append(gidx)
        return torch.stack(per, dim=-2), idxs

    def levels_with_rate(self, z, n_levels=None):
        """Rate-aware pass. Like all_levels, but per level also yields the
        distance matrix `d` (for the soft-rate softmax(-d/tau); live grad to
        encoder via the residual and to the codebook under grad_codebook) and
        `ctx` = the cumulative reconstruction z_q^{(<l)} BEFORE this level
        (the causal context the entropy model conditions on). Returns a list
        of dicts [{d, q, idx, commit, ctx}, ...]."""
        n_levels = n_levels or self.num_levels
        residual = z
        cum = torch.zeros_like(z)                       # z_q^{(<l)}
        out = []
        for i in range(n_levels):
            d = self.levels[i].distances(residual)      # (N, K)
            q, idx, c = self.levels[i](residual)
            out.append({"d": d, "q": q, "idx": idx, "commit": c, "ctx": cum})
            residual = residual - q.detach()
            cum = cum + q.detach()
        return out

    def usage(self, all_indices):
        """Average codebook usage across levels."""
        return sum(lev.usage(idx) for lev, idx in zip(self.levels, all_indices)) / len(all_indices)

    def ema_usage(self):
        """Average global codebook usage (EMA-based) across levels."""
        return sum(lev.ema_usage() for lev in self.levels) / len(self.levels)


def _chunked_assign(x, C, chunk=32768):
    outs, cn = [], C.pow(2).sum(1)
    for xc in x.split(chunk):
        d = xc.pow(2).sum(1, keepdim=True) - 2 * xc @ C.t() + cn[None]
        outs.append(d.argmin(1))
    return torch.cat(outs)


@torch.no_grad()
def kmeans_init_rvq(rvq, z, iters=6, expected_batch=4096):
    """Residual k-means init after a VQ-free warmup: fit level 1 on latents,
    level 2 on its residuals, etc. Writes codebooks and EMA buffers (counts
    at steady-state assignment rates so nothing is instantly 'dead') and
    marks levels initialized so the lazy data-init never fires."""
    residual = z.reshape(-1, z.shape[-1]).float()
    if residual.shape[0] > 1_000_000:
        residual = residual[torch.randperm(residual.shape[0],
                                           device=z.device)[:1_000_000]]
    N = residual.shape[0]
    dev = residual.device
    for lev in rvq.levels:
        K = lev.num_codes
        C = residual[torch.randperm(N, device=dev)[:K]].clone()
        ones = torch.ones(N, device=dev)
        for _ in range(iters):
            assign = _chunked_assign(residual, C)
            sums = torch.zeros_like(C)
            cnt = torch.zeros(K, device=dev)
            sums.index_add_(0, assign, residual)
            cnt.index_add_(0, assign, ones)
            live = cnt > 0
            C[live] = sums[live] / cnt[live, None]
            n_dead = int((~live).sum())
            if n_dead:
                C[~live] = residual[torch.randint(0, N, (n_dead,), device=dev)] \
                    + 1e-4 * torch.randn(n_dead, C.shape[1], device=dev)
        assign = _chunked_assign(residual, C)
        cnt = torch.zeros(K, device=dev)
        cnt.index_add_(0, assign, ones)
        lev.codebook.weight.data.copy_(C)
        lev.ema_count.copy_((cnt / N * expected_batch)
                            .clamp_min(2 * lev.dead_threshold))
        lev.ema_sum.copy_(C * lev.ema_count[:, None])
        lev.initialized.fill_(True)
        residual = residual - C[assign]
