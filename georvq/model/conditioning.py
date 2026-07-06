"""Conditioning: use side information to make codes cheaper or better.

Two mechanisms, one theme (subtract what is predictable, code the rest):

- `TierPredictor` (predict-and-subtract cascade): each tier's latents are
  predicted from the already-decoded coarser tiers (gathered at ancestor
  cells) plus an optional location embedding; the RVQ codes the residual.
  Tier 0's prediction comes from a frozen pretrained location->latent prior
  (or zero without one), so the same component covers pure cross-tier and
  location-conditioned cascades.

- `CodebookSelector` (codebook subsetting): an oversized codebook of which
  only a location-selected top fraction is usable per window. Rate-neutral
  (4x codes = +2 bits, 1/4 subset = -2 bits) but every usable code is
  location-appropriate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from georvq.model.layers import thermometer


def ancestor_map(g_child: int, g_parent: int) -> torch.Tensor:
    """Child-cell index -> its ancestor's cell index on a coarser grid."""
    f = g_child // g_parent
    idx = [(r // f) * g_parent + (c // f)
           for r in range(g_child) for c in range(g_child)]
    return torch.tensor(idx, dtype=torch.long)


def _mlp(i, o, h=512, layers=3):
    seq = [nn.Linear(i, h), nn.GELU()]
    for _ in range(layers - 1):
        seq += [nn.Linear(h, h), nn.GELU()]
    seq += [nn.Linear(h, o)]
    m = nn.Sequential(*seq)
    nn.init.zeros_(m[-1].weight)
    nn.init.zeros_(m[-1].bias)   # zero-init -> predict 0 at start
    return m


class TierPredictor(nn.Module):
    """Per-tier predict-and-subtract. Predictor i>0 sees every coarser
    tier's reconstruction (at its own ancestor cell), the location
    embedding, and the coarser tiers' depth thermometers."""

    def __init__(self, specs, loc_dim=0, coarse_prior=None, hidden=512,
                 layers=3):
        super().__init__()
        self.specs = specs
        self.loc_dim = loc_dim
        self.coarse_prior = coarse_prior          # frozen loc -> z0, or None
        self.predictors = nn.ModuleList()
        for i, sp in enumerate(specs[1:], start=1):
            d_in = (sum(specs[j].dim for j in range(i)) + loc_dim
                    + sum(specs[j].depth for j in range(i)))
            self.predictors.append(_mlp(d_in, sp.dim, h=hidden, layers=layers))
        for i, sp in enumerate(specs[1:], start=1):
            for j in range(i):
                self.register_buffer(
                    f"anc_{i}_{j}", ancestor_map(sp.grid, specs[j].grid),
                    persistent=False)

    def _predict(self, i, recons, depths, loc, B):
        """Prediction for tier i's latents (B, cells_i, dim_i)."""
        sp = self.specs[i]
        cells = sp.cells
        parts = []
        for j in range(i):
            r = recons[j]                                    # (B, cells_j, D_j)
            amap = getattr(self, f"anc_{i}_{j}")
            parts.append(r[:, amap] if r.shape[1] > 1
                         else r.expand(B, cells, r.shape[-1]))
        if self.loc_dim:
            parts.append(loc[:, None].expand(B, cells, self.loc_dim))
        for j in range(i):
            th = thermometer(depths[j], self.specs[j].depth, parts[0].device)
            parts.append(th[:, None].expand(B, cells, self.specs[j].depth))
        return self.predictors[i - 1](torch.cat(parts, -1))

    def quantize(self, rvqs, specs, zs, loc, Ls, cumulative=False,
                 return_indices=False):
        """Cascade quantization.
        cumulative=False: (recons masked to Ls, commit[, idxs]) — training.
        cumulative=True: per-tier (B, cells, depth+1, dim) cumulative recons
        including the prior at index 0 — eval (any tuple = a gather)."""
        B, dev = zs[0].shape[0], zs[0].device
        commit = zs[0].new_zeros(())
        outs, idxs, recons, depths = [], [], [], []
        for i, (rvq, sp, z) in enumerate(zip(rvqs, specs, zs)):
            if i == 0:
                g = (self.coarse_prior(loc).unsqueeze(1)
                     if self.coarse_prior is not None and loc is not None
                     else torch.zeros_like(z))
            else:
                g = self._predict(i, recons, depths, loc, B)
            per, idx, c = rvq.all_levels(z - g)
            commit = commit + c
            idxs.append(idx)
            if cumulative:
                cum = torch.cat([torch.zeros_like(per[:, :, :1]),
                                 per.cumsum(-2)], -2) + g.unsqueeze(2)
                outs.append(cum)
                recons.append(cum[:, :, sp.depth])         # full-depth context
                depths.append(torch.full((B,), sp.depth, device=dev))
            else:
                L = Ls[:, i]
                m = thermometer(L, sp.depth, dev)[:, None, :, None]
                q = g + (per * m).sum(-2)
                outs.append(q)
                recons.append(q)
                depths.append(L)
        commit = commit / len(specs)
        if return_indices:
            return outs, commit, idxs
        return outs, commit

    def cumsums(self, rvqs, specs, zs, loc):
        return self.quantize(rvqs, specs, zs, loc, None, cumulative=True)[0]

    @torch.no_grad()
    def full_depth_indices(self, rvqs, specs, zs, loc):
        """Per-tier per-level indices at full depth (codebook health)."""
        B, dev = zs[0].shape[0], zs[0].device
        full = torch.stack([torch.full((B,), sp.depth, device=dev,
                                       dtype=torch.long) for sp in specs], 1)
        return self.quantize(rvqs, specs, zs, loc, full,
                             return_indices=True)[2]


class CodebookSelector(nn.Module):
    """Location -> per-(tier, depth) code scores. A shared MLP trunk encodes
    the location to a bottleneck; per-table heads map it to a query that is
    dot-scored against learned per-code embeddings (separate from the
    codebook, so near-duplicate codes can be scored differently)."""

    def __init__(self, specs, loc_dim=256, proj_dim=64, hidden=256,
                 bottleneck=128, subset_frac=0.25, sel_n=4096, temp=0.5,
                 tau=0.4):
        super().__init__()
        self.subset_frac = subset_frac
        self.sel_n = sel_n
        self.temp = temp
        self.tau = tau
        self.trunk = nn.Sequential(
            nn.Linear(loc_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck))
        self.head = nn.ModuleList()
        self.emb = nn.ParameterList()
        self.shapes = []
        for sp in specs:
            for _ in range(sp.depth):
                self.head.append(nn.Sequential(
                    nn.Linear(bottleneck, proj_dim), nn.GELU(),
                    nn.Linear(proj_dim, proj_dim)))
                self.emb.append(nn.Parameter(
                    torch.randn(sp.codes, proj_dim) * (proj_dim ** -0.5)))
            self.shapes.append((sp.codes, sp.depth))

    def subset_k(self, sp) -> int:
        return max(1, int(sp.codes * self.subset_frac))

    def forward(self, loc):
        b = self.trunk(loc)                               # (B, bottleneck)
        out, ei = [], 0
        for (codes, depth) in self.shapes:
            lvl = []
            for _ in range(depth):
                lvl.append(self.head[ei](b) @ self.emb[ei].t())   # (B, codes)
                ei += 1
            out.append(lvl)
        return out

    @torch.no_grad()
    def diagnostics(self, model, x, loc, valid, diag_windows=48):
        """Selector diagnostics on a small sub-batch:
        sel_recall_p{i}_L{l} — P(the full codebook's best code is INSIDE the
        selected subset); 1.0 = subsetting never excludes the code the
        encoder wanted. sel_div_p{i} — |union of per-window subsets| / K;
        subset_frac = same subset everywhere (degenerate), ->1 = strongly
        location-varying. sel_gated_cos / sel_ungated_cos — full-depth recon
        with the subset vs the full oversized codebook (the quality the
        subset costs vs the upper bound the selector chases)."""
        specs = model.specs
        n = min(diag_windows, x.shape[0])
        x = x[:n]; loc = loc[:n]; valid = valid[:n]
        zs = model.encode_levels(x)
        B = x.shape[0]
        scores = self(loc)
        out = {}
        zq_g, zq_u = [], []
        for i, (z, rvq, sp) in enumerate(zip(zs, model.rvqs, specs)):
            cells = z.shape[1]; K = sp.codes; subk = self.subset_k(sp)
            rg = z.float(); ru = z.float(); pg, pu = [], []
            # subset diversity from the per-window depth-0 scores
            s0 = scores[i][0]                                # (B, K)
            thr0 = torch.kthvalue(s0.float(), K - subk + 1,
                                  -1).values.unsqueeze(-1)
            out[f"sel_div_p{i}"] = (s0 >= thr0).any(0).float().mean().item()
            zN = z.reshape(-1, z.shape[-1]).shape[0]
            win = torch.arange(zN, device=z.device) // cells   # row -> window
            recalls = []
            for j, lvl in enumerate(rvq.levels):
                E = lvl.codebook.weight
                Eh = E.t().bfloat16()
                En = (E.pow(2).sum(1) * 0.5).bfloat16()[None]
                sw = scores[i][j]                            # (B, K)
                zg = rg.reshape(-1, lvl.dim); zu = ru.reshape(-1, lvl.dim)
                N = zg.shape[0]; rc = max(1, 67_108_864 // K)
                gi = torch.empty(N, dtype=torch.long, device=z.device)
                ui = torch.empty(N, dtype=torch.long, device=z.device)
                hit = 0
                for a in range(0, N, rc):        # row-chunked (K up to 65536)
                    s_ = sw[win[a:a + rc]]
                    thr = torch.kthvalue(s_.float(), K - subk + 1,
                                         -1).values.unsqueeze(-1)
                    dg = zg[a:a + rc].bfloat16() @ Eh - En
                    gb = dg.argmax(-1)           # full-codebook best (unmasked)
                    hit += (s_.gather(1, gb[:, None]) >= thr).sum().item()
                    dg.masked_fill_(s_ < thr, float("-inf"))
                    gi[a:a + rc] = dg.argmax(-1)             # best ALLOWED
                    ui[a:a + rc] = (zu[a:a + rc].bfloat16() @ Eh - En).argmax(-1)
                out[f"sel_recall_p{i}_L{j+1}"] = hit / N
                recalls.append(hit / N)
                qg = lvl.codebook(gi).reshape_as(rg); rg = rg - qg; pg.append(qg)
                qu = lvl.codebook(ui).reshape_as(ru); ru = ru - qu; pu.append(qu)
            out[f"sel_recall_p{i}_mean"] = sum(recalls) / len(recalls)
            zq_g.append(torch.stack(pg, -2).sum(-2))
            zq_u.append(torch.stack(pu, -2).sum(-2))
        Lf = torch.tensor([[sp.depth for sp in specs]],
                          device=x.device).expand(B, -1)
        og = model.decode(zq_g, Lf).float()
        ou = model.decode(zq_u, Lf).float()
        out["sel_gated_cos"] = (og * x).sum(1)[valid].mean().item()
        out["sel_ungated_cos"] = (ou * x).sum(1)[valid].mean().item()
        return out


def build_conditioning(codec, cond_cfg):
    """Attach the configured conditioning submodule to a PyramidCodec."""
    loc_dim = 256 if cond_cfg.location is not None else 0
    if cond_cfg.kind == "select":
        codec.selector = CodebookSelector(
            codec.specs, loc_dim=loc_dim, proj_dim=cond_cfg.proj_dim,
            subset_frac=cond_cfg.subset_frac, sel_n=cond_cfg.sel_n,
            temp=cond_cfg.temp, tau=cond_cfg.tau)
    else:
        prior = None
        if cond_cfg.coarse_prior_ckpt is not None:
            assert cond_cfg.location is not None, \
                "coarse_prior_ckpt needs conditioning.location set"
            from georvq.model.location import load_coarse_prior
            prior, loc_dim = load_coarse_prior(cond_cfg.coarse_prior_ckpt,
                                               codec.specs[0].dim, "cpu")
        codec.conditioner = TierPredictor(
            codec.specs, loc_dim=loc_dim, coarse_prior=prior,
            hidden=cond_cfg.hidden, layers=cond_cfg.layers)
