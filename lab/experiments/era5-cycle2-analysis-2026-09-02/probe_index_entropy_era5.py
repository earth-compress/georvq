"""Probe: RVQ index entropy over the whole corpus — order-0 and conditional.

Encodes every store chip (3x3 window grid, full coverage) to full-depth RVQ
indices, then per (tier, level) reports:
  - nominal bits  log2(K)
  - order-0 H     empirical entropy -> dead codes / effective K
  - conditional H cross-entropy of a small MLP whose context is *all prior
    codebook entries*: the within-cell accumulated quantized vector (levels
    < l), the parent-tier cells' full-depth accumulated vectors (tiers < i),
    and a learned cell-position embedding.
The conditional CE is the rate an rANS coder with that context model
achieves, so "effective K" = 2^H is how far the codebooks could shrink
after entropy coding. Corpus-fit numbers: the predictor trains and
evaluates on the same (whole) corpus, matching closed-distribution use.

  python scripts/probe_index_entropy.py \
      --ckpt checkpoints/dem128_overnight_resume/model_best.pt \
      --store-dir /data/georvq-stores/copdem_356
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn

from georvq.data.store import open_store
from georvq.factory import from_checkpoint


@torch.no_grad()
def encode_corpus(model, store, window, batch, device, max_chips=None,
                  grid_n=3):
    """Full-depth greedy RVQ indices for every window of every chip.
    Returns [(N, cells, depth) int16 per tier], N windows kept."""
    rows = store.index.sort_values("shard")
    if max_chips:
        rows = rows.head(max_chips)
    offs = np.linspace(0, store.chip_px - window, grid_n, dtype=int)
    specs = model.specs
    N = len(rows) * len(offs) ** 2
    outs = [np.empty((N, sp.cells, sp.depth), np.int16) for sp in specs]
    buf, n, shard = [], 0, {}

    def flush():
        nonlocal buf, n
        if not buf:
            return
        raw = torch.from_numpy(np.stack(buf))
        x, _ = model.source.prep(raw, device)
        zs = model.encode_levels(x, model._stats_of(x))
        for i, (z, rvq) in enumerate(zip(zs, model.rvqs)):
            residual = z.float()
            for j, lev in enumerate(rvq.levels):
                q, idx, _ = lev(residual)
                residual = residual - q
                outs[i][n:n + len(buf), :, j] = \
                    idx.view(len(buf), -1).cpu().numpy()
        n += len(buf)
        buf = []

    for _, r in rows.iterrows():
        sh = int(r["shard"])
        if sh not in shard:
            shard = {sh: store.open_shard(sh)}   # one mmap at a time
        a = np.asarray(shard[sh][int(r["slot"])])
        for rr in offs:
            for cc in offs:
                t = a[:, rr:rr + window, cc:cc + window]
                if store.nodata is None or (t[0] != store.nodata).any():
                    buf.append(np.ascontiguousarray(t))
                if len(buf) == batch:
                    flush()
    flush()
    return [o[:n] for o in outs], n


def entropy_bits(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


class LevelPredictor(nn.Module):
    """(within-cell accum, parent accums, cell position) -> K logits.
    Probe-sized: a dense hidden x K output head dominates the params."""

    def __init__(self, d_in, cells, k, hidden=1024, pos_dim=32):
        super().__init__()
        self.pos = nn.Embedding(cells, pos_dim)
        self.net = nn.Sequential(
            nn.Linear(d_in + pos_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, k))

    def forward(self, feats, cell):
        return self.net(torch.cat([feats, self.pos(cell)], 1))


class TinyPredictor(nn.Module):
    """Deployable variant: the output head is tied to the tier's frozen
    codebook — the net predicts a d-dim direction in code space and logits
    are scaled dot products against the code vectors (+ a per-code usage
    bias). Head cost drops from hidden*K to hidden*d + K."""

    def __init__(self, d_in, cells, codebook, hidden=256, pos_dim=32,
                 bias_init=None):
        super().__init__()
        self.register_buffer("E", codebook.detach().clone())  # ships w/ codec
        self.pos = nn.Embedding(cells, pos_dim)
        self.net = nn.Sequential(
            nn.Linear(d_in + pos_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, codebook.shape[1]))
        self.bias = nn.Parameter(torch.zeros(codebook.shape[0])
                                 if bias_init is None else bias_init)
        self.scale = nn.Parameter(torch.zeros(()))   # start at order-0

    def forward(self, feats, cell):
        h = self.net(torch.cat([feats, self.pos(cell)], 1))
        return self.scale * (h @ self.E.t()) + self.bias

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
class SharedTierModel(nn.Module):
    """One entropy model per tier: a single shared context projection
    (exploits useful/redundant context dims once, not per level) feeding
    per-level MLPs with rank-factorized output heads."""

    def __init__(self, d_in, cells, ks, freqs, proj_dim=96, hidden=256,
                 rank=128, pos_dim=32):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, proj_dim), nn.GELU())
        self.pos = nn.Embedding(cells, pos_dim)
        self.heads = nn.ModuleList()
        for k, freq in zip(ks, freqs):
            out = nn.Linear(rank, k)
            out.bias.data.copy_(freq)
            self.heads.append(nn.Sequential(
                nn.Linear(proj_dim + pos_dim, hidden), nn.GELU(),
                nn.Linear(hidden, rank), out))

    def forward(self, feats, cell, ell):
        h = torch.cat([self.proj(feats), self.pos(cell)], 1)
        return self.heads[ell](h)


def accum_full(idx_np, levels, device, batch=1 << 18):
    """Full-depth accumulated codebook vectors: (N, cells, d) float16 cpu."""
    N, cells, depth = idx_np.shape
    flat = torch.from_numpy(idx_np.reshape(-1, depth).astype(np.int64))
    out = torch.empty(N * cells, levels[0].codebook.weight.shape[1],
                      dtype=torch.float16)
    for s in range(0, flat.shape[0], batch):
        ii = flat[s:s + batch].to(device)
        acc = sum(levels[j].codebook.weight[ii[:, j]] for j in range(depth))
        out[s:s + batch] = acc.half().cpu()
    return out.view(N, cells, -1)


def parent_cell(c, g, gp):
    """Row-major cell index in the coarser grid gp containing cell c of
    grid g."""
    if gp == 1:
        return torch.zeros_like(c)
    ratio = g // gp
    return (c // g // ratio) * gp + (c % g) // ratio


def context_dim(mode, i, dims):
    """Input feature width for a (mode, tier) predictor; dims = per-tier
    latent dims."""
    return {"all": dims[i] + sum(dims[:i]),
            "prev1": dims[i],
            "prev2": 2 * dims[i],
            "first_own": dims[i],
            "first_all": dims[i] + sum(dims[:i])}[mode]


def gather_context(mode, i, ell, w, c, idxs, levels, parents, device):
    """Context features for windows w / cells c at (tier i, level ell).

    all:       accumulated entries of ALL prior levels + coarser tiers'
               full-depth accumulated vectors
    prev1:     the entry chosen at level ell-1 only
    prev2:     the entries at ell-1 and ell-2
    first_own: the entry at level 0 of the same tier/cell
    first_all: level-0 entries of this cell and of the containing cell in
               every coarser tier
    Levels without a defined context (e.g. ell=0 for prev/first_own) get a
    zero vector; the predictor still sees the cell-position embedding."""
    feats = []
    wn, cn = w.numpy(), c.numpy()
    d = levels[i][0].codebook.weight.shape[1]
    zero = lambda: torch.zeros(len(w), d, device=device)

    def own_entry(j):
        ii = torch.from_numpy(idxs[i][wn, cn, j].astype(np.int64)).to(device)
        return levels[i][j].codebook.weight[ii]

    g = int(math.isqrt(idxs[i].shape[1]))
    if mode == "all":
        feats.append(sum(own_entry(j) for j in range(ell)) if ell > 0
                     else zero())
        for p, acc in enumerate(parents[:i]):
            pc = parent_cell(c, g, int(math.isqrt(acc.shape[1])))
            feats.append(acc[w, pc].to(device).float())
    elif mode == "prev1":
        feats.append(own_entry(ell - 1) if ell > 0 else zero())
    elif mode == "prev2":
        feats.append(own_entry(ell - 1) if ell >= 1 else zero())
        feats.append(own_entry(ell - 2) if ell >= 2 else zero())
    elif mode == "first_own":
        feats.append(own_entry(0) if ell > 0 else zero())
    elif mode == "first_all":
        feats.append(own_entry(0) if ell > 0 else zero())
        for p in range(i):
            gp = int(math.isqrt(idxs[p].shape[1]))
            pc = parent_cell(c, g, gp)
            ii = torch.from_numpy(
                idxs[p][wn, pc.numpy(), 0].astype(np.int64)).to(device)
            feats.append(levels[p][0].codebook.weight[ii])
    return torch.cat(feats, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-chips", type=int, default=None)
    ap.add_argument("--grid-n", type=int, default=3,
                    help="windows per chip axis (chip_px/window for exact "
                         "full coverage)")
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--train-batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-samples", type=int, default=20_000_000)
    ap.add_argument("--save-idx", default=None,
                    help="optionally save index arrays (npz) for reuse")
    ap.add_argument("--load-idx", default=None,
                    help="load index arrays saved by --save-idx (skips "
                         "encoding; arrays are already shuffled)")
    ap.add_argument("--context", default="all",
                    choices=["all", "prev1", "prev2", "first_own",
                             "first_all"])
    ap.add_argument("--arch", default="probe",
                    choices=["probe", "tiny", "shared"],
                    help="probe: dense K-wide heads (measures ceiling); "
                         "tiny: codebook-tied heads (deployable size); "
                         "shared: per-tier shared context projection + "
                         "rank-factorized heads")
    ap.add_argument("--proj-dim", type=int, default=96)
    ap.add_argument("--head-rank", type=int, default=128)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() and args.gpu >= 0
                          else "cpu")
    torch.set_float32_matmul_precision("high")

    store = open_store(args.store_dir)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "config" in ck:
        model = from_checkpoint(args.ckpt, device).eval()
    else:
        from georvq.compat import from_legacy_checkpoint
        model = from_legacy_checkpoint(args.ckpt, device)[0].eval()
    del ck
    specs = model.specs
    if args.load_idx:
        z = np.load(args.load_idx)
        idxs = [z[f"tier{i}"] for i in range(len(specs))]
        N = idxs[0].shape[0]
        print(f"loaded {N} windows from {args.load_idx}", flush=True)
    else:
        idxs, N = encode_corpus(model, store, args.window, args.batch,
                                device, args.max_chips, args.grid_n)
        print(f"encoded {N} windows from {len(store.index)} chips",
              flush=True)
        perm = np.random.default_rng(0).permutation(N)
        idxs = [a[perm] for a in idxs]   # random held-out, not shard tail
        if args.save_idx:
            np.savez(args.save_idx,
                     **{f"tier{i}": a for i, a in enumerate(idxs)})

    levels = [list(rvq.levels) for rvq in model.rvqs]
    run_entropy_models(idxs, levels, specs, args, device,
                       px=args.window ** 2)


def run_entropy_models(idxs, levels, specs, args, device, px):
    """Order-0 + conditional entropy analysis over pre-encoded index
    arrays. `levels[i][j]` needs only `.codebook.weight`; `specs[i]` needs
    cells/depth/codes. Reusable across codecs (see probe_index_entropy_s2)."""
    N = idxs[0].shape[0]
    print(f"context mode: {args.context}", flush=True)
    parents = ([accum_full(idxs[i], levels[i], device)
                for i in range(len(specs) - 1)]
               if args.context == "all" else [])

    rows = []
    n_params = 0
    for i, sp in enumerate(specs):
        cells = sp.cells
        if args.arch == "shared":
            # one jointly-trained model per tier: shared context projection,
            # per-level heads; same total step budget per level
            dims = [lv[0].codebook.weight.shape[1] for lv in levels]
            d_in = context_dim(args.context, i, dims)
            lvls = [l for l in range(sp.depth) if not (i == 0 and l == 0)]
            all_counts = [np.bincount(idxs[i][:, :, l].ravel(),
                                      minlength=sp.codes)
                          for l in range(sp.depth)]
            freqs = [torch.from_numpy(np.log(all_counts[l] + 0.5)
                                      - np.log(all_counts[l].sum())).float()
                     for l in lvls]
            tm = SharedTierModel(d_in, cells, [sp.codes] * len(lvls), freqs,
                                 proj_dim=args.proj_dim, hidden=256,
                                 rank=args.head_rank).to(device)
            n_params += sum(p.numel() for p in tm.parameters())
            opt = torch.optim.AdamW(tm.parameters(), lr=args.lr,
                                    weight_decay=0.01)
            for step in range(args.train_steps * len(lvls)):
                ell = lvls[step % len(lvls)]
                w = torch.randint(0, N, (args.train_batch,))
                c = torch.randint(0, cells, (args.train_batch,))
                y = torch.from_numpy(
                    idxs[i][w.numpy(), c.numpy(), ell].astype(np.int64)
                ).to(device)
                with torch.no_grad():
                    feats = gather_context(args.context, i, ell, w, c,
                                           idxs, levels, parents, device)
                loss = nn.functional.cross_entropy(
                    tm(feats, c.to(device), lvls.index(ell)), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tm.eval()
            for ell in range(sp.depth):
                h0 = entropy_bits(all_counts[ell])
                dead = int((all_counts[ell] == 0).sum())
                hc = h0
                if ell in lvls:
                    take = min(args.eval_samples, N * cells)
                    perm = torch.randperm(N * cells)[:take]
                    ce = 0.0
                    with torch.no_grad():
                        for s in range(0, take, 1 << 16):
                            sel = perm[s:s + (1 << 16)]
                            w, c = sel // cells, sel % cells
                            y = torch.from_numpy(
                                idxs[i][w.numpy(), c.numpy(), ell]
                                .astype(np.int64)).to(device)
                            feats = gather_context(args.context, i, ell, w,
                                                   c, idxs, levels, parents,
                                                   device)
                            ce += float(nn.functional.cross_entropy(
                                tm(feats, c.to(device), lvls.index(ell)),
                                y, reduction="sum"))
                    hc = ce / take / math.log(2)
                rows.append((i, ell, math.log2(sp.codes), h0, dead, hc,
                             cells))
                print(f"tier{i} L{ell}: log2K={math.log2(sp.codes):.1f} "
                      f"H0={h0:.2f} dead={dead} Hcond={hc:.2f} "
                      f"(effK {2 ** hc:.0f})", flush=True)
            continue
        for ell in range(sp.depth):
            counts = np.bincount(idxs[i][:, :, ell].ravel(),
                                 minlength=sp.codes)
            h0 = entropy_bits(counts)
            dead = int((counts == 0).sum())

            hc = None
            if not (i == 0 and ell == 0):        # tier0/L0 has no context
                dims = [lv[0].codebook.weight.shape[1] for lv in levels]
                d_in = context_dim(args.context, i, dims)
                if args.arch == "tiny":
                    freq = torch.from_numpy(
                        np.log(counts + 0.5) - np.log(counts.sum())).float()
                    pred = TinyPredictor(
                        d_in, cells, levels[i][ell].codebook.weight,
                        bias_init=freq).to(device)
                    n_params += pred.n_params()
                else:
                    pred = LevelPredictor(d_in, cells, sp.codes).to(device)
                    n_params += sum(p.numel() for p in pred.parameters())
                opt = torch.optim.AdamW(pred.parameters(), lr=args.lr,
                                        weight_decay=0.01)
                target = idxs[i][:, :, ell]
                # closed-distribution corpus fit: train on ALL windows and
                # report in-sample CE — the rate this model achieves when
                # shipped with the archive (overfitting is legal here)
                n_tr = N
                for step in range(args.train_steps):
                    w = torch.randint(0, n_tr, (args.train_batch,))
                    c = torch.randint(0, cells, (args.train_batch,))
                    y = torch.from_numpy(
                        target[w.numpy(), c.numpy()].astype(np.int64)
                    ).to(device)
                    with torch.no_grad():
                        feats = gather_context(args.context, i, ell, w, c,
                                               idxs, levels, parents, device)
                    loss = nn.functional.cross_entropy(
                        pred(feats, c.to(device)), y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                pred.eval()

                def ce_on(w_lo, w_hi):
                    tot = (w_hi - w_lo) * cells
                    take = min(args.eval_samples, tot)
                    perm = torch.randperm(tot)[:take]
                    ce = 0.0
                    with torch.no_grad():
                        for s in range(0, take, 1 << 16):
                            sel = perm[s:s + (1 << 16)]
                            w, c = w_lo + sel // cells, sel % cells
                            y = torch.from_numpy(
                                target[w.numpy(), c.numpy()]
                                .astype(np.int64)).to(device)
                            feats = gather_context(args.context, i, ell, w,
                                                   c, idxs, levels, parents,
                                                   device)
                            ce += float(nn.functional.cross_entropy(
                                pred(feats, c.to(device)), y,
                                reduction="sum"))
                    return ce / take / math.log(2)

                hc = ce_on(0, N)

            rows.append((i, ell, math.log2(sp.codes), h0, dead,
                         hc if hc is not None else h0, cells))
            r = rows[-1]
            print(f"tier{i} L{ell}: log2K={r[2]:.1f} H0={r[3]:.2f} "
                  f"dead={r[4]} Hcond={r[5]:.2f} "
                  f"(effK {2 ** r[5]:.0f})", flush=True)

    nom = sum(r[2] * r[6] for r in rows)
    h0t = sum(r[3] * r[6] for r in rows)
    hct = sum(r[5] * r[6] for r in rows)
    print(f"\nfull-ladder bits/window: nominal {nom:,.0f} ({nom / px:.4f} "
          f"bpp) -> order-0 {h0t:,.0f} ({h0t / px:.4f}) "
          f"-> conditional {hct:,.0f} ({hct / px:.4f} bpp, "
          f"{100 * (1 - hct / nom):.1f}% below nominal)")
    print(f"entropy model: {n_params / 1e6:.2f}M params "
          f"({n_params * 4 / 2 ** 20:.1f} MiB fp32), arch={args.arch}, "
          f"context={args.context}")


if __name__ == "__main__":
    main()
