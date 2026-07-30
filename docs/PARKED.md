# Parked ideas

Validated-but-deferred work. Each entry records the evidence gathered at
parking time so it can be picked up without re-deriving it.

## Error-envelope predictor (decoder-side pixel uncertainty)

*Parked 2026-07-28. Feasibility: confirmed empirically, see below.*

**Idea.** A small network that, given only what a decoder has (quantized
tokens + depth tuple — never the original), predicts the distribution of
per-pixel reconstruction error (`1-cos`): quantiles p50/p90/p99 or a
log-normal (mu, sigma) per fine cell. Users get "which pixels are likely
troublesome" per operating point, and — via the depth conditioning —
"how much would this region improve if I fetched deeper levels", which
pairs with the progressive container's prefix reads.

**Evidence it works** (val split of ae_pyr64_divfull, July 2026):

- The 24 per-(tier, level) RVQ *increment norms* — magnitudes of the
  codebook vectors the decoder already sums, i.e. pure bitstream
  features — predict log window-mean error at **r = 0.94 with a linear
  probe** (better than the encode-side coarse-residual norm, 0.898).
  RVQ non-convergence is visible from indices alone.
- Precision of the linear probe alone: 42% of predicted top-5% windows
  are in the true worst 5%.
- Error tail structure (320M-pixel analysis): outliers are coarse-tier
  codebook misses (window-scale, corr 0.744 with coarse residual norm);
  mid/fine quantization contributes ~nothing to the tail. Pixel error
  histogram is roughly log-normal.

**Build sketch** (one evening on GPU1; all infra exists):

- Reuse `scripts/train_lite_decoder.py` skeleton: same window-index
  cache, strip reads, depth sampling, AdaLN depth conditioning. Swap the
  head for K quantiles per 4px cell (pinball loss) or (mu, sigma) NLL.
  ~1–2M params; feed increment norms as auxiliary channels.
- True error targets computed on the fly against the frozen codec
  (original needed at training only).
- **Calibrate**: split-conformal on held-out chips for finite-sample
  marginal coverage of the envelope (predicted p99 covers >= 99%);
  reliability curves per depth tuple. Eval: coverage, Spearman vs true
  error, precision@k against the saved tail data
  (`tail_analysis.npz` in the 2026-07-28 job dir).

**Caveats recorded at parking time:**

- Conditional coverage on truly novel regions is only marginal-, not
  per-region-guaranteed; the increment-norm features mitigate (they
  measure non-convergence, not content).
- Train AFTER the coarse-residual-sidecar decision: the sidecar
  collapses the error tail (worst-0.1% pixel median 0.035 -> 0.009 for
  66 B/window at 2 bits), so the predictor should be conditioned on
  sidecar presence or trained against sidecar-on decodes.

## Lite decoder family (light/medium)

*Parked 2026-07-28: quality below the release bar (Ali). Single deep
decoder ships in the beta.*

Trained variants on the frozen divfull bitstream
(`scripts/train_lite_decoder.py`; ckpts `checkpoints/ae_lite_{light,
medium}` in the release worktree): light 0.76M params, tier-0 only,
4px head + x16 bilinear — val cos 0.873 @ 88 bits, 36.6k win/s GPU;
medium 4.25M, tiers 0+1, 16px head — 0.971 @ 296 b / 0.983 @ 1752 b vs
heavy 0.977/0.989, 22.3k win/s (17x heavy). Parked because 0.87 (and
-0.006 for medium) is not shippable quality; revisit if a
thumbnail/browse product tier appears, or scale the variants up (the
16px-head + interpolation trade was the main quality cost).

**Status of previously-pending decisions**: the coarse-residual (8*)
sidecar SHIPPED (on by default in the AE encoder); the lite decoder
family is parked above. The deblocker SHIPPED 2026-07-30 as the second
AE decoder (`georvq-ae-tile decode --decoder deblock`, canonical
512+64-halo recipe — see docs/AE_RELEASE.md *Decoders* and
`georvq/ae/deblock.py`).
