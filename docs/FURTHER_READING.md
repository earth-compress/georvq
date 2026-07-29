# Further reading: architecture + file-by-file guide

Shorter orientation available at: `docs/OVERVIEW.md`
(invariants, personas). The shipped AlphaEarth codec's contract:
`docs/AE_RELEASE.md`. Empirical design history: `docs/FINDINGS.md`.
Live demo: `notebooks/demo_ae_codec.ipynb`.

---

## Part 1 — The model

georvq compresses geospatial rasters with a **pyramidal residual-VQ
autoencoder**: a ViT encoder summarizes each fixed-size window into a
small pyramid of latent grids, each grid is quantized by a stack of
residual codebooks, and an attention decoder turns any *prefix* of
those codes back into pixels. One trained model serves an entire
rate–distortion curve.

Numbers below are the shipped AlphaEarth release (`ae_pyr64_divfull`):
64 px windows over 64-band unit-norm embedding pixels.

### 1.1 End-to-end dataflow

```
          one 64x64 px window, 64 bands (prepped, unit-norm pixels)
                                 |
                        source stem (AE: identity)
                                 |
                     ViTTokenizer (dim 256, 9 blocks)
          CLS token ---+---------+---------+
                       |         |         |
              tier 0   |  tier 1 |  tier 2 |         "the pyramid"
             (coarse)  |  (mid)  |  (fine) |
            1 x 1 cell | 4 x 4   | 16 x 16 |   <- cells per window
              256-d    | 192-d   |  64-d   |   <- latent dim per cell
                       |         |         |
                 ResidualVQ  ResidualVQ  ResidualVQ      8 levels each
                 K = 2048    K = 8192    K = 16384       (codebook size)
                       |         |         |
                     [ code indices = the bitstream ]
                       |         |         |
              PyramidAttnDecoder (dim 384, 12 AdaLN blocks;
              full self-attention over all 1 + 16 + 256 tokens,
              conditioned on how many levels each tier received)
                                 |
               DenseHead: fine tokens -> 4 px patches -> conv
               smoothing -> 64-band plane -> per-pixel L2 norm
                                 |
                     reconstructed 64x64 window
```

Why a pyramid: coarse content (land-cover identity of the whole
window) is cheap to describe and shared across many pixels; fine
texture is expensive and local. Giving each scale its own latent grid
and codebooks lets the rate go where the information is. The decoder
sees all tiers at once (full attention), so coarse tokens provide
context that makes the fine codes cheaper to use.

### 1.2 Residual VQ in one picture

Each tier's cell vector is quantized by a *stack* of codebooks, each
encoding what the previous levels missed:

```
   z ────────────────► nearest code in book 1 ──► q1
   z - q1 ───────────► nearest code in book 2 ──► q2
   z - q1 - q2 ──────► nearest code in book 3 ──► q3
        ...                     ...                     (8 levels)

   reconstruction at depth L  =  q1 + q2 + ... + qL
```

Each level is one integer per cell (11/13/14 bits for the three
tiers). The series converges geometrically for in-distribution
content; the size of late increments is itself a usable signal of how
well the quantizer is coping (we use it as an error predictor).

Codebooks are trained by EMA with health machinery (k-means init,
dead-code resets, decay ramps) — that's most of `model/vq.py` and it is
**parity-sacred**: the repo's value is that trained checkpoints decode
bit-faithfully forever.

### 1.3 Matryoshka: one model, every rate

During training each window is decoded at a *random monotone depth
tuple* (L0, L1, L2) — e.g. (8,4,0) — and the decoder is conditioned on
the tuple (a thermometer code feeding AdaLN). So at inference, indices
are stored at full depth and any prefix decodes well:

```
  bits/window:  88        1,752        5,336                 30,952
                |           |            |                      |
  tuple:      8-1-0  ...  8-8-0  ...   8-8-1   ...   8-8-8 + 8* sidecar
  val cos:    0.977       0.990        0.994                  0.998
                |___________|____________|______________________|
                     the measured Pareto ladder: max coarse ->
                     fill mid -> add fine levels one at a time
```

Level costs are wildly asymmetric — coarse 11 b, mid 208 b, fine
3,584 b per window — which is why the efficient ladder saturates the
cheap tiers first (measured, not assumed: a full tuple sweep on the
val set; e.g. 6-4-4 is strictly dominated).

**The 8\* sidecar**: even at depth 8 the coarse tier leaves a small
residual, and that residual carries most of the *outlier* error (rare
land covers the 2048-code book can't cover; window-scale misses that
finer tiers can't absorb). Shipping it as a 2-bit scalar-quantized
vector (66 B/window) cuts the 99.9th-percentile pixel error ~3x for
+1.5% rate. Mid/fine residuals measured worthless (their codebooks sit
at their ceiling — and are 4.6x cheaper than scalar-quantized vectors
at equal quality, which is the codebooks earning their keep).

### 1.4 Optional submodules (not in the AE release)

Variants are the *presence of submodules*, never runtime flags:
`TierPredictor` (cross-tier predict-and-subtract; cascade),
`CodebookSelector` (location-gated codebook subsets),
`PyramidRateModel` (learned entropy model). All exist, all trained,
none shipped for AE — `docs/FINDINGS.md` records why.

### 1.5 Deployment shape (AE release)

```
   source COG tile (8192 px, 2.25 GB)
        └─ encode (GPU, ~76 s incl. self-measured quality stats)
             ├─ coarse.tif   128x128x8   uint16   ─┐  indices as
             ├─ mid.tif      512x512x8   uint16    ├─ georeferenced
             ├─ fine.tif    2048x2048x8  uint16   ─┘  COG rasters
             ├─ star.tif     8* sidecar  (66 B/window)
             ├─ mask.tif     validity (exact nodata restore)
             ├─ quality/<decoder_id>.tif  measured err/window/rate-point
             └─ metadata.json             stats, ids, source profile
   total ~58 MB; bands = RVQ levels -> rate control = band selection;
   blocks mirror the source's 1024 px quantum -> ranged reads via
   plain GDAL. Two ids guard correctness: bitstream_id (codebooks;
   wrong model refused) and decoder_id (keys quality rasters).
```

---

## Part 2 — File by file

Personas: **[R]** codec researcher · **[E]** release engineer ·
**[U]** AE data end user (users normally touch only `georvq/ae` CLIs
and `docs/AE_RELEASE.md`).

### Top level

| file | what it is |
|---|---|
| `README.md` | The method, for first contact. |
| `CLAUDE.md` | Working agreements for AI-assisted development. |
| `pyproject.toml` | Package, extras (`[loc]` SatCLIP, `[archive]` rANS, `[eval]` geo/benchmarks, `[viz]`, `[onnx]`), console entry points. |

### `georvq/` core **[R]**

| file | what it is |
|---|---|
| `config.py` | The reproducibility spine: pydantic-v2 `Config` (`extra="forbid"`), YAML loader with `inherit:` + dot-overrides, `config_hash`/`config_diff`. Anything that affects a run lives here. |
| `factory.py` | `build_model(cfg)` assembles the codec; `from_checkpoint(path)` rebuilds any checkpoint exactly from its embedded config. |
| `compat.py` | Loads legacy research-repo checkpoints: prefix-renames state dicts, reconstructs a `Config` from saved argv. State-dict names are load-bearing. |
| `metrics.py` | `masked_recon_stats` (mse + cos over valid pixels), `masked_r2`, `centered_cos` (spatial-structure probe that a block-mean recon can't fool). |
| `repro.py` | Seeding, git/env capture, `save_checkpoint` (embeds resolved config + hash). |
| `cli.py` | `georvq-train`, `georvq-train-stage1` entry points. |

### `georvq/model/` — the architecture **[R]**

| file | what it is |
|---|---|
| `codec.py` | `PyramidCodec`: composition root (stem → tokenizer → RVQs → decoder), monotone depth sampling, `level_cumsums`/`eval_tuple` (encode once, decode any rate), `quantize()` as the single routing point for plain/skip-masked/cascade/gated paths. |
| `vq.py` | `VectorQuantizer`/`ResidualVQ`: EMA codebooks, k-means init, dead-code resets (batch/split/fps), repulsion, bf16 assignment scores. VQ state stays fp32 under autocast. |
| `tokenizer.py` | `ViTTokenizer`: conv patch embeds per tier + transformer trunk; CLS token is the coarse tier. |
| `decoder.py` | `PyramidAttnDecoder` (AdaLN-Zero blocks over all tier tokens, depth-tuple/sigma conditioning, optional cls-bridge lineage concat) + `DenseHead` (patch projection + conv seam smoothing + optional per-pixel L2 norm). |
| `layers.py` | Shared blocks: `AdaLNBlock`, `thermometer` depth encoding, sigma embedding. |
| `rate.py` | Learned entropy models (`ContextRateModel`, `PyramidRateModel`), differentiable `soft_rate`, tau calibration. Trained, measured, not shipped. |
| `conditioning.py` | `TierPredictor` (cascade predict-and-subtract), `CodebookSelector` (location-gated subsets). |
| `location.py` | SatCLIP clone-on-demand loader, location tables, location→latent priors. |

### `georvq/data/` + `georvq/sources/` **[R]**

| file | what it is |
|---|---|
| `data/store.py` | Chip-store v1 (sharded int8 memmaps + parquet index + manifest), `geographic_split` (hashed lat/lon cells — never split randomly), `open_store` factory. |
| `data/pool.py` | Resident chip pools with rate-limited background refresh; seed-stable, not bit-identical. |
| `data/build.py` | `ShardWriter` for store builders. |
| `sources/__init__.py` | The `Source` interface + registry — sources own *all* data-specific behavior (prep, stem, head, metrics). |
| `sources/ae.py` | AlphaEarth adapter (minimal reference): signed-square dequant, −128 nodata, identity stem, normalized `DenseHead`. |
| `sources/s2.py` | Sentinel-2 adapter: the multi-resolution/multi-group worked example. |

### `georvq/train/` + `georvq/eval/` + `georvq/viz/` **[R]**

| file | what it is |
|---|---|
| `train/loop.py` | The one trainer: pools, autocast policy per stage, matryoshka sampling, validation cadence, wandb. |
| `train/stage1.py` | σ-conditioned continuous pretrain (no VQ) — the "unquantized first" recipe. |
| `train/validate.py`, `figures.py` | Canonical tuple sweeps, codebook-health tallies; R-D and recon figures. |
| `train/warmstart.py` | Checkpoint surgery for warm starts (codebook maturation rules). |
| `eval/extract.py`, `benchmarks.py`, `probes.py`, `codec_ab.py` | Held-out extraction, benchmark registry (ASTER-GED, LUCAS, LCMAP), frozen-embedding kNN/linear probes, codec A/B plumbing. |
| `viz/panels.py`, `style.py` | PCA-RGB panel/GIF builders and shared styling. |

### `georvq/inference/` — source-agnostic deployment **[E]**

| file | what it is |
|---|---|
| `api.py` | `Codec` (encode → `Packet`s of per-tier indices; decode any monotone tuple) and `Packet` (incl. the optional coarse-residual field). The deployment mirror of `level_cumsums`. |
| `release.py` | Release bundles (`save_release`/`load_release`) and the two identity hashes: `bitstream_id` (tiers+codebooks+priors — what defines the bitstream) and `decoder_id` (decoder weights — keys decoder-specific artifacts). |
| `onnx_backend.py` | ONNX export (encoder/decoder graphs, dynamic batch) + `OnnxCodec`, a torch-free runtime; RVQ stays numpy so bitstream math is shared bytes. Refuses conditioned models rather than mis-exporting. |
| `archive.py` | rANS coding over indices (compressai). Research tool — measured ~0.6% on AE and deliberately not shipped; kept for future context-model coders. |
| `stats_codec.py` | PCA + tolerance-coded per-window stats side-channel for normalized sources (S2-style); unused by AE (unit-norm input). |

### `georvq/ae/` — the AlphaEarth release layer **[E]/[U]**

Everything here is AE release *policy* (see `docs/AE_RELEASE.md`); a
new source should write its own release layer, using this as the
worked example.

| file | what it is |
|---|---|
| `cli.py` | The executables: `georvq-ae-tile` (encode/decode/info: the release codec), `georvq-ae-patch` (dev tool, npz windows), `georvq-ae-export-onnx`. Also: AE pixel prep/requant, the 2-bit 8\* sidecar (pack/unpack), the quality ladder + encode-time stats (fused GPU pass), rate/quality decode controls, torch/onnx backends. |
| `cog.py` | The COG asset container: per-tier index rasters (bands = levels), star/mask/quality assets, metadata, id tags — mirrors the AEF source's 1024 px block quantum by design. |

### `scripts/` — thin research drivers **[R]**

| file | what it is |
|---|---|
| `train_pyramid.py`, `train_stage1.py` | Entry-point bodies for the two trainers. |
| `train_lite_decoder.py` | Lightweight decoder variants on the frozen bitstream (parked — see `docs/PARKED.md`). |
| `build_store_ae.py` / `build_store_s2.py` | Store builders: equal-area sphere sampling of AEF blocks; S2 equivalent. |
| `build_archive.py`, `bench_decode.py` | rANS archive builder + decode benchmarks (research). |
| `baseline_classical.py` | PCA+J2K / HEVC baselines at matched granularity. |
| `eval_build_dataset.py`, `eval_run_ab.py`, `eval_rate_sweep.py` | Downstream benchmark datasets; original-vs-recon A/B probes; Pareto-front sweeps. |
| `pretrain_location_prior.py` | SatCLIP→coarse-latent prior pretraining (cascade input). |
| `make_recon_gif.py`, `make_panel_gifs.py` | Qualitative recon/rate GIFs. |

### `configs/`, `tests/`, `docs/`, `notebooks/`

| file | what it is |
|---|---|
| `configs/pyr64.yaml` | The flagship AE recipe (argv-matched to the final training stage; the full warm-start history is in the research repo). |
| `configs/stage1.yaml`, `hybrid_rate.yaml`, `cascade.yaml`, `satclip_select.yaml` | Historical recipes that ran and worked (provenance notes in OVERVIEW — one contains an explicitly unvalidated variant). |
| `configs/smoke.yaml` | Synthetic CPU-friendly end-to-end smoke. |
| `tests/` | Fast suite (~20 s): config/codec/VQ/conditioning/rate units, `test_vq_parity.py` (bit-exactness vs the research repo, self-skipping), `test_compat.py` (+ golden tokens), `test_inference.py`, `test_ae_codec.py` (containers, ONNX parity, CLIs end-to-end, sidecar, quality stats), `test_train_smoke.py` (GPU, marked slow). |
| `docs/OVERVIEW.md` | Start here: pipeline, module map, invariants, personas. |
| `docs/AE_RELEASE.md` | The shipped codec's complete contract. |
| `docs/BITSTREAM.md` | Why the bitstream excludes the decoder, and what changes an id. |
| `docs/ADDING_A_SOURCE.md` | The recipe for onboarding a new dataset. |
| `docs/FINDINGS.md` | Empirical results behind the design, including negative ones. |
| `docs/PARKED.md` | Validated-but-deferred ideas with their evidence (error-envelope predictor, lite decoders, deblocker). |
| `notebooks/demo_ae_codec.ipynb` | Executed demo: original → encode → codes → progressive decode → client/server with live byte counts (`make_demo_nb.py` regenerates it). |
