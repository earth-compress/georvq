# Further reading: architecture + file-by-file guide

Shorter orientation available at: `docs/OVERVIEW.md`
(invariants, personas). The shipped AlphaEarth codec's contract:
`docs/AE_RELEASE.md`. Empirical design history: `docs/FINDINGS.md`.
Live demo: `notebooks/demo_ae_codec.ipynb`.

---

## Part 1 — The model

### 1.1 The one-paragraph version

georvq is a *learned* image compressor for geospatial data. A classical
codec like JPEG applies one fixed mathematical recipe to everything;
georvq instead trains a neural network on the kind of data it will
compress, so the network learns what real landscapes tend to look like
and can describe them in far fewer bits. Each map tile is cut into
small windows; an **encoder** network boils each window down to a
handful of vectors; those vectors are snapped to entries in learned
lookup tables (**codebooks**), so that only small integer IDs need to
be stored; and a **decoder** network turns the IDs back into pixels.
Crucially, the IDs are ordered most-important-first, so one stored
file can be decoded at many quality levels — read a small prefix for a
rough preview, or all of it for a near-perfect reconstruction.

All concrete numbers below describe the shipped AlphaEarth release
(`ae_pyr64_divfull`), which works on 64×64-pixel windows where each
pixel is a 64-band unit-length embedding vector.

### 1.2 The pipeline, end to end

```
             one 64×64-pixel window (64 bands per pixel)
                               │
                               ▼
                 ┌───────────────────────────┐
                 │        ViT encoder        │  a vision transformer that
                 │      (ViTTokenizer)       │  reads the whole window at
                 └─────────────┬─────────────┘  once (9 blocks, width 256)
                               │
                               │  summarizes it at three scales:
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
      ┌───────────┐      ┌───────────┐      ┌───────────┐
      │  COARSE   │      │    MID    │      │   FINE    │
      │ 1×1 cell  │      │ 4×4 cells │      │16×16 cells│  ← grid over the window
      │   256-d   │      │   192-d   │      │   64-d    │  ← numbers per cell
      └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
            │  "the whole      │  "this          │  "this 4×4-px
            │   window is      │   quadrant-ish   │   patch has this
            │   farmland"      │   area is..."    │   texture"
            ▼                  ▼                  ▼
      residual VQ        residual VQ        residual VQ      8 codebooks
      2,048 entries      8,192 entries      16,384 entries   per tier (§1.3)
            │                  │                  │
            └──────────────────┼──────────────────┘
                               ▼
             ╔═══════════════════════════════════════╗
             ║  the bitstream: one integer ID per    ║
             ║  cell per codebook — this is all      ║
             ║  that ever gets written to disk       ║
             ╚═══════════════════════════════════════╝
                               │
                               ▼
                 ┌───────────────────────────┐
                 │    attention decoder      │  all 1+16+256 cells attend to
                 │   (PyramidAttnDecoder)    │  each other (12 blocks, width
                 └─────────────┬─────────────┘  384); it is also *told* how
                               │                many codebook levels of each
                               ▼                tier it actually received
                 ┌───────────────────────────┐
                 │        DenseHead          │  each fine cell → a 4×4-px
                 └─────────────┬─────────────┘  patch; a small conv smooths
                               │                the seams; pixels re-scaled
                               ▼                to unit length
              reconstructed 64×64-pixel window
```

Two design choices carry most of the weight here:

**Why three scales?** Because information in imagery is not spread
evenly across scales. "This window is mostly farmland" is one fact
shared by 4,096 pixels — describing it once at the coarse tier is
almost free. Fine texture is the opposite: local, unpredictable, and
expensive. Giving each scale its own grid and its own codebooks lets
the bit budget flow to where the information actually is.

**Why does the decoder see everything at once?** The decoder's
attention spans all three tiers, so the fine cells are always
interpreted *in context*. A fine code doesn't have to spell out "dark
green, vegetation-like" from scratch — the coarse tier already
established that — so the same fine codebook entry can mean subtly
different things in different surroundings. That context-sharing is a
big part of why so few bits suffice.

### 1.3 Codebooks, and quantizing by successive approximation

A **codebook** is a learned lookup table: a list of, say, 2,048
prototype vectors that between them cover the kinds of cell summaries
the encoder actually produces. To store a cell you don't store its
numbers — you find the closest prototype and store its index. 2,048
entries means 11 bits, no matter how many dimensions the vector has.

One lookup alone is too crude, so georvq quantizes in rounds — like a
game of "warmer/colder", or writing a number digit by digit. Each
round has its *own* codebook, trained specifically on whatever the
previous rounds failed to capture:

```
   target vector z
        │
        ▼
   book 1: closest of its 2,048 entries ─► q₁      a rough first guess
        │
        ▼   what's still missing:  z − q₁
   book 2: closest entry to that  ─► q₂            a correction
        │
        ▼   still missing:  z − q₁ − q₂
   book 3: closest entry to that  ─► q₃            a finer correction
        │
       ...                                          (8 rounds in total)

   stored on disk:   8 small integers per cell
   reconstruction:   q₁ + q₂ + ... + qL  — and you may stop at any L
```

This is **residual vector quantization** (residual VQ, or RVQ). Each
round costs one integer per cell — 11, 13, or 14 bits for the three
tiers. The corrections shrink rapidly round over round on typical
content, and the size of the *late* corrections turns out to be a
useful health signal: if round 7 is still making big moves, the
quantizer is struggling with this window, and we use exactly that as
an error predictor.

The codebooks themselves are learned during training (by exponential
moving averages, with machinery to keep every entry earning its keep —
k-means initialization, resets for entries that fall out of use).
That's most of `model/vq.py`, and it is **parity-sacred**: the entire
value of the repo is that a trained checkpoint decodes bit-identically
forever, so these numerics must never drift.

### 1.4 One model, every compression level

Most codecs make you pick a quality setting at *encode* time. georvq
is trained so the choice can wait until *decode* time — nesting
qualities inside one bitstream the way matryoshka dolls nest.

The trick is in training: for every window, the decoder is handed a
randomly chosen number of quantization rounds per tier — maybe all 8
coarse rounds, 4 mid, and no fine at all — and is explicitly told what
it got (the depth tuple conditions the network). Having practiced at
every budget, it degrades gracefully instead of falling apart when
codes are missing. At encode time everything is stored at full depth
once; readers then decode whatever prefix suits them:

```
  reconstruction quality (cosine similarity, held-out data)

  0.998 ┤                                              ● 8-8-8 + sidecar
  0.994 ┤                    ● 8-8-1
  0.990 ┤        ● 8-8-0
  0.977 ┤ ● 8-0-0
        └──┬─────────┬─────────┬─────────────────────────┬──
           88      1,752     5,336                    30,952
                                                bits per window

  "8-8-1" = all 8 coarse rounds, all 8 mid, 1 fine.
```

The tiers have wildly different price tags — one round costs 11 bits
at the coarse tier, 208 at mid (13 bits × 16 cells), 3,584 at fine
(14 bits × 256 cells). So the efficient path up the quality ladder is:
saturate coarse first, then fill mid, then add fine rounds one at a
time. That ordering is measured, not assumed — a full sweep of depth
combinations on the validation set confirms it (e.g. 6-4-4 is strictly
worse than same-cost alternatives).

### 1.5 The 8\* sidecar: catching the outliers

Even after all 8 rounds, the coarse tier's approximation isn't quite
exact — and that last sliver of error is concentrated in *unusual*
windows: rare land covers the 2,048-entry codebook never learned well,
whole-window effects the finer tiers can't make up for. Since averages
were already fine and only the outliers hurt, the release ships that
final coarse leftover directly, crudely rounded to 2 bits per number
(66 bytes per window). Cost: +1.5% file size. Benefit: the worst
pixel errors (99.9th percentile) drop about 3×.

The same trick was tested on the mid and fine tiers and found
worthless — their codebooks already capture everything they're going
to, and they do it ~4.6× cheaper than the crude-rounding approach at
equal quality. Which is a nice sanity check that the codebooks are
earning their keep.

### 1.6 Optional submodules (not in the AE release)

Model variants are expressed as the *presence of submodules*, never as
runtime flags: `TierPredictor` (predicts one tier from another and
codes only the surprise; "cascade"), `CodebookSelector` (restricts
codebook entries by geographic location), `PyramidRateModel` (a
learned model of which codes are likely, for entropy coding). All
exist and all were trained; none made the AlphaEarth release —
`docs/FINDINGS.md` records what each one bought and why it was left
out.

### 1.7 What actually lands on disk (AE release)

The unit of deployment is a satellite image tile, and the compressed
form is itself a set of ordinary georeferenced rasters — the code IDs
are just pixel values, so standard geospatial tooling (GDAL) can read
any region of any tier without touching the rest:

```
   source COG tile (8192 px, 2.25 GB)
        │
        ▼  encode (GPU, ~76 s — includes measuring its own quality)
        │
        ├─ coarse.tif    128×128, 8 bands  ─┐   the code IDs, one band
        ├─ mid.tif       512×512, 8 bands   ├─  per quantization round,
        ├─ fine.tif    2048×2048, 8 bands  ─┘   as georeferenced COGs
        ├─ star.tif      the 8* sidecar (66 B/window)
        ├─ mask.tif      validity mask (nodata restored exactly)
        ├─ quality/<decoder_id>.tif   measured error per window,
        │                             per quality level
        └─ metadata.json              stats, identity hashes, profile

   total: ~58 MB  (vs 2.25 GB source)
```

Because bands are quantization rounds, rate control at read time is
just band selection; and because the internal block layout mirrors the
source's 1024-px blocks, ranged partial reads work with plain GDAL.
Two identity hashes guard correctness: `bitstream_id` (the codebooks —
a mismatched model refuses to decode rather than decode wrongly) and
`decoder_id` (the decoder weights — keys the quality rasters, which
are decoder-specific).

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
