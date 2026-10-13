# Sources

Draft for review: the AlphaEarth row's v2 mention is new and not yet approved.

Where each data source currently stands. The repository does not rank
sources; this table tracks maturity, not priority.

| source | status | adapter | preset | release layer and contract |
|---|---|---|---|---|
| AlphaEarth | beta release (`ae-beta-v1`); v2 release in preparation | `georvq/sources/ae.py` | `configs/ae/pyr64.yaml` | `georvq/ae/`, `docs/AE_RELEASE.md`; v2 `georvq/ae/v2.py`, `docs/AE_V2_RELEASE.md` |
| Tessera | beta release (`tessera-beta-v1`) | `georvq/sources/tessera.py` | `configs/tessera/tessera_vq8_k256_statscond.yaml` | `georvq/tessera/`, `docs/TESSERA_RELEASE.md` |
| Sentinel-2 L1C | beta | `georvq/sources/s2.py` | `configs/s2/s2_pyr96.yaml` | none |
| COPDEM | in development | `georvq/sources/dem.py` | `configs/dem/dem128.yaml` | none |
| ERA5 | exploratory, separate track | `georvq/sources/era5.py` | `configs/era5/era5_t64.yaml` | none |
| Sentinel-1 RTC | not started; characterised in the five-source measurement table (`docs/SOURCE_ONBOARDING.md`) | none | none | none |

Notes:

- Sentinel-2 is multi-resolution (10/20/60 m groups). The pool, the
  training loop and the figures handle multi-group batches, and Major-TOM
  scenes can be read in place by `LiveS2Store`.
- ERA5 is on its own track because its accuracy at full rate is not yet
  at a level a domain scientist would rely on.
- New sources join by the procedure in `docs/ADDING_A_SOURCE.md`, with
  the measurements described in `docs/SOURCE_ONBOARDING.md` taken
  first.

## Features

Stable means a shipped release depends on the feature, so its behaviour
and checkpoint format are protected. Exploratory means it trains and is
tested, but no release uses it and it may change or be removed.

| feature | status | used by |
|---|---|---|
| pyramid tiers, matryoshka depth conditioning | stable | every source |
| strict fill-order depth sampling (`depth_sampling: path`) | stable | Tessera release |
| transmitted mean (`stats_cond`) | stable | Tessera release |
| int8 codebooks (`vq.book_bits`) | stable | Tessera release |
| additive refinements (`georvq/inference/refine.py`) | stable | AE (`star`), Tessera (opt-in) |
| deblock decoder | stable | AE release |
| ONNX export and torch-free decode | stable for AE; models with a transmitted mean are not supported yet | AE release |
| multi-resolution groups, decoder bank, grown training | exploratory | S2; grown training also a Tessera preset |
| ladder and tiersum decoders | exploratory | ERA5 and Tessera presets |
| `tokenizer.stem_pyramid` | exploratory | S2 |
| `tokenizer.embed_norm` | exploratory | no preset yet |
| `data.prep` / `data.sampler` | being adopted source by source | ERA5 |
| cascade and location-gated codebooks | exploratory | AE presets |
| learned rate model, rANS archives | exploratory | `configs/ae/hybrid_rate.yaml` |
| stage-1 continuous pretraining | exploratory | AE and DEM presets |
| multi-GPU data parallelism (`georvq/train/dist.py`) | frozen codebooks only | none |
