# The AlphaEarth release codec

The frozen, shippable codec for the AlphaEarth Foundations embedding
tiles. Everything documented here is **release policy for this source**
(`georvq/ae/`), built on the source-agnostic machinery in
`georvq/inference/`. Other sources get their own release layer and their
own version of this document.

## Who is this for

| you are | you touch |
|---|---|
| **AE data user** — decode archive tiles | `georvq-ae-tile decode/info`, the COG assets, this doc's *Format* + *Controls* sections; never the encoder or training code |
| **archive operator** — bulk-encode source tiles | `georvq-ae-tile encode` (GPU), the stats/quality outputs, *Economics* below |
| **codec researcher** — train/improve models | `georvq/{model,train,...}`, `docs/OVERVIEW.md`; this package is your worked example of a release layer |

## The release

- Model: `ae_pyr64_divfull` (window 64, tiers 1:256:2048:8 / 4:192:8192:8
  / 16:64:16384:8), **bitstream_id `2f7c352718b87fe6`**. Blobs and models
  cross-check ids; mismatches are refused (same-architecture codebooks
  decode each other's indices into plausible garbage).
- Backends: PyTorch (CPU/CUDA) or ONNX (torch-free; `[onnx]` extra).
  Encoders are backend-consistent, not bit-identical (fp32 vs bf16
  assignment flips ~13% of near-tie deep indices at equal quality);
  decode is deterministic everywhere.
- Verified (held-out val + real tiles): full-ladder cos ~0.9976-0.998,
  exact nodata + GeoTIFF-profile round-trip; a 2,250 MiB source tile
  becomes ~58 MiB of assets.

## Format: one tile = a COG asset directory

```
<tile>/
  coarse.tif   128x128,  8 uint16 bands   RVQ levels, 640 m cells
  mid.tif      512x512,  8 uint16 bands   160 m cells
  fine.tif    2048x2048, 8 uint16 bands    40 m cells
  star.tif     128x128, 66 uint8 bands    2-bit coarse-residual sidecar
  mask.tif    8192x8192 validity          exact -128 nodata restore
  quality/<decoder_id>.tif                measured mean dcos per window
                                          per rate point (decoder-keyed)
  metadata.json                           source profile, quality stats,
                                          specs, bitstream_id
```

Bands are quantizer levels → rate control is band selection. Blocks
mirror the source's 1024 px quantum at every level → windowed access is
stock ranged-COG reads (rasterio/GDAL `/vsicurl`; no custom client).
Assets are north-up in the source CRS with `BITSTREAM_ID` in the tags.
**Values are code indices: never interpolate, never build averaged
overviews** (the quality raster is the one asset that averages
meaningfully).

Measured access costs (local ranged-HTTP): whole-tile coarse ~180 KiB /
4 requests; a 1024 px viewport at browse quality ~150-220 KiB; a point
query ~1 MiB (one source-block quantum — the source tiff needs ~12 MiB
for the same read).

## Controls: rate or quality

- **Rate control**: `--tuple L0,L1,L2`. The efficient ladder (measured
  Pareto front) is: max coarse, fill mid, then fine levels one at a
  time — `8-1-0` (0.07 bpp) ... `8-8-0` (0.43) ... `8-8-8` (7.43).
  Level costs: coarse 11 / mid 208 / fine 3,584 bits per window.
- **Quality control**: `--quality STAT<=X` with STAT in
  mean|p50|p90|p99|max of pixel dcos — resolves to the cheapest ladder
  point whose *encode-time measured* stats satisfy the bound
  (`metadata.json`), and reads only that prefix.
- The 8* sidecar (2-bit coarse residual, 66 B/window) applies
  automatically whenever the coarse tier decodes at full depth. It
  recovers the coarse quantizer gap and most of the OOD error tail
  (worst-0.1% pixel median 0.035 -> 0.009) for +1.5% rate.

**Getting the model**: weights are deliberately not in git.

- *On the group server*: `/data/georvq-releases/ae-beta-v1/`
  (`bundle/` for torch, `onnx/` for torch-free). Notebooks and examples
  default to it; override with `GEORVQ_AE_BUNDLE`.
- *Elsewhere*: download `ae-beta-v1.tar.gz` (524 MB) — link in the repo
  README / from Ali — then:

  ```bash
  sha256sum -c <<< "d535fe3c692327acbafcfd8816be9d776d4596fe02ea18e6963759fbbbf969b5  ae-beta-v1.tar.gz"
  tar xzf ae-beta-v1.tar.gz && export GEORVQ_AE_BUNDLE=$PWD/ae-beta-v1/bundle
  ```

  Verify the checksum before use — `bundle/model.pt` is a torch pickle
  (arbitrary code on load); the `onnx/` directory is the no-torch,
  no-pickle alternative and is all that decoding requires.
- Proper remote distribution (HF Hub registry keyed by bitstream_id) is
  the planned replacement for the download link.

```bash
georvq-ae-tile encode --model <bundle> --device cuda src.tiff out_dir/
georvq-ae-tile info   out_dir/
georvq-ae-tile decode --model <bundle> out_dir/ recon.tiff --quality "p90<=0.02"
georvq-ae-export-onnx --ckpt <bundle> --out onnx_dir/     # torch-free decode
```

## Economics (server-side encoding)

76 s/tile with full quality stats on one A100 (35 s without), bf16
decoder forwards + fp32 encoder (indices bit-stable). One annual AEF
layer (~34k tiles) ~ 30 A100-days -> ~1.9 TB of assets; the source
bucket is 302,466 tiles over 9 years.

## Deliberate non-features

- **No entropy coding**: order-0 rANS measured at ~0.6% on real tiles
  (fine tier ~95% entropy-saturated; deflate recovers the rest) while
  dominating decode time. The coarse tier's ~30% entropy gap is left on
  the table (a regional coarse-mosaic product is the fix if it matters).
- **Single deep decoder**: lite variants trained and parked below the
  quality bar (see Deferred with evidence in docs/FINDINGS.md, which also
  covers the deblocker and the
  error-envelope predictor.
