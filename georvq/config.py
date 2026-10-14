"""Typed, validated configuration for every georvq entry point.

One nested pydantic schema is the single source of truth. Unknown keys are
hard errors (`extra="forbid"`), cross-field validators enforce routing
invariants, and the fully-resolved config is embedded in every checkpoint
(`ck["config"]`) so models are rebuilt exactly with `georvq.from_checkpoint`.

Loading: a YAML preset (optionally chaining another via `inherit:`) deep-
merged with `key.path=value` command-line overrides, then validated.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class TierSpec(StrictModel):
    """One pyramid tier: a grid x grid latent field over the window.
    Accepts the shorthand string "grid:dim:codes:depth" in YAML."""
    grid: int    # cells per side (1 = one latent for the whole window)
    dim: int     # latent dim per cell
    codes: int   # RVQ codebook size K
    depth: int   # max RVQ levels

    @property
    def cells(self) -> int:
        return self.grid * self.grid

    def bits(self, L: int) -> float:
        import math
        return self.cells * L * math.log2(self.codes)

    def __str__(self):
        return f"{self.grid}:{self.dim}:{self.codes}:{self.depth}"


def parse_tier(v):
    if isinstance(v, str):
        g, d, k, dep = (int(t) for t in v.split(":"))
        return {"grid": g, "dim": d, "codes": k, "depth": dep}
    return v


class TokenizerConfig(StrictModel):
    dim: int = 256
    depth: int = 9
    # finest tier joins the trunk only for the last k layers (0 = every
    # layer). High-frequency content is local: a few joint layers suffice
    # and the coarse-only prefix runs at a fraction of the sequence length
    late_join: int = 0
    # One conv pyramid instead of an independent patch embed per tier and
    # a CLS parameter for the coarsest. The coarse token then arrives with
    # a convolutional read of the image rather than as a constant the
    # trunk has to fill by attention. Off by default: it renames the
    # tokenizer's parameters, so every existing checkpoint needs the
    # other path.
    stem_pyramid: bool = False

    # Parameter-free LayerNorm on the token stream entering the trunk.
    # The patch-embed output is small and low-variance, and layer 0's
    # pre-norm backward scales by 1/std of it: the tessera trap runs
    # measured a ~300x single-layer gradient amplification there, the
    # site of every recorded spike. Normalizing the stream bounds that
    # factor. New runs only -- it changes the forward, so it must stay
    # false for existing checkpoints (the default).
    embed_norm: bool = False


class DecoderStageConfig(StrictModel):
    """One member of a decoder bank: the decoder used when exactly
    `tiers` tiers carry codes.

    A rung that only sends the coarse tier does not need the compute that
    decodes 273 tokens to 96 px. Each member covers a prefix of the tier
    list and emits at `out_px`, which the codec resamples to the model's
    window so the loss and every metric stay at full resolution.
    """
    tiers: int                     # how many tiers this member decodes
    out_px: int                    # its native output edge, in window px
    dim: int = 384
    depth: int = 12
    heads: int = 8
    head_feat: int = 64
    cls_bridge: bool = True
    ff_ratio: Optional[float] = None


class DecoderConfig(StrictModel):
    # "attn": mixing decoder + per-source pixel head read from the fine
    # tokens. "tiersum": every present tier gets its own map head, the
    # maps are bilinearly upsampled and summed, absent tiers contribute
    # no tokens, and coarse-only rows skip attention entirely (see
    # georvq/model/tiersum.py). tiersum requires map_px (one output map
    # size per tier) and fill-order depth sampling.
    # "ladder": staged decoding with one exit per rate-ladder prefix and
    # deep supervision of every licensed exit (georvq/model/ladder.py);
    # map_px gives each exit head's feature-map resolution and
    # solo_layers the coarse stem depth.
    kind: Literal["attn", "tiersum", "ladder"] = "attn"
    dim: int = 384
    depth: int = 12
    heads: int = 8
    head_feat: int = 64
    cls_bridge: bool = True
    late_join: int = 0             # same contract as tokenizer.late_join
    # Decoder-only FFN width multiplier. None = share `model.ff_ratio`
    # with the tokenizer, which is the historical behaviour. Set it to
    # widen the DECODER's MLPs without touching the encoder -- required
    # when the encoder is frozen (a decoder-capacity experiment, or
    # training from cached latents), because a wider encoder FFN is a
    # different encoder and its cached latents would no longer apply.
    ff_ratio: Optional[float] = None
    head_dw: bool = False          # depthwise-separable head convs (8x fewer
                                   # MACs; smooths intra-window tile seams)
    map_px: Optional[list[int]] = None   # tiersum: per-tier map size, px
    solo_hidden: int = 512               # tiersum coarse-only MLP width
    solo_layers: int = 3
    # tiersum: sum the tiers in a feature space this many channels wide,
    # then render bands through a shared conv stack FiLM-conditioned on
    # the depth thermometers. None = sum band maps directly.
    sum_feat: Optional[int] = None


class VQConfig(StrictModel):
    commit_weight: float = 0.25
    dead_threshold: float = 0.01
    ema_decay: float = 0.99
    ema_decay_final: Optional[float] = None
    decay_ramp: tuple[int, int] = (2000, 6000)
    reset_freeze_frac: float = 0.9
    warmup_steps: int = 0          # VQ-free warmup, then residual k-means init
    kmeans_sample: int = 131072
    kmeans_iters: int = 6
    bf16_scores: bool = False
    skip_masked: bool = False      # skip VQ work for levels the sampled depth masks off
    grad_codebook: bool = False
    book_bits: Optional[int] = None  # b-bit QAT codebooks (vq_quant.py); None = fp32
    # Hard-freeze warm-started codebooks: no EMA, no dead-code resets, no
    # gradient. `mature_codebooks` only SLOWS the EMA (decay 0.9995), which
    # still drifts a corpus-fitted book over a long run; this stops it dead
    # so the shipped books stay bit-identical to the fitted ones.
    freeze_books: bool = False
    # Fraction of each batch decoded from UNQUANTIZED latents at full
    # depth (the `cos_ceiling` path). Without it a stage-2 decoder
    # specializes to quantized inputs and its continuous path decays --
    # measured on w32hd, the ceiling fell 0.9899 -> 0.9830 over 25k
    # decoder-only steps. Keeping a slice of clean batches keeps the
    # encoder/decoder pair trained on the un-quantized mapping, keeps
    # cos_ceiling a meaningful diagnostic rather than a drifting one, and
    # leaves the codec usable continuously. Costs a little capacity that
    # would otherwise go to the quantized path.
    passthrough_frac: float = 0.0
    # Global grad-norm clip; None or 0 disables. ON by default because its
    # absence is a live failure mode, not a tuning choice: with no clip a
    # single bad batch blew up a run at step 1900 (loss 0.035 -> 0.148,
    # centred ceiling cosine 0.807 -> 0.036, i.e. collapse to emitting only
    # the transmitted mean) and Adam did not recover, because the enormous
    # gradient inflates the second moment and suppresses every subsequent
    # update. It sits with the quantizer rather than the optimizer because
    # that is where the spikes come from: working dead-code resets
    # diversify the coarse z_q mid-training, and the decoder shock plus a
    # hot LR is what blows up. Small batches make it likelier, so it
    # matters most exactly where we want to run.
    clip_norm: Optional[float] = 1.0
    # How a dead code is revived. "random" reseeds from a random encoder
    # output (the historical behaviour); "split" sends it into the live
    # cell with the largest distortion contribution, at that cell's
    # worst-fitting point (LBG splitting).
    reset_mode: Literal["random", "split"] = "random"
    # Codebook seeding. "kmeans" draws K random data rows (historical);
    # "tree" builds a balanced PCA tree of median splits, which does not
    # waste seeds on near-duplicates in dense regions (heavily duplicated
    # rows can still leave a leaf empty, emitted as a zero vector). Lloyd
    # iterations follow either way and reseed empty cells.
    init_mode: Literal["kmeans", "tree"] = "kmeans"
    # Per-tier init budget, in vectors PER CODEWORD. The legacy path sizes
    # the init by batch count, which couples the tiers: at
    # kmeans_sample=32768 and batch 2048 the coarse tier sees 16 vectors
    # per centroid while the finest sees 512 and costs 12 GiB. Setting
    # this gives each tier its own target and subsamples the fine tiers
    # per batch. 0 = legacy behaviour.
    init_per_code: int = 0
    # load_partial "matures" warm-started codebooks: dead-code resets off,
    # EMA decay 0.9995. That is right when the books arrive already fitted
    # to the encoder and should barely move -- and WRONG when the point of
    # the run is to keep them live, because it silently overrides
    # ema_decay and disables reset_mode entirely.
    mature_warmstart: bool = True
    # Matryoshka truncation DURING the VQ-free warmup. The warmup branch
    # otherwise decodes at full depth every step, so the model has no
    # reason to make the coarse tiers informative -- the decoder always
    # has the fine tier available -- and "ignore tier 0" is a perfectly
    # good solution to the warmup objective. Measured on a run with an
    # 11k-step warmup: tier 0's latent fell to 1.16 effective dimensions
    # of 512 with its first codebook level at perplexity 1 (one codeword
    # for every window), while tiers 1 and 2 rose to 5.2 and 7.3. It was
    # the RICHEST tier at step 5000 and atrophied from there.
    # With this on, a random prefix of tiers is kept and the rest are
    # zeroed to true informational absence, so a coarse-only row has to be
    # reconstructed from the coarse tier alone.
    warmup_truncate: bool = True


class StatsCondConfig(StrictModel):
    """Transmit per-window stats as a side channel; code what is left.

    The mean is subtracted before the encoder and added back after the
    decoder, so loss and metrics stay on the full field. Both encoder and
    decoder are conditioned on the stats. See georvq/model/stats_cond.py.

    Which stats is a per-field choice
    (lab/findings/stats-channel-is-per-field.md): matching the std was
    measured to hurt on Tessera, whose dims are a learned embedding
    already on comparable scales, while S2 and DEM standardise by both.
    `use_std` additionally divides by the per-window std after centering
    (floored at `std_floor` in model units against near-uniform windows),
    appends it to the conditioning vector, and doubles the channel rate.
    It requires `subtract_mean`: dividing an un-centred field would scale
    its offset. The inference mean codec does not yet quantize a std
    channel, so a use_std model trains and evaluates but cannot ship a
    real bitstream until that exists.

    bits_per_band is rate accounting only -- it does not quantize anything
    during training. 6 bits/band reaches cos 0.9767 against 0.9769 for the
    exact mean, and below 4 the channel is worse than nothing."""
    subtract_mean: bool = True
    cond_encoder: bool = True
    cond_decoder: bool = True
    use_std: bool = False
    std_floor: float = 0.05
    bits_per_band: int = 6

    @model_validator(mode="after")
    def _check(self):
        """Refuse the combinations that would silently do nothing or
        silently distort (std division without centering)."""
        if not (self.subtract_mean or self.cond_encoder or self.cond_decoder):
            raise ValueError("stats_cond does nothing with every option off")
        if self.use_std and not self.subtract_mean:
            raise ValueError("stats_cond.use_std requires subtract_mean")
        if self.std_floor <= 0:
            raise ValueError("stats_cond.std_floor must be > 0")
        if self.bits_per_band <= 0:
            raise ValueError("stats_cond.bits_per_band must be > 0")
        return self


class SigmaConfig(StrictModel):
    """Stage-1 continuous mode: no codebooks; the decoder is conditioned on a
    per-tier additive-noise level instead of the matryoshka depth.

    truncate: nested tier-wise matryoshka sampling — probability of keeping
    the k coarsest tiers, ordered [all, all-but-finest, ..., coarse only]
    (must have one entry per tier and sum to ~1). Dropped tiers train at
    sigma=max (informationally absent); kept tiers draw U(0, kept_max) so
    the decoder stays quantization-noise-robust. None = legacy independent
    U(0, max) per tier."""
    max: float = 4.0
    freqs: int = 8
    truncate: Optional[list[float]] = None
    kept_max: float = 1.0


class RateConfig(StrictModel):
    """Learned conditional entropy model (attached per tier)."""
    model_dim: int = 256
    blocks: int = 4
    ctx_dim: int = 96
    weight: float = 0.005
    warmup: int = 4000
    batch: int = 64
    tau: float = 0.5
    lr: float = 5e-4


class LocationConfig(StrictModel):
    encoder: Literal["satclip", "sphharm"] = "satclip"
    degree: int = 40    # sphharm only: SH band limit (degree^2 features)
    standardize: bool = True


class CascadeConfig(StrictModel):
    """Predict-and-subtract conditioning: predictors estimate each tier's
    latents from coarser-tier reconstructions (and optionally a location
    embedding); the RVQ codes the residual."""
    kind: Literal["cascade"] = "cascade"
    coarse_prior_ckpt: Optional[Path] = None
    location: Optional[LocationConfig] = Field(default_factory=LocationConfig)
    hidden: int = 512
    layers: int = 3
    # Append each child cell's one-hot position within its parent on the
    # immediately-coarser grid, so the 16 children of one parent get
    # DISTINCT predictions. Without it the predictor's inputs are
    # identical across siblings and the prediction is constant within a
    # parent, which can only remove the child tier's per-parent mean.
    child_pos: bool = False


class SelectConfig(StrictModel):
    """Location-gated codebook subsetting: an oversized codebook of which a
    location-selected top fraction is usable per window (rate-neutral)."""
    kind: Literal["select"] = "select"
    location: LocationConfig = Field(default_factory=LocationConfig)
    proj_dim: int = 64
    weight: float = 1.0
    sel_n: int = 4096
    temp: float = 0.5
    tau: float = 0.4
    subset_frac: float = 0.25

    @field_validator("subset_frac")
    @classmethod
    def _frac(cls, v):
        if not (0 < v <= 1):
            raise ValueError("subset_frac must be in (0, 1]")
        return v


Conditioning = Annotated[Union[CascadeConfig, SelectConfig],
                         Field(discriminator="kind")]


class ModelConfig(StrictModel):
    window: int = 64
    bands: int = 64
    tiers: list[TierSpec]
    min_depths: Optional[list[int]] = None
    full_frac: float = 0.0
    # "nested": draw the coarse depth uniformly and each finer tier
    # conditionally. Weights the R-D curve toward the high-rate end, but
    # starves deep fine-tier books: the deepest fine levels train on a
    # fraction of a percent of rows. See `path_tuples` in model/codec.py
    # for the measurement and the fix.
    # "path": spread uniformly along the monotone truncation path, one
    # step per codebook, so every book gets an equal share (same level
    # rises to 0.197).
    depth_sampling: Literal["nested", "path"] = "nested"
    ff_ratio: float = 2.0
    # Emit L2-normalized * sqrt(dim) latents from encode_levels, i.e. the
    # geometry encode_stage1 (and therefore stage-1 training) uses. Set
    # this on any stage-2 run warm-started from a stage-1 checkpoint;
    # without it the decoder and the fitted codebooks are handed raw
    # latents from a different scale. Default off so existing checkpoints
    # and the bitstream contract are unaffected. (Named `norm_latents`;
    # `normalize_latents` in older checkpoints is migrated on load.)
    norm_latents: bool = False
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    # Resolution-matched decoders, one per tier prefix. None keeps the
    # single decoder every existing preset and checkpoint uses.
    decoder_bank: Optional[list[DecoderStageConfig]] = None
    vq: Optional[VQConfig] = Field(default_factory=VQConfig)
    # PCA basis for a source NORMALIZER's stats side channel, fitted by
    # scripts/fit_stats_pca_s2.py. Distinct from `stats_cond`, which
    # transmits a mean the model is also conditioned on; a normalizer's
    # stats are removed before the encoder and merely shipped. It defines
    # part of the bitstream (docs/BITSTREAM.md), hence model-level. None
    # ships them raw as f16, which is ~5x larger.
    stats_pca: Optional[Path] = None
    sigma: Optional[SigmaConfig] = None
    stats_cond: Optional[StatsCondConfig] = None
    conditioning: Optional[Conditioning] = None
    rate: Optional[RateConfig] = None
    # hierarchical local attention: per-tier radius in own-tier cells
    # (null = global tier). e.g. [null, null, 2, 4]. None = full attention.
    local_attn: Optional[list[Optional[int]]] = None

    @field_validator("tiers", mode="before")
    @classmethod
    def _tiers_shorthand(cls, v):
        if isinstance(v, (list, tuple)):
            return [parse_tier(t) for t in v]
        return v

    @model_validator(mode="after")
    def _check(self):
        """Reject model geometries the rest of the stack assumes away.

        The pyramid contract is strictly ascending grids starting at 1, each
        dividing the window: the coarsest tier's single cell IS the window,
        and `level_cumsums` and the decoder both index cells by that
        relationship. `vq` xor `sigma` separates quantized training from
        stage-1 continuous mode -- they are different objectives, not a
        flag. The rate-model warning is not pedantry: with EMA codebooks
        dR/dC is identically zero, so a rate model trains against a codebook
        it cannot move."""
        if not self.tiers:
            raise ValueError("at least one tier required")
        grids = [t.grid for t in self.tiers]
        if grids != sorted(grids) or len(set(grids)) != len(grids):
            raise ValueError(f"tier grids must be strictly ascending: {grids}")
        if grids[0] != 1:
            raise ValueError("coarsest tier must have grid=1")
        for t in self.tiers:
            if self.window % t.grid:
                raise ValueError(
                    f"window {self.window} not divisible by grid {t.grid}")
        if self.min_depths is None:
            self.min_depths = [1] + [0] * (len(self.tiers) - 1)
        if len(self.min_depths) != len(self.tiers):
            raise ValueError("min_depths must have one entry per tier")
        if self.min_depths[0] < 1:
            raise ValueError("coarsest tier needs min depth >= 1")
        for m, t in zip(self.min_depths, self.tiers):
            if not 0 <= m <= t.depth:
                raise ValueError(f"min depth {m} outside [0, {t.depth}]")
        if self.depth_sampling == "path" and any(self.min_depths[1:]):
            # path_tuples starts at the min_depths tuple, so a nonzero
            # finer minimum puts points such as (6, 1, 0) on the path,
            # which are not on the saturating fill ladder
            raise ValueError(
                "depth_sampling=path needs min_depths[1:] all zero; got "
                f"{self.min_depths}")
        if not 0.0 <= self.full_frac <= 1.0:
            raise ValueError(f"full_frac {self.full_frac} outside [0, 1]")
        if self.decoder_bank is not None:
            b = sorted(self.decoder_bank, key=lambda s: s.tiers)
            if [s.tiers for s in b] != list(range(1, len(self.tiers) + 1)):
                raise ValueError(
                    "decoder_bank needs one entry per tier prefix, 1..n: got "
                    f"{[s.tiers for s in b]} for {len(self.tiers)} tiers")
            if b[-1].out_px != self.window:
                raise ValueError(
                    "the last decoder_bank entry decodes every tier, so its "
                    f"out_px must be the window {self.window}, not "
                    f"{b[-1].out_px}")
            for s in b:
                g = self.tiers[s.tiers - 1].grid
                if s.out_px % g:
                    raise ValueError(
                        f"decoder_bank[{s.tiers}]: out_px {s.out_px} not "
                        f"divisible by that prefix's finest grid {g}")
                if self.window % s.out_px:
                    raise ValueError(
                        f"decoder_bank[{s.tiers}]: window {self.window} not "
                        f"divisible by out_px {s.out_px}; the resample back "
                        "to full resolution has to be an integer factor")
            if self.local_attn is not None:
                # the mask is built for the full token set and registered
                # on the bank, which its members never read
                raise ValueError(
                    "local_attn is not supported with decoder_bank")
            if self.stats_cond is not None and self.stats_cond.cond_decoder:
                # the patch-mean embedding is added to each member's AdaLN
                # conditioning, and one StatsEmbed serves them all, so the
                # members have to agree on width
                bad = [s.tiers for s in b if s.dim != self.decoder.dim]
                if bad:
                    raise ValueError(
                        "stats_cond.cond_decoder is on, so every decoder_bank "
                        f"member must use decoder.dim {self.decoder.dim}; "
                        f"entries {bad} differ")
            self.decoder_bank = b
        # The latents must be the TIGHTEST bottleneck in the model: a
        # trunk narrower than a tier's latent caps that latent's usable
        # rank and silently confounds any latent-width experiment (the
        # era5_t64_ladder256 null result was exactly this).
        widest = max(t.dim for t in self.tiers)
        if self.tokenizer.dim < widest:
            raise ValueError(
                f"tokenizer.dim {self.tokenizer.dim} < widest tier latent "
                f"{widest}: the encoder trunk would bottleneck the latents")
        if self.decoder.dim < widest:
            raise ValueError(
                f"decoder.dim {self.decoder.dim} < widest tier latent "
                f"{widest}: the decoder trunk would bottleneck the latents")
        if (self.vq is None) == (self.sigma is None):
            raise ValueError(
                "set exactly one of model.vq (quantized) / model.sigma "
                "(stage-1 continuous)")
        if self.sigma is not None:
            if self.conditioning is not None or self.rate is not None:
                raise ValueError(
                    "stage-1 continuous mode takes no conditioning/rate model")
            if self.stats_cond is not None:
                # forward_stage1 neither passes the stats nor adds the
                # mean back after the decoder
                raise ValueError(
                    "stage-1 continuous mode does not support stats_cond")
        if self.rate is not None and self.vq is not None \
                and not self.vq.grad_codebook:
            warnings.warn("rate model with EMA codebooks: dR/dC is dead — "
                          "set model.vq.grad_codebook=true for rate-aware "
                          "codebook training", stacklevel=2)
        return self


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class PoolConfig(StrictModel):
    size: int = 192                 # resident chips (or strip slots)
    workers: int = 2
    slot_rows: Optional[int] = None  # rows per slot: strip pool for diversity
    refresh_per_s: float = 2.0
    device: Literal["cuda", "cpu"] = "cuda"
    cycle: bool = False             # FIFO sweep of the whole store per epoch
    align_crops: bool = False       # snap crop offsets to the store's grid period

    @model_validator(mode="after")
    def _check(self):
        if self.cycle and not self.slot_rows:
            raise ValueError("pool.cycle requires pool.slot_rows")
        return self


class PrepConfig(StrictModel):
    """Model-space transform (georvq/sources/prep.py). name=None
    resolves to the source's default prep, which delegates to the
    source's existing prep/normalizer pair."""
    name: Optional[str] = None
    params: Optional[str] = None    # named fitted-state artifact


class DataConfig(StrictModel):
    source: str = "ae"              # key into georvq.sources.SOURCES
    store_dir: Path
    # How to read store_dir. "auto" detects a chip store, a live_dem store
    # or a Major-TOM Sentinel-2 root; pin it when a run must not depend on
    # what happens to be on disk. Live readers copy no pixels.
    reader: Literal["auto", "chipstore", "live_s2"] = "auto"
    val_frac: float = 0.04
    split_seed: int = 0
    limit_chips: Optional[int] = None   # data-scaling probes: train on a
    #   deterministic subsample of N chips (nested: N ⊂ 2N for the same
    #   split_seed); the val split always stays full
    pool: PoolConfig = Field(default_factory=PoolConfig)
    prep: PrepConfig = Field(default_factory=PrepConfig)

    @field_validator("val_frac")
    @classmethod
    def _frac(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"val_frac {v} outside [0, 1]")
        return v


# --------------------------------------------------------------------------
# train / eval / log
# --------------------------------------------------------------------------

class OptimConfig(StrictModel):
    lr: float = 5e-4
    wd: float = 0.01
    warmup: int = 500
    lr0_steps: int = 0              # optional initial LR hold
    # "cosine" anneals to zero over train.steps -- the run must finish to
    # get its final LR. "constant" holds `lr` after warmup, so the run can
    # be stopped at any point without leaving the schedule half-applied.
    schedule: Literal["cosine", "constant"] = "cosine"


class BatchRamp(StrictModel):
    """Run the opening steps at a fraction of the batch, then step up.

    Early training is an easy problem -- getting through many parameter
    updates matters more than the gradient quality of any one of them --
    so a small batch buys more progress per unit of wall-clock. Note the
    learning rate is NOT rescaled: a quarter batch at the same lr means
    ~2x the gradient noise, which is usually benign (and mildly helpful)
    this early, and the lr warmup covers the first steps regardless."""
    until: int = 5000
    divisor: int = 4

    @model_validator(mode="after")
    def _check(self):
        if self.until <= 0 or self.divisor <= 1:
            raise ValueError("batch_ramp needs until > 0 and divisor > 1")
        return self


class TrainConfig(StrictModel):
    steps: int = 20000
    batch_windows: int = 512
    batch_ramp: Optional[BatchRamp] = None
    # Collapse guard: abort (exit 3, after saving a forensic checkpoint)
    # when the val ceiling falls this far below its running best. A
    # stage latch (era5_t64_ladder @38k: ceiling 0.9994 -> 0.8025,
    # frozen thereafter) trips it at the first validation instead of
    # training dead weights for hours; benign spikes recover between
    # validations and never trip it. None disables.
    guard_drop: Optional[float] = 0.05
    # Carry optimizer+scheduler state in the periodic model.pt (about 3x
    # the file) so a resume continues the Adam trajectory instead of
    # rebuilding fresh moments, which storms; see save_checkpoint.
    save_optim: bool = True
    amp: bool = True
    grad_ckpt: bool = False
    tf32: bool = True
    compile: bool = False          # torch.compile the trunk layer forwards
    init_from: Optional[Path] = None
    # A warm start with vq.warmup_steps > 0 refits the codebooks by
    # k-means at warmup_steps + 1. When the init_from checkpoint supplied
    # shape-matching books, that refit discards them, and the trainer
    # refuses to start unless this is set to say the refit is intended
    # (a cascade whose books must now quantize residuals, for example).
    refit_books: bool = False
    # Continue a run from its own checkpoint (a model.pt written by this
    # trainer): weights, optimizer, scheduler, step, best score, RNG
    # state and per-level reset/decay state are restored, and the
    # schedule carries on from the saved step. Exclusive with init_from.
    resume: Optional[Path] = None
    freeze_encoder_steps: int = 0
    freeze_codec: bool = False
    # Hold the decoder's output end still while its internal representation
    # re-adapts to quantized latents: freeze the last `freeze_head_blocks`
    # decoder blocks plus norm_out/head for this many steps.
    freeze_head_steps: int = 0
    freeze_head_blocks: int = 3
    optim: OptimConfig = Field(default_factory=OptimConfig)
    seed: int = 0
    val_interval: int = 1000
    fig_interval: int = 2000
    save_interval: int = 2500
    val_tuples: int = 16


class EvalConfig(StrictModel):
    recon_tuples: Optional[list[str]] = None   # e.g. ["8-1-0", "8-8-8"]
    rd_refs: list[str] = Field(default_factory=list)


class ReportConfig(StrictModel):
    """Per-run narrowing of what the source declares it reports.

    Every field defaults to None, meaning "whatever the source's
    `ReportSpec` says" -- the source is the authority on what its field
    admits, and a run may only take things AWAY. Naming something the
    source does not declare raises at config resolution
    (`georvq.sources.report.resolve_report`), not at the first validation.
    """
    #: Metrics to log, most important FIRST: the head of the list becomes
    #: the headline metric that names the val scalars, labels the R-D axis
    #: and selects model_best.pt.
    metrics: Optional[list[str]] = None
    #: Zero-rate reference lines on the R-D plot (e.g. ["blockmean"]).
    baselines: Optional[list[str]] = None
    #: Figures to draw: rd, rd_log, codebook, recon, reset, cosdist.
    #: [] logs scalars only -- the cheap setting for a long sweep.
    figures: Optional[list[str]] = None
    #: The source's preview key, or "none" to skip reconstruction panels.
    preview: Optional[str] = None


class LogConfig(StrictModel):
    run_tag: str = "dev"
    wandb: bool = True
    wandb_project: str = "georvq"
    # Deferred reporting: figure decodes are harvested on the GPU, then
    # percentiles/matplotlib/wandb uploads render on a background worker
    # while training resumes -- recovers most of the GPU idle time at
    # every fig-interval validation. false = the old synchronous path.
    # Figures log against their own `figs/step` metric in this mode.
    async_report: bool = True
    out_dir: Path = Path("checkpoints")
    # added to the step of every wandb log call, so a run continued from
    # another run's checkpoint plots on the same x-axis; the schedule and
    # stdout keep counting from 1
    step_offset: int = 0
    report: ReportConfig = Field(default_factory=ReportConfig)


#: Fields removed from the schema, as "section.field" paths. A checkpoint
#: embeds the config it was trained with, so retiring a knob would
#: otherwise make every checkpoint that carried it unloadable under
#: `extra="forbid"` -- including released weights. `drop_retired` strips
#: them when rebuilding a config FROM A CHECKPOINT; YAML presets still
#: hard-fail on unknown keys, because there a stray key is a typo, not
#: history.
RETIRED_FIELDS = (
    "model.vq.gain_shape", "model.vq.gain_bits",           # negative result
    "model.vq.refit_interval", "model.vq.refit_offset",    # superseded by
    "model.vq.refit_per_code", "model.vq.refit_lloyd_iters",  # fit-once
    "model.vq.refit_dead_frac", "model.vq.refit_recycle_frac",
    "model.vq.drift_interval", "model.vq.drift_windows",   # served its purpose
    "model.tokenizer.suffix_dim", "model.decoder.suffix_dim",
    "train.trap",                                          # debugging tool
    "data.sampler",                                        # one geometry
)


#: Fields that were RENAMED, as old -> new dotted paths. Unlike a retired
#: field a renamed one still means something, so its value must be moved
#: rather than dropped -- silently discarding it would change how an old
#: checkpoint decodes. Both of these were named twice in the project
#: because two branches solved the same problem independently.
RENAMED_FIELDS = {
    "train.optim.grad_clip": "model.vq.clip_norm",
    "model.normalize_latents": "model.norm_latents",
}


def _pop_path(raw, path):
    cur, keys = raw, path.split(".")
    for k in keys[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return None, False
        cur = cur[k]
    if isinstance(cur, dict) and keys[-1] in cur:
        return cur.pop(keys[-1]), True
    return None, False


def _set_path(raw, path, value):
    cur, keys = raw, path.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            return
    cur.setdefault(keys[-1], value)


def apply_renames(raw: dict) -> dict:
    """Move renamed fields to their current paths, keeping their values.

    `setdefault` on the destination, so a config that already carries the
    new name wins over a stale copy of the old one.
    """
    import copy
    if not any(_pop_path(copy.deepcopy(raw), old)[1] for old in RENAMED_FIELDS):
        return raw
    raw = copy.deepcopy(raw)
    for old, new in RENAMED_FIELDS.items():
        val, found = _pop_path(raw, old)
        if found:
            _set_path(raw, new, val)
    return raw


def migrate(raw: dict) -> dict:
    """Bring a config dict from any past schema to the current one."""
    return drop_retired(apply_renames(raw))


def drop_retired(raw: dict) -> dict:
    """Remove retired fields from a raw config dict, in place-safe fashion.

    Returns the same mapping when nothing was dropped, so the common path
    allocates nothing.
    """
    import copy
    hit = []
    for path in RETIRED_FIELDS:
        node, *rest = path.split(".")
        cur = raw
        keys = [node] + rest
        for k in keys[:-1]:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, dict) and keys[-1] in cur:
            hit.append(path)
    if not hit:
        return raw
    raw = copy.deepcopy(raw)
    for path in hit:
        keys = path.split(".")
        cur = raw
        for k in keys[:-1]:
            cur = cur[k]
        cur.pop(keys[-1], None)
    return raw


class Config(StrictModel):
    """The whole of a run, validated.

    This object is what makes a run reproducible: it is embedded in every
    checkpoint, hashed into `config_hash`, and is the only place a
    setting that affects a run may live. `extra="forbid"` throughout, so
    a typo in a YAML key is an error rather than a silently ignored
    setting.
    """
    schema_version: int = SCHEMA_VERSION
    data: DataConfig
    model: ModelConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    @model_validator(mode="after")
    def _check(self):
        """Cross-section checks: the ones that need more than one section
        of the config to be visible at once, and so cannot live on the
        individual models above."""
        v = self.model.vq
        if v is not None:
            # these two partition one batch -- full-depth rows, path-sampled
            # rows, and un-quantized pass-through rows -- but they live in
            # different sections, so nothing else notices when they overrun
            # it. Over 1.0 the path fraction goes negative and the sampler
            # silently returns full depth for everything.
            if self.model.full_frac + v.passthrough_frac > 1.0:
                raise ValueError(
                    f"model.full_frac ({self.model.full_frac}) + "
                    f"model.vq.passthrough_frac ({v.passthrough_frac}) > 1: "
                    "they are shares of the same batch and leave no rows for "
                    "the sampled depth tuples")
        if self.train.freeze_codec:
            if self.train.init_from is None:
                raise ValueError("train.freeze_codec requires train.init_from")
            c = self.model.conditioning
            if c is None or c.kind != "select":
                raise ValueError("train.freeze_codec is only meaningful with "
                                 "conditioning=select")
        return self


# --------------------------------------------------------------------------
# loading / overrides / diff
# --------------------------------------------------------------------------

def deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; `over` wins. Lists are replaced, not merged."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_chain(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    if parent:
        base = _load_yaml_chain((path.parent / parent).resolve())
        raw = deep_merge(base, raw)
    return raw


def _set_dotted(tree, key: str, value):
    """Set tree['a']['b']... = value for key 'a.b....'; integer path
    components index lists."""
    parts = key.split(".")
    node = tree
    for i, p in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        if isinstance(node, list):
            node = node[int(p)]
        else:
            if p not in node or node[p] is None:
                node[p] = [] if nxt.isdigit() else {}
            node = node[p]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def apply_overrides(raw: dict, overrides) -> dict:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key.path=value, got {ov!r}")
        key, _, val = ov.partition("=")
        _set_dotted(raw, key.strip(), yaml.safe_load(val))
    return raw


def load_config(path, overrides=()) -> Config:
    """YAML preset (+ `inherit:` chain) + dot-path overrides -> validated Config."""
    raw = _load_yaml_chain(Path(path).resolve())
    raw = apply_overrides(raw, overrides)
    return Config.model_validate(raw)


def config_hash(cfg: Config) -> str:
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _flatten(d, prefix=""):
    """Nested config -> {"a.b.0.c": value}, so two configs can be diffed
    as flat key sets rather than by walking two trees in parallel. List
    indices become path components, which means a reordered list reads
    as several changed keys.
    """
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def config_diff(a, b) -> list[tuple[str, object, object]]:
    """Key-level diff of two configs (Config, dict, or checkpoint path).
    Returns [(dotted_key, a_value, b_value)] for keys that differ."""
    def as_dict(x):
        """Accept a Config, a plain dict, or a checkpoint path, so runs can
        be diffed against each other without first being loaded.
        """
        if isinstance(x, Config):
            return x.model_dump(mode="json")
        if isinstance(x, dict):
            return x
        from georvq.repro import load_checkpoint
        ck = load_checkpoint(x)
        return ck["config"]
    fa, fb = _flatten(as_dict(a)), _flatten(as_dict(b))
    keys = sorted(set(fa) | set(fb))
    return [(k, fa.get(k, "<absent>"), fb.get(k, "<absent>"))
            for k in keys if fa.get(k, "<absent>") != fb.get(k, "<absent>")]
