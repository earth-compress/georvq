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
    # with suffix_dim set, the finest tier becomes a NARROW stream: base
    # tiers run the full self-attention stack alone (cheap — few tokens),
    # and the finest tier runs `late_join` cross+local layers at this
    # width over [projected base | fine] keys. None = shared-width trunk.
    suffix_dim: Optional[int] = None


class DecoderConfig(StrictModel):
    dim: int = 384
    depth: int = 12
    heads: int = 8
    head_feat: int = 64
    cls_bridge: bool = True
    late_join: int = 0             # same contract as tokenizer.late_join
    suffix_dim: Optional[int] = None
    head_dw: bool = False          # depthwise-separable head convs (8x fewer
                                   # MACs; smooths intra-window tile seams)


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
    gain_shape: bool = False       # unit-sphere shape book + scalar gain
    gain_bits: int = 5             # log-domain gain levels (gain_shape only)


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
    encoder: Literal["satclip"] = "satclip"
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
        assert 0 < v <= 1, "subset_frac must be in (0, 1]"
        return v


Conditioning = Annotated[Union[CascadeConfig, SelectConfig],
                         Field(discriminator="kind")]


class ModelConfig(StrictModel):
    window: int = 64
    bands: int = 64
    tiers: list[TierSpec]
    min_depths: Optional[list[int]] = None
    full_frac: float = 0.0
    ff_ratio: float = 2.0
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    vq: Optional[VQConfig] = Field(default_factory=VQConfig)
    # Emit L2-normalized * sqrt(dim) latents from encode_levels, i.e. the
    # geometry encode_stage1 (and therefore stage-1 training) uses. Set
    # this on any stage-2 run warm-started from a stage-1 checkpoint;
    # without it the decoder and the fitted codebooks are handed raw
    # latents from a different scale. Default off so existing checkpoints
    # and the bitstream contract are unaffected.
    normalize_latents: bool = False
    sigma: Optional[SigmaConfig] = None
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
        assert self.tiers, "at least one tier required"
        grids = [t.grid for t in self.tiers]
        assert grids == sorted(grids) and len(set(grids)) == len(grids), \
            f"tier grids must be strictly ascending: {grids}"
        assert grids[0] == 1, "coarsest tier must have grid=1"
        for t in self.tiers:
            assert self.window % t.grid == 0, \
                f"window {self.window} not divisible by grid {t.grid}"
        if self.min_depths is None:
            self.min_depths = [1] + [0] * (len(self.tiers) - 1)
        assert len(self.min_depths) == len(self.tiers), \
            "min_depths must have one entry per tier"
        assert self.min_depths[0] >= 1, "coarsest tier needs min depth >= 1"
        for m, t in zip(self.min_depths, self.tiers):
            assert 0 <= m <= t.depth, f"min depth {m} > tier depth {t.depth}"
        assert 0.0 <= self.full_frac <= 1.0
        assert (self.vq is None) != (self.sigma is None), \
            "set exactly one of model.vq (quantized) / model.sigma (stage-1 continuous)"
        if self.sigma is not None:
            assert self.conditioning is None and self.rate is None, \
                "stage-1 continuous mode takes no conditioning/rate model"
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
        if self.cycle:
            assert self.slot_rows, "pool.cycle requires pool.slot_rows"
        return self


class DataConfig(StrictModel):
    source: str = "ae"              # key into georvq.sources.SOURCES
    store_dir: Path
    val_frac: float = 0.04
    split_seed: int = 0
    limit_chips: Optional[int] = None   # data-scaling probes: train on a
    #   deterministic subsample of N chips (nested: N ⊂ 2N for the same
    #   split_seed); the val split always stays full
    pool: PoolConfig = Field(default_factory=PoolConfig)

    @field_validator("val_frac")
    @classmethod
    def _frac(cls, v):
        assert 0.0 <= v <= 1.0
        return v


# --------------------------------------------------------------------------
# train / eval / log
# --------------------------------------------------------------------------

class OptimConfig(StrictModel):
    lr: float = 5e-4
    wd: float = 0.01
    warmup: int = 500
    lr0_steps: int = 0              # optional initial LR hold


class TrainConfig(StrictModel):
    steps: int = 20000
    batch_windows: int = 512
    amp: bool = True
    grad_ckpt: bool = False
    tf32: bool = True
    compile: bool = False          # torch.compile the trunk layer forwards
    init_from: Optional[Path] = None
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


class LogConfig(StrictModel):
    run_tag: str = "dev"
    wandb: bool = True
    wandb_project: str = "georvq"
    out_dir: Path = Path("checkpoints")


class Config(StrictModel):
    schema_version: int = SCHEMA_VERSION
    data: DataConfig
    model: ModelConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    @model_validator(mode="after")
    def _check(self):
        if self.train.freeze_codec:
            assert self.train.init_from is not None, \
                "train.freeze_codec requires train.init_from"
            assert self.model.conditioning is not None \
                and self.model.conditioning.kind == "select", \
                "train.freeze_codec is only meaningful with conditioning=select"
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
        if isinstance(x, Config):
            return x.model_dump(mode="json")
        if isinstance(x, dict):
            return x
        import torch
        ck = torch.load(x, map_location="cpu", weights_only=False)
        return ck["config"]
    fa, fb = _flatten(as_dict(a)), _flatten(as_dict(b))
    keys = sorted(set(fa) | set(fb))
    return [(k, fa.get(k, "<absent>"), fb.get(k, "<absent>"))
            for k in keys if fa.get(k, "<absent>") != fb.get(k, "<absent>")]
