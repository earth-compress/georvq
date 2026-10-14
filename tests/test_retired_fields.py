"""Retiring a config knob must not brick existing checkpoints.

A checkpoint embeds the config it was trained with. Under
`extra="forbid"` that means deleting a field from the schema makes every
checkpoint that carried it unloadable -- including released weights,
which is how a cleanup turns into a recall.
"""
import pytest

from georvq.config import RETIRED_FIELDS, Config, drop_retired, load_config


def _raw():
    cfg = load_config("configs/smoke.yaml", [])
    return cfg.model_dump(mode="json")


def test_a_checkpoint_carrying_retired_fields_still_loads():
    raw = _raw()
    raw["model"]["vq"]["gain_shape"] = True      # as an old checkpoint has it
    raw["model"]["vq"]["drift_interval"] = 500
    raw["model"]["tokenizer"]["suffix_dim"] = 64
    with pytest.raises(Exception):               # forbidden, as it should be
        Config.model_validate(raw)
    Config.model_validate(drop_retired(raw))     # ...but recoverable


def test_slim_tier2_fields_still_validate_from_a_checkpoint():
    """Configs saved before the trap and the sampler seam were removed
    carry them, sometimes with non-default values."""
    raw = _raw()
    raw["train"]["trap"] = {"gnorm_abs": 0.5, "gnorm_ratio": 10.0,
                            "min_step": 200, "n_dumps": 3, "cooldown": 250,
                            "hold": True, "hold_port": 4444,
                            "hold_timeout": 14400.0, "snapshot_every": 500,
                            "stats_every": 10, "batch_history_bytes": 0,
                            "flight_maxlen": 20000}
    raw["data"]["sampler"] = {"name": "window", "params": {"stride": 2}}
    with pytest.raises(Exception):
        Config.model_validate(raw)
    cfg = Config.model_validate(drop_retired(raw))
    assert cfg.model_dump(mode="json") == Config.model_validate(
        _raw()).model_dump(mode="json")
    raw["train"]["trap"] = None
    raw["data"]["sampler"] = {"name": None, "params": {}}
    Config.model_validate(drop_retired(raw))


def test_yaml_presets_still_hard_fail_on_unknown_keys():
    """Retirement is for history, not for typos: a stray key in a preset
    is a mistake and must still be an error."""
    with pytest.raises(Exception):
        load_config("configs/smoke.yaml", ["model.vq.gian_shape=true"])


def test_drop_retired_is_a_noop_when_nothing_is_retired():
    raw = _raw()
    assert drop_retired(raw) is raw              # no copy on the common path


def test_every_retired_path_is_actually_gone_from_the_schema():
    """A path left in the list after the field came back would silently
    strip a live setting."""
    raw = _raw()
    for path in RETIRED_FIELDS:
        keys = path.split(".")
        cur = raw
        for k in keys[:-1]:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        assert keys[-1] not in (cur or {}), f"{path} is still a live field"


def test_renamed_fields_move_their_value_rather_than_dropping_it():
    """A rename is not a retirement. Dropping `grad_clip` from an old
    checkpoint would silently turn clipping off at a different value than
    the run used, so the value has to travel to the new path."""
    from georvq.config import RENAMED_FIELDS, migrate
    raw = _raw()
    raw["train"]["optim"]["grad_clip"] = 0.5
    raw["model"]["normalize_latents"] = True
    raw["model"]["vq"].pop("clip_norm", None)
    raw["model"].pop("norm_latents", None)
    out = migrate(raw)
    cfg = Config.model_validate(out)
    assert cfg.model.vq.clip_norm == 0.5
    assert cfg.model.norm_latents is True
    assert "grad_clip" not in out["train"]["optim"]
    assert "normalize_latents" not in out["model"]
    for old, new in RENAMED_FIELDS.items():
        assert old != new


def test_a_current_config_wins_over_a_stale_old_name():
    """Both present means the old one is a leftover; the current path is
    what the run was actually validated with."""
    from georvq.config import migrate
    raw = _raw()
    raw["train"]["optim"]["grad_clip"] = 0.5      # stale
    raw["model"]["vq"]["clip_norm"] = 2.0         # current
    assert Config.model_validate(migrate(raw)).model.vq.clip_norm == 2.0


def test_slim_tier1_fields_still_validate_from_a_checkpoint():
    """Configs dumped before the tier-1 removals carry these fields at
    their defaults (every checkpoint on record does)."""
    raw = _raw()
    raw["model"]["local_attn"] = None
    raw["model"]["tokenizer"]["late_join"] = 0
    raw["model"]["decoder"]["late_join"] = 0
    raw["model"]["decoder"]["head_dw"] = False
    raw["train"]["compile"] = False
    raw["train"]["freeze_head_steps"] = 0
    raw["train"]["freeze_head_blocks"] = 3
    raw["data"]["limit_chips"] = None
    with pytest.raises(Exception):
        Config.model_validate(raw)
    cfg = Config.model_validate(drop_retired(raw))
    assert cfg.model_dump(mode="json") == Config.model_validate(
        _raw()).model_dump(mode="json")


@pytest.mark.parametrize("path,value", [
    ("model.local_attn", [None, 2]),
    ("model.tokenizer.late_join", 2),
    ("model.decoder.late_join", 1),
    ("model.decoder.head_dw", True),
    ("train.compile", True),
    ("train.freeze_head_steps", 500),
    ("train.freeze_head_blocks", 2),
    ("data.limit_chips", 1000),
])
def test_each_tier1_field_is_retired(path, value):
    assert path in RETIRED_FIELDS
    raw = _raw()
    *head, last = path.split(".")
    cur = raw
    for k in head:
        cur = cur[k]
    cur[last] = value
    Config.model_validate(drop_retired(raw))
