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
