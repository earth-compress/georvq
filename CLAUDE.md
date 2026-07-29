# georvq

Pyramidal residual-VQ compression for geospatial datasets.

**Read `docs/OVERVIEW.md` before making changes** — it has the module map,
the invariants (VQ numerics are parity-tested against trained checkpoints;
state-dict names and the bitstream contract are load-bearing), preset
provenance, and practical gotchas.

- Verify changes with `pytest -m "not slow and not gpu"` (~20 s). Prefer
  `configs/smoke.yaml` or a few-hundred-step override run over long
  training runs.
- Never edit `georvq/model/vq.py` numerics: trained checkpoints must
  decode bit-faithfully forever (the golden-checkpoint gate in
  `tests/test_compat.py` must stay exact).
- Configs are pydantic-validated (`extra="forbid"`); anything affecting a
  run belongs in the `Config` schema.
- `georvq/ae/` is the AlphaEarth *release layer* (source-specific
  policy, `docs/AE_RELEASE.md`); `georvq/inference/` stays
  source-agnostic. Keep that boundary.
