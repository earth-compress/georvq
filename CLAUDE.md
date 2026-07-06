# georvq

Pyramidal residual-VQ compression for geospatial datasets.

**Read `docs/OVERVIEW.md` before making changes** — it has the module map,
the invariants (VQ numerics are parity-tested against trained checkpoints;
state-dict names and the bitstream contract are load-bearing), preset
provenance, and practical gotchas.

- Verify changes with `pytest -m "not slow and not gpu"` (~6 s). Prefer
  `configs/smoke.yaml` or a few-hundred-step override run over long
  training runs.
- Never edit `georvq/model/vq.py` numerics without keeping the
  `-m ringspace` parity tests exact.
- Configs are pydantic-validated (`extra="forbid"`); anything affecting a
  run belongs in the `Config` schema.
