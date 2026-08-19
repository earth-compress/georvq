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
- `georvq/ae/` and `georvq/tessera/` are *release layers*
  (source-specific policy, `docs/AE_RELEASE.md` and
  `docs/TESSERA_RELEASE.md`); `georvq/inference/` stays source-agnostic.
  Keep that boundary.
- Shipped payloads are frozen. The AlphaEarth `star.tif` bytes are
  public, so `tests/test_refine.py` pins them against an inlined copy of
  the original packer; generalising code behind a shipped format is fine,
  changing its bytes is not.
- Measured figures live in `docs/FINDINGS.md` plus at most one mechanism
  home; elsewhere state the claim and cite. `tests/test_figures_consistent.py`
  pins this — a new quotation fails it, which is the prompt to ask whether
  the reader needs the number or just the claim.
- Before designing for a new source, read `docs/SOURCE_ONBOARDING.md`:
  window, tier grids, latent widths and codebook sizes are derived from
  measurements, not copied from a preset.
