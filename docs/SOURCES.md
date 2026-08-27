# Sources

Where each data source currently stands. The repository does not rank
sources; this table tracks maturity, not priority.

| source | status | adapter | release layer and contract |
|---|---|---|---|
| AlphaEarth | beta release (`ae-beta-v1`) | `georvq/sources/ae.py` | `georvq/ae/`, `docs/AE_RELEASE.md` |
| Tessera | beta release (`tessera-beta-v1`) | `georvq/sources/tessera.py` | `georvq/tessera/`, `docs/TESSERA_RELEASE.md` |
| Sentinel-2 L1C | beta, built on an older version of the codebase | `georvq/sources/s2.py` | none |
| COPDEM | in development | `georvq/sources/dem.py` | none |
| Sentinel-1 RTC | not started; characterised in the five-source measurement table (`docs/SOURCE_ONBOARDING.md`) | none | none |

Notes:

- Sentinel-2: the model layer supports multi-group batches (stems,
  heads, quantization all tested), but the streaming pool and the
  standard training loop are single-plane only. Later S2 work sits
  unmerged on `origin/worktree-s2-source`.
- New sources join by the procedure in `docs/ADDING_A_SOURCE.md`, with
  the measurements described in `docs/SOURCE_ONBOARDING.md` taken
  first.
