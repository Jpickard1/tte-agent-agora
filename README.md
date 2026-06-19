# tte-agent-agora

Trial-general, multi-dataset **Target Trial Emulation (TTE)**. Read a trial from
ClinicalTrials.gov, figure out how to extract the right data from each of three
EHR databases (**eICU**, **MIMIC-IV**, **MGB**), normalize all three into one
common EHR format, run the emulation, and report the emulated effect vs the
trial's observed result — across as many trials as possible.

This repo is built by a team of autonomous agents; humans monitor progress via
GitHub issues + PRs. The Python package is `tteEngine`.

📐 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** is the canonical reference:
the end-to-end pipeline, the 5-column common-format contract, the integration
seams, and the import-light layering. Run `python -m tteEngine.reproduce` to
regenerate the full artifact deterministically.

## Pipeline

```
ctgov trial
  -> TargetTrialSpec       (PICO-T + estimand)          #2
  -> ExtractionPlan        (what to pull, per DB)        #3      SEAM (a)
  -> Event stream (5-col)  (canonical common format)     #6/#7/#8 SEAM (b)
  -> CohortResult          (arms, time-zero, features)   #9      SEAM (c)
  -> TTEResult             (effect + CI + diagnostics)   #10
  -> emulated-vs-observed report                         #11
all orchestrated per-trial, resumable, batched over trials x DBs   #12 / #13
```

## Common format

The canonical store is the EHR-DE **5-column long event-stream** — *not* OMOP:

| column | type | |
|--------|------|--|
| `TRAJECTORY_ID` | int64 | one hospital admission (≥1 ICU stay); `0` = dictionary rows |
| `TIMESTAMP` | datetime64[ns, UTC] | sub-day, orderable (immortal-time safe) |
| `EVENT_TYPE` | str | controlled set: `measu, lab, medic, diagn, proce, locat, demog, outco, micro, emar, drg, order` |
| `EVENT_NAME` | str | raw source field (pre-normalization) |
| `EVENT_VALUE` | str \| json-str | scalar value, or JSON metadata |

Two of three DBs already emit this format with working code (MGB Snowflake
natively; MIMIC via `EHR-DE/MIMIC-IV/extraction_v1.py`). Wide cohort/feature
tables are deterministic **views** over the stream. Cross-DB semantics come from
a decoupled vocab layer (`EVENT_NAME`/unit → ICD/RxNorm/LOINC, #5).

## Status

Scaffold + typed seam contracts (`src/tteEngine/contracts/`) are in. Each
feature is a GitHub issue with an owner; see [CONTRIBUTING.md](CONTRIBUTING.md)
for lanes and the build workflow.

## Reused upstream

- **trialsim** (`th789/trialsim`) — ctgov fetch, eICU/MIMIC dictionaries + prebuilt cohorts, the Cox/PSM/IPTW/AIPW TTE engine, and `compare.py` (emulated-vs-observed).
- **emulaTTE** (`viktoriaschuster/emulaTTE`) — the 5-step orchestration core, typed schemas, and the check-&-correct pause loop.
- **EHR-DE** (`Jpickard1/EHR-DE`) — the 5-col format + the MIMIC adapter + MGB Snowflake extraction + clinical constants (sepsis codes, SOFA).
