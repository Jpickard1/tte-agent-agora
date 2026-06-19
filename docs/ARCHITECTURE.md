# Architecture & Common-Format Contract

> **Canonical reference for the build (#14).** The end-to-end pipeline, the
> 5-column common-format decision (NOT OMOP), the integration seams, and the
> import-light layering. Pinned — read this before changing a seam.
>
> Executable entry point: [`tteEngine.reproduce`](../src/tteEngine/reproduce.py)
> regenerates the whole artifact (corpus → JSONL → meta/calibration/drivers →
> `RESULTS_NARRATIVE.md`) deterministically from a vendored frozen corpus:
>
> ```bash
> python -m tteEngine.reproduce      # -> outputs/corpus.jsonl + RESULTS_NARRATIVE.md
> ```

`tteEngine` is a **trial-general, multi-dataset Target Trial Emulation** system:
read any [ClinicalTrials.gov](https://clinicaltrials.gov) trial, derive a
dataset-agnostic extraction plan, materialize the emulated cohort from multiple
ICU EHR datasets (eICU-CRD, MIMIC-IV; MGB gated) normalized to **one common
format**, run an estimand-aware causal analysis, and compare the emulated effect
to the trial's *reported* result — at corpus scale, sepsis-prioritized.

---

## 1. End-to-end flow

```
ClinicalTrials.gov study (JSON, API v2)
        │  ctgov/reader.py (#1)  — fetch + cache
        ▼
TargetTrialSpec ───────────────────────────────── ctgov/spec.py (#2)
   eligibility · arms/comparator · outcomes · time-zero · estimand
        │  ctgov/intelligence.py (#3)  — the generalizer
        ▼
┌─ SEAM (a) ── ExtractionPlan ─────────────────── contracts/extraction_plan.py
│     dataset-AGNOSTIC: concepts + roles + window/timing
        │  adapters/{eicu,mimic,mgb}.py (#6/#7/#8) + vocab (#5)
        ▼
┌─ SEAM (b) ── 5-col EVENT stream ─────────────── contracts/events.py
│     canonical LONG parquet, one row per event
        │  cohort/builder.py (#9)  — eligibility · arm assignment · landmark
        ▼
┌─ SEAM (c) ── CohortResult ───────────────────── contracts/cohort.py
│     arms · index_times · wide feature VIEW · attrition diagnostics
        │  analysis/runner.py + engine.py (#10)  — PSM/IPTW/Cox/AIPW/KM
        ▼
┌─ SEAM (d) ── TTEResult ──────────────────────── contracts/results.py
│     measure · estimate · CI · n · extra{diagnostics,E-value,…}
        │  analysis/compare.py + benchmark.py (#11)  — vs reported effect
        ▼
┌─ SEAM (e) ── ComparisonResult ──────────────── contracts/results.py
│     emulated vs observed → concordant / discordant / inconclusive
        │  contracts/io.py (#36/#82)  — persist
        ▼
   corpus.jsonl  (one ComparisonResult per line; streams, >10k-safe)
        │
        ├─ analysis/meta.py (#64)         — concordance + Wilson CI; DL random-effects, I²/τ²; sepsis subgroup
        ├─ analysis/reliability.py (#41)  — calibration slope/intercept/coverage/RMSE
        ├─ analysis/drivers.py (#61)      — ranked concordance drivers → RESULTS_NARRATIVE.md
        └─ figures/* (#60) + web/results_app.py (#49/#40) — forest / calibration / KM + gallery
```

Orchestration (`orchestration/`, #12/#13) drives this resumably and batched over
a trial corpus (`ctgov/corpus.py`, #58); `run_corpus_to_jsonl` (#84) is the
import-light bridge from a live run to the persisted corpus the analysis reads.

---

## 2. The common format: a 5-column long event-stream (NOT OMOP)

**Decision (#4):** the canonical inter-dataset format is the **EHR-DE 5-column
LONG event-stream**, not OMOP CDM. Rationale: it is the format the source
extraction pipelines (EHR-DE/MIMIC-IV, the MGB Snowflake pipeline) already emit;
it is trivially append-only across heterogeneous event domains; and wide
cohort/feature tables are recovered as **deterministic VIEWS** over it rather
than a parallel source of truth. OMOP's relational normalization buys
interoperability we don't need here at the cost of per-dataset mapping work that
the vocab sidecar (below) handles more cheaply.

On-disk canonical = parquet with **exactly** these 5 columns
([`contracts/events.py`](../src/tteEngine/contracts/events.py)):

| Column | dtype | Meaning |
|---|---|---|
| `TRAJECTORY_ID` | `int64` | one hospital admission (≥1 ICU stay); `0` reserved for data-dictionary rows |
| `TIMESTAMP` | `datetime64[ns, UTC]` | event time; sub-day, orderable (immortal-time safe) |
| `EVENT_TYPE` | `str`/category | one of the `EventType` enum (`measu`/`lab`/`medic`/`diagn`/`proce`/`locat`/`demog`/`outco`/`micro`/`emar`/`drg`/`order`) |
| `EVENT_NAME` | `str` | raw source field identifier, pre-normalization |
| `EVENT_VALUE` | `str` | measured value, or JSON metadata |

Adapters **MUST** emit exactly `CANONICAL_COLUMNS` with `CANONICAL_DTYPES` and
**MUST NOT** emit free-string event types (`common_format.validate_canonical`
enforces this).

### Structure vs semantics are decoupled

The vocab layer (#5) does **not** mutate the 5 columns. It produces a **sidecar
normalized view** — `NormalizedEvent` (`concept_id` / `value_num` / `unit` /
`value_text`, keyed by row) — so that *structure* (this file, owned by the
common-format contract) and *semantics* (cross-DB concept mapping, owned by the
vocab lane) evolve independently. The analysis layer reads typed values from the
sidecar for threshold-based eligibility/outcomes.

---

## 3. The integration seams

Each seam is a typed Pydantic model in `tteEngine.contracts` — the lowest,
dependency-light layer. Seams are the *only* coupling between lanes; a lane can
be rewritten freely as long as its seam types are honored.

| Seam | Type | Producer → Consumer | File |
|---|---|---|---|
| **(a)** extraction plan | `ExtractionPlan` (+ `ConceptRequest`, `TimingConfig`) | intelligence #3 → adapters #6/#7/#8 | `contracts/extraction_plan.py` |
| **(b)** event stream | 5-col parquet / `Event` (+ `NormalizedEvent` sidecar) | adapters #6/#7/#8 → cohort #9 | `contracts/events.py` |
| **(c)** cohort | `CohortResult` (+ `ArmAssignment`, `CohortDiagnostics`) | cohort #9 → engine #10 | `contracts/cohort.py` |
| **(d)** engine result | `TTEResult` (`EffectMeasure`, estimate, CI, `extra{}`) | engine #10 → benchmark #11 | `contracts/results.py` |
| **(e)** comparison | `ComparisonResult` (`Agreement`) | benchmark #11 → persist/analysis/UI | `contracts/results.py` |

Notes that have bitten us before, encoded here:

- **`TTEResult` is the single engine→benchmark seam.** `run_tte` returns
  `contracts.results.TTEResult`; method-specific diagnostics (balance tables, KM
  curves, E-values) go in `extra{}`, never as a parallel result type. There is
  exactly one `TTEResult`.
- **`ExtractionPlan` is dataset-agnostic.** `dataset=None` is the portable plan;
  each adapter resolves it against its own raw schema via the vocab layer. Timing
  is unified by `TimingConfig` (#31); `timing=None` falls back to legacy
  `window_hours` (back-compat).
- **The wide feature table is a VIEW.** `CohortResult.feature_table_ref` points
  at a materialized parquet that is a deterministic projection of the canonical
  long stream — never an independent source of truth.
- **`CohortDiagnostics` makes attrition explicit** (screened → eligible →
  immortal-excluded → enrolled, landmark hours, leakage warnings). No silent
  attrition.

---

## 4. Import-light layering

The dependency graph points **down to `contracts/`** and never up:

```
            contracts/   (pydantic only — no heavy deps; the seam types + io.py)
              ▲     ▲
              │     │
   orchestration/   analysis/        ctgov/        adapters/ · cohort/ · figures/
   (#12/#13/#36)    (#10/#11/#64/…)  (#1/#2/#3)
```

- **Heavy deps live behind extras** (`pyproject.toml`): `analysis`
  (lifelines/statsmodels/scikit-learn/numpy/pandas/pyarrow), `ctgov` (requests),
  `viz` (matplotlib). Installing none of them still lets `import tteEngine` and
  the contracts work.
- **`analysis/engine.py` is imported lazily** by `run_tte`, so orchestration can
  build, persist, and even degrade gracefully without the `analysis` extra
  installed.
- **JSONL persistence lives in `contracts/io.py`**, deliberately *not* in
  `analysis/` — so orchestration persists the corpus stream **without importing
  analysis**, and meta/figures/UI all read the *one* saved corpus via
  `contracts.io.load_comparisons_jsonl`. No lane owns the seam; everyone imports
  down. (This resolved the two-reader / up-import debate; see git history #78/#82.)
- **Plot-ready data is import-light** (`figures/*` expose `forest_rows` /
  `calibration_points` / `km_data`) so the UI renders the same figures **without
  matplotlib**.

---

## 5. Analysis & rigor layer

Over the persisted `ComparisonResult` corpus (reader-agnostic — live stream or
saved JSONL):

- **Estimators (#10):** propensity-score matching (1:N, MAD caliper), stabilized
  & trimmed IPTW (cluster-robust SE), Cox PH, logistic OR, AIPW doubly-robust,
  Kaplan–Meier, log-rank, E-values. Self-contained port of the trialsim engine.
- **Estimand variants (#38):** ITT vs per-protocol; prespecified subgroups with a
  CI-overlap heterogeneity flag.
- **Meta-analysis (#64):** concordance rate + Wilson CI; DerSimonian–Laird
  random-effects pooling with I²/τ²/Cochran's Q; sepsis subgroup; forest rows.
- **Calibration (#41):** emulated-vs-observed slope/intercept (log scale),
  Pearson r, CI coverage, RMSE vs identity.
- **Robustness (#37/#39):** E-values, negative-control outcomes, positive/negative
  **control trials** for whole-pipeline calibration.
- **Drivers (#61):** ranked features predicting concordance → `RESULTS_NARRATIVE.md`.
- **Cross-dataset rigor (worker1):** measurability (#33), missingness (#34),
  variability explainer (#32), clock/window harmonization (#31, `docs/TIMING.md`)
  — explain *why* datasets differ per protocol element.
- **Figures + UI (#60/#49/#40):** forest / calibration / KM plots + a Streamlit
  results gallery with per-trial Emulation Cards, reading the persisted corpus.

---

## 6. Reproducibility

[`tteEngine.reproduce`](../src/tteEngine/reproduce.py) (#62) is the executable
entry to the whole artifact:

- **Frozen corpus:** `data/frozen_corpus_studies.jsonl` — a vendored, pinned,
  sepsis-first ClinicalTrials.gov snapshot (real trials with posted results), so
  the result reproduces **offline** with no live API.
- **Deterministic:** the OBSERVED side is the trials' real posted results; the
  EMULATED side defaults to a **seeded** synthetic emulator
  (`sha256(nct|dataset|seed)`), so `python -m tteEngine.reproduce` is
  byte-identical across runs and in CI.
- **Swap-in for the real numbers:** pass `emulate=<real engine pipeline>` (or run
  with MIMIC/eICU mounted) — *same command, same seeds* — to produce the live
  headline concordance/calibration figures.

The live MIMIC/eICU corpus run (which box has the data mounted) is the one
remaining gate to the real headline numbers; the machinery above is built and
green against the frozen/synthetic corpus.

---

## 7. Lanes (who owns what)

| Lane | Owner | Issues |
|---|---|---|
| ctgov ingestion + intelligence + engine + benchmark + analysis/journal | **probe** | #1 #2 #3 #10 #11 #37 #38 #39 #41 #61 #62 #63 #64 |
| per-DB adapters + vocab + measurability/missingness/variability/timing | **worker1** | #5 #6 #7 #8 #31 #32 #33 #34 |
| common format + cohort + orchestration + figures + UI | **tte1** | #0 #4 #9 #12 #13 #40 #49 #60 |

See [`README.md`](../README.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md) for
setup and the contribution workflow.
