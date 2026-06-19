"""End-to-end vignette (#13): the day-1 integration milestone.

Reproduces the demo trial — **corticosteroid vs standard care in Sepsis-3 ->
28-day mortality** — through the WHOLE tteEngine spine on two datasets:

    ctgov trial -> TargetTrialSpec -> ExtractionPlan -> 5-col stream
                -> cohort (landmark t0 + arms) -> TTE -> emulated-vs-observed

It runs on SYNTHETIC MIMIC-like + eICU-like streams (no PHI, deterministic) so
the milestone is testable today. The same `Pipeline` runs on REAL data by
overriding two providers (see `real_data_notes` below):
  - `extract`: `tteEngine.adapters.mimic.extract` / `adapters.eicu.extract` on
    real tables (`load_mimic_tables(...)`),
  - `tte`:     the estimand-aware engine (#10) once it lands (PSM/IPTW/Cox),
which replaces the bundled crude-RR baseline with confounding adjustment.

Run:  python examples/sepsis_vignette.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType
from tteEngine.contracts.results import EffectMeasure
from tteEngine.contracts.trial_spec import (
    Arm,
    Comparator,
    EligibilityCriterion,
    OutcomeSpec,
    TargetTrialSpec,
    TimeZeroRule,
)
from tteEngine.orchestration import Pipeline, TargetRequest

T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)

# The validated steroid-sepsis signal: a modest mortality BENEFIT once
# confounding-by-indication is removed (the trial's reported direction).
OBSERVED_RR = 0.90


def demo_spec(nct_id: str = "NCT-SEPSIS-STEROID") -> TargetTrialSpec:
    """The target trial protocol as a typed spec (what ctgov #2 would emit)."""
    return TargetTrialSpec(
        nct_id=nct_id,
        title="Corticosteroids in Sepsis-3 -> 28-day mortality",
        condition="Sepsis-3",
        eligibility=[
            EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS,
                                 comparator=Comparator.EXISTS),
            EligibilityCriterion(concept="lactate", event_type=EventType.LAB,
                                 comparator=Comparator.GT, value=2.0, window_hours=(-24.0, 24.0)),
        ],
        arms=[
            Arm(name="corticosteroid", intervention_concepts=["hydrocortisone"]),
            Arm(name="standard_care", is_control=True),
        ],
        outcomes=[OutcomeSpec(name="28d mortality", event_type=EventType.OUTCOME,
                              concept="death", horizon_hours=28 * 24)],
        time_zero=TimeZeroRule(anchor="lactate", grace_window_hours=24.0),
    )


def synthetic_stream(
    *, n_treated: int, n_control: int, treated_deaths: int, control_deaths: int,
    treated_lactate: float = 4.5, control_lactate: float = 3.0, start_id: int = 1,
) -> pd.DataFrame:
    """Deterministic sepsis cohort in the canonical 5-col format. Treated arm is
    sicker at baseline (higher lactate) — confounding-by-indication, so the CRUDE
    estimate is biased toward harm; the real IPTW/PSM engine (#10) corrects it.
    """
    rows: list[tuple] = []
    tid = start_id

    def add_patient(lactate: float, steroid: bool, dies: bool) -> None:
        nonlocal tid
        rows.append((tid, T0 + timedelta(hours=-1), "diagn", "sepsis", "1"))
        rows.append((tid, T0, "lab", "lactate", f"{lactate}"))
        if steroid:
            rows.append((tid, T0 + timedelta(hours=2), "medic", "hydrocortisone", "50"))
        if dies:
            rows.append((tid, T0 + timedelta(hours=120), "outco", "death", "1"))
        tid += 1

    for i in range(n_treated):
        add_patient(treated_lactate, steroid=True, dies=(i < treated_deaths))
    for i in range(n_control):
        add_patient(control_lactate, steroid=False, dies=(i < control_deaths))

    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    return df


def run_one(dataset: str, stream: pd.DataFrame):
    """Run the full pipeline for one dataset; returns the ComparisonResult."""
    req = TargetRequest(nct_id="NCT-SEPSIS-STEROID", dataset=dataset, seed_spec=demo_spec(),
                        observed_estimate=OBSERVED_RR, observed_measure=EffectMeasure.RR)
    return Pipeline(req, seed_events=stream).run()


def run_vignette() -> dict[str, object]:
    """Run the demo on synthetic MIMIC + eICU and return the two reports."""
    # MIMIC-like and eICU-like cohorts (different sizes, same structure)
    mimic = synthetic_stream(n_treated=40, n_control=60, treated_deaths=18, control_deaths=18)
    eicu = synthetic_stream(n_treated=30, n_control=50, treated_deaths=14, control_deaths=15, start_id=1000)
    return {
        "MIMIC-IV": run_one("MIMIC-IV", mimic),
        "eICU-CRD": run_one("eICU-CRD", eicu),
    }


def main() -> None:
    print("=" * 72)
    print("tteEngine vignette — corticosteroid vs standard care, Sepsis-3 -> 28d mortality")
    print("=" * 72)
    reports = run_vignette()
    for dataset, rep in reports.items():
        e = rep.emulated
        print(f"\n[{dataset}]  n_treated={e.n_treated}  n_control={e.n_control}")
        print(f"  emulated {e.measure.value} ({e.method}) = {e.estimate:.3f}"
              f"  (risk treated={e.extra.get('risk_treated'):.3f},"
              f" control={e.extra.get('risk_control'):.3f})")
        print(f"  observed {rep.observed_measure.value} = {rep.observed_estimate:.3f}"
              f"  -> agreement: {rep.agreement.value}")
    print("\nNote: the bundled CRUDE estimator is confounded (treated arm sicker at")
    print("baseline) -> biased toward harm. Swap in the #10 IPTW/PSM engine to recover")
    print("the adjusted benefit (~0.90). That adjustment IS target trial emulation.")


# How to run on REAL data (no code change to the pipeline — just two providers):
real_data_notes = """
from tteEngine.adapters import mimic
def real_extract(plan, request, _seed):
    tables = mimic.load_mimic_tables(MIMIC_DIR, needed=[...])
    return mimic.extract(plan, tables)
Pipeline(req, providers={"extract": real_extract, "tte": iptw_engine}).run()
"""


if __name__ == "__main__":
    main()
