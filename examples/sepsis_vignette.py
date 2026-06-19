"""End-to-end vignette (#13): the day-1 integration milestone, with the flip.

Reproduces the demo trial — **corticosteroid vs standard care in Sepsis-3 ->
28-day mortality** — through the WHOLE tteEngine spine, twice:

    ctgov trial -> TargetTrialSpec -> ExtractionPlan -> 5-col stream
                -> cohort (landmark t0 + arms) -> TTE -> emulated-vs-observed

It runs on a SYNTHETIC, deterministic, CONFOUNDED-BY-INDICATION cohort (sicker
patients are preferentially treated, with overlap) so it shows the whole point
of target trial emulation:

  * CRUDE estimate (bundled baseline)   -> apparent HARM (RR > 1) — confounded
  * IPTW-ADJUSTED estimate (#10 engine) -> the real BENEFIT (effect < 1)

The same `Pipeline` runs on REAL data by overriding two providers:
  - `extract`: `adapters.mimic.extract` / `adapters.eicu.extract` on real tables,
  - `tte`:     `make_engine_provider(...)` (here) — the estimand-aware engine.

Run:  python examples/sepsis_vignette.py   (the IPTW arm needs the `analysis` extra)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from tteEngine.common_format import Aggregation, FeatureSpec
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
OBSERVED_RR = 0.90  # the trial's reported direction: a mortality benefit

#: the confounder, materialized as the IPTW adjustment covariate
LACTATE = FeatureSpec(name="lactate_max", event_type=EventType.LAB, event_name="lactate",
                      agg=Aggregation.MAX, window_hours=(-24.0, 24.0))


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


def _stratum(rows, tid, *, lactate, n_control, control_deaths, n_treated, treated_deaths):
    """Append one severity stratum (a fixed lactate level) with both arms present
    (overlap) — within the stratum treatment is protective.
    """
    def patient(treated: bool, dies: bool):
        nonlocal tid
        rows.append((tid, T0 + timedelta(hours=-1), "diagn", "sepsis", "1"))
        rows.append((tid, T0, "lab", "lactate", f"{lactate}"))
        if treated:
            rows.append((tid, T0 + timedelta(hours=2), "medic", "hydrocortisone", "50"))
        if dies:
            rows.append((tid, T0 + timedelta(hours=120), "outco", "death", "1"))
        tid += 1

    for i in range(n_control):
        patient(False, i < control_deaths)
    for i in range(n_treated):
        patient(True, i < treated_deaths)
    return tid


def confounded_stream(scale: int = 2) -> pd.DataFrame:
    """Deterministic, confounded-by-indication sepsis cohort in 5-col format.

    Two severity strata (lactate 3.0 / 6.0). Sicker patients are preferentially
    treated (confounding) but BOTH arms appear in BOTH strata (overlap -> IPTW
    positivity holds). Within each stratum treatment lowers mortality. Net: the
    crude estimate is biased toward HARM; adjusting for lactate reveals benefit.
    """
    rows: list[tuple] = []
    tid = 1
    # low severity: mostly control, low mortality, treatment 20% -> 10%
    tid = _stratum(rows, tid, lactate=3.0, n_control=100 * scale, control_deaths=20 * scale,
                   n_treated=40 * scale, treated_deaths=4 * scale)
    # high severity: mostly treated, high mortality, treatment 50% -> 40%
    tid = _stratum(rows, tid, lactate=6.0, n_control=40 * scale, control_deaths=20 * scale,
                   n_treated=100 * scale, treated_deaths=40 * scale)
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    return df


def _request(dataset: str) -> TargetRequest:
    return TargetRequest(nct_id="NCT-SEPSIS-STEROID", dataset=dataset, seed_spec=demo_spec(),
                         observed_estimate=OBSERVED_RR, observed_measure=EffectMeasure.RR)


def run_crude(dataset: str, stream: pd.DataFrame):
    """Bundled baseline engine (no confounding adjustment)."""
    return Pipeline(_request(dataset), seed_events=stream).run()


def run_adjusted(dataset: str, stream: pd.DataFrame, adjustment: str = "iptw"):
    """The #10 engine via the engine provider, adjusting for lactate (the confounder)."""
    from tteEngine.orchestration.engine_provider import make_engine_provider

    provider = make_engine_provider([LACTATE], adjustment=adjustment)
    return Pipeline(_request(dataset), seed_events=stream, providers={"tte": provider}).run()


def run_vignette() -> dict[str, dict]:
    """Crude vs adjusted on synthetic MIMIC + eICU."""
    streams = {"MIMIC-IV": confounded_stream(scale=2), "eICU-CRD": confounded_stream(scale=1)}
    return {
        ds: {"crude": run_crude(ds, s), "adjusted": run_adjusted(ds, s)}
        for ds, s in streams.items()
    }


def main() -> None:
    print("=" * 74)
    print("tteEngine vignette — corticosteroid vs standard care, Sepsis-3 -> 28d mortality")
    print("=" * 74)
    for dataset, r in run_vignette().items():
        c, a = r["crude"].emulated, r["adjusted"].emulated
        print(f"\n[{dataset}]  n_treated={c.n_treated}  n_control={c.n_control}")
        print(f"  CRUDE     {c.measure.value} = {c.estimate:.3f}   -> {r['crude'].agreement.value}"
              f"   (confounded by indication: apparent harm)")
        ci = f" (95% CI {a.ci_low:.2f}-{a.ci_high:.2f})" if a.ci_low is not None else ""
        print(f"  IPTW-ADJ  {a.measure.value} = {a.estimate:.3f}{ci} -> {r['adjusted'].agreement.value}"
              f"   (adjust for lactate -> real benefit)")
    print(f"\n  observed RR = {OBSERVED_RR}.  Crude shows harm; adjusting for the confounder")
    print("  recovers the benefit. That reversal IS target trial emulation, end to end.")


if __name__ == "__main__":
    main()
