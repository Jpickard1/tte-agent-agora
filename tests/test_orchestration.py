"""Tests for the #12 orchestration spine: end-to-end run on synthetic data,
provider injection, and the check-&-correct edit/resume loop.
"""

from datetime import datetime, timedelta, timezone

import pytest

pd = pytest.importorskip("pandas")

from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType  # noqa: E402
from tteEngine.contracts.results import Agreement, EffectMeasure  # noqa: E402
from tteEngine.contracts.trial_spec import (  # noqa: E402
    Arm,
    Comparator,
    EligibilityCriterion,
    OutcomeSpec,
    TargetTrialSpec,
    TimeZeroRule,
)
from tteEngine.orchestration import Pipeline, TargetRequest  # noqa: E402

T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _hr(h):
    return T0 + timedelta(hours=h)


def _events():
    rows = [
        # treated (steroid) trajectories 1,2 ; one dies
        (1, _hr(-1), "diagn", "sepsis", "1"), (1, _hr(0), "lab", "lactate", "4"),
        (1, _hr(2), "medic", "hydrocortisone", "50"), (1, _hr(50), "outco", "death", "1"),
        (2, _hr(-1), "diagn", "sepsis", "1"), (2, _hr(0), "lab", "lactate", "5"),
        (2, _hr(3), "medic", "hydrocortisone", "50"),
        # control trajectories 3,4 ; one dies
        (3, _hr(-1), "diagn", "sepsis", "1"), (3, _hr(0), "lab", "lactate", "4"),
        (3, _hr(60), "outco", "death", "1"),
        (4, _hr(-1), "diagn", "sepsis", "1"), (4, _hr(0), "lab", "lactate", "6"),
    ]
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    return df


def _spec():
    return TargetTrialSpec(
        nct_id="NCT-DEMO",
        eligibility=[EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS,
                                          comparator=Comparator.EXISTS)],
        arms=[Arm(name="steroid", intervention_concepts=["hydrocortisone"]),
              Arm(name="control", is_control=True)],
        outcomes=[OutcomeSpec(name="mortality", event_type=EventType.OUTCOME, concept="death",
                              horizon_hours=672)],
        time_zero=TimeZeroRule(anchor="lactate", grace_window_hours=24.0),
    )


def test_end_to_end_synthetic_run():
    req = TargetRequest(nct_id="NCT-DEMO", dataset="TEST", seed_spec=_spec(),
                        observed_estimate=0.9, observed_measure=EffectMeasure.RR)
    pipe = Pipeline(req, seed_events=_events())
    report = pipe.run()
    assert report.nct_id == "NCT-DEMO"
    assert report.emulated.n_treated == 2 and report.emulated.n_control == 2
    # treated 1/2 died, control 1/2 died -> crude RR ~ 1.0
    assert report.emulated.estimate == pytest.approx(1.0)
    assert report.agreement in (Agreement.CONCORDANT, Agreement.DISCORDANT)


def test_missing_extract_provider_raises_without_seed():
    pipe = Pipeline(TargetRequest(nct_id="X", seed_spec=_spec()))  # no seed_events, no adapter
    with pytest.raises(RuntimeError):
        pipe.run()


def test_provider_injection_overrides_engine():
    from tteEngine.contracts.results import TTEResult

    def fake_engine(events, cohort, spec):
        return TTEResult(nct_id=spec.nct_id, dataset=cohort.dataset, method="injected",
                         measure=EffectMeasure.HR, estimate=0.5, n_treated=1, n_control=1)

    pipe = Pipeline(TargetRequest(nct_id="NCT-DEMO", dataset="TEST", seed_spec=_spec()),
                    seed_events=_events(), providers={"tte": fake_engine})
    report = pipe.run()
    assert report.emulated.method == "injected"
    assert report.emulated.measure == EffectMeasure.HR


def test_check_and_correct_edit_resume_invalidates_downstream():
    req = TargetRequest(nct_id="NCT-DEMO", dataset="TEST", seed_spec=_spec())
    pipe = Pipeline(req, seed_events=_events())
    pipe.run_until("spec")
    # narrow the grace window to 2.5h -> trajectory 2's steroid (3h) falls out of window
    edited = _spec().model_copy(update={"time_zero": TimeZeroRule(anchor="lactate", grace_window_hours=2.5)})
    pipe.edit("spec", edited)
    assert pipe.get("cohort") is None  # downstream invalidated
    pipe.resume()
    by_arm = {a.name: a.trajectory_ids for a in pipe.get("cohort").arms}
    assert by_arm.get("steroid") == [1]      # only traj 1's steroid (2h) is within 2.5h
    assert 2 in by_arm.get("control", [])    # traj 2 now control
