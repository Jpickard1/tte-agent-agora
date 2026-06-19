"""Tests for the #9 cohort builder: eligibility, landmark time-zero, arm
assignment, and the analysis frame. Skips if pandas isn't installed.
"""

from datetime import datetime, timedelta, timezone

import pytest

pd = pytest.importorskip("pandas")

from tteEngine.cohort import build_analysis_frame, build_cohort  # noqa: E402
from tteEngine.common_format import Aggregation, FeatureSpec  # noqa: E402
from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType  # noqa: E402
from tteEngine.contracts.trial_spec import (  # noqa: E402
    Arm,
    Comparator,
    EligibilityCriterion,
    OutcomeSpec,
    TargetTrialSpec,
    TimeZeroRule,
)

T0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)


def _frame(rows):
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    return df


def _hr(h):
    return T0 + timedelta(hours=h)


# sepsis/steroid-style demo: 3 trajectories
#  1: septic (lactate>2) + steroid in window -> treated, dies in horizon
#  2: septic + no steroid -> control, survives
#  3: NOT septic -> excluded
EVENTS = _frame([
    (1, _hr(-1), "diagn", "sepsis", "1"),
    (1, _hr(0), "lab", "lactate", "4.0"),
    (1, _hr(2), "medic", "hydrocortisone", "50"),
    (1, _hr(100), "outco", "death", "1"),
    (2, _hr(-1), "diagn", "sepsis", "1"),
    (2, _hr(0), "lab", "lactate", "3.0"),
    (3, _hr(0), "lab", "lactate", "5.0"),  # high lactate but no sepsis dx
])

SPEC = TargetTrialSpec(
    nct_id="NCT-DEMO",
    eligibility=[
        EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS, comparator=Comparator.EXISTS),
        EligibilityCriterion(concept="lactate", event_type=EventType.LAB, comparator=Comparator.GT, value=2.0),
    ],
    arms=[
        Arm(name="steroid", intervention_concepts=["hydrocortisone"]),
        Arm(name="control", is_control=True),
    ],
    outcomes=[OutcomeSpec(name="28d mortality", event_type=EventType.OUTCOME, concept="death", horizon_hours=672)],
    time_zero=TimeZeroRule(anchor="lactate", grace_window_hours=24.0),
)


def test_eligibility_excludes_non_sepsis():
    c = build_cohort(EVENTS, SPEC, dataset="TEST")
    enrolled = sorted(tid for arm in c.arms for tid in arm.trajectory_ids)
    assert enrolled == [1, 2]  # trajectory 3 excluded (no sepsis dx)
    assert c.n_total == 2


def test_arm_assignment_by_treatment_window():
    c = build_cohort(EVENTS, SPEC, dataset="TEST")
    by_arm = {a.name: a.trajectory_ids for a in c.arms}
    assert by_arm["steroid"] == [1]
    assert by_arm["control"] == [2]
    assert any(a.is_control for a in c.arms)


def test_landmark_time_zero_is_anchor_event():
    c = build_cohort(EVENTS, SPEC, dataset="TEST")
    # anchor = first 'lactate' event, at T0 for both enrolled
    assert c.index_times[1] == pd.Timestamp(T0)
    assert c.index_times[2] == pd.Timestamp(T0)


def test_steroid_after_grace_window_is_control():
    late = _frame([
        (5, _hr(-1), "diagn", "sepsis", "1"),
        (5, _hr(0), "lab", "lactate", "4.0"),
        (5, _hr(48), "medic", "hydrocortisone", "50"),  # after 24h grace -> not treated
    ])
    c = build_cohort(late, SPEC, dataset="TEST")
    by_arm = {a.name: a.trajectory_ids for a in c.arms}
    assert by_arm.get("control") == [5]
    assert "steroid" not in by_arm


def test_analysis_frame_group_outcome_covariate():
    c = build_cohort(EVENTS, SPEC, dataset="TEST")
    covs = [FeatureSpec(name="lactate_max", event_type=EventType.LAB, event_name="lactate",
                        agg=Aggregation.MAX, window_hours=(-24.0, 24.0))]
    frame = build_analysis_frame(EVENTS, c, SPEC, covariates=covs)
    frame = frame.set_index("TRAJECTORY_ID")
    assert frame.loc[1, "group"] == "steroid"
    assert frame.loc[2, "group"] == "control"
    assert frame.loc[1, "lactate_max"] == 4.0
    # mortality outcome within 672h horizon
    assert bool(frame.loc[1, "outcome_28d_mortality"]) is True
    assert bool(frame.loc[2, "outcome_28d_mortality"]) is False
    assert "outcome_28d_mortality" in c.feature_columns
