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


def test_immortal_time_guard_excludes_outcome_before_landmark():
    # patient dies at +10h, before the 24h landmark -> excluded (immortal time)
    early = _frame([
        (9, _hr(-1), "diagn", "sepsis", "1"),
        (9, _hr(0), "lab", "lactate", "4.0"),
        (9, _hr(2), "medic", "hydrocortisone", "50"),
        (9, _hr(10), "outco", "death", "1"),
    ])
    c = build_cohort(early, SPEC, dataset="TEST")
    enrolled = [tid for arm in c.arms for tid in arm.trajectory_ids]
    assert 9 not in enrolled
    assert c.diagnostics.n_excluded_immortal == 1
    assert c.diagnostics.n_eligible == 1 and c.diagnostics.n_enrolled == 0


def test_outcome_after_landmark_is_kept():
    c = build_cohort(EVENTS, SPEC, dataset="TEST")  # deaths at +100h, landmark 24h
    assert c.diagnostics.n_excluded_immortal == 0
    assert c.diagnostics.n_enrolled == 2


def test_diagnostics_populated():
    d = build_cohort(EVENTS, SPEC, dataset="TEST").diagnostics
    assert d.n_screened == 3 and d.n_eligible == 2
    assert d.anchor == "lactate" and d.landmark_hours == 24.0
    assert d.arm_sizes == {"steroid": 1, "control": 1}


def test_post_t0_eligibility_window_is_flagged():
    from tteEngine.contracts.trial_spec import Comparator, EligibilityCriterion
    spec2 = SPEC.model_copy(update={"eligibility": [
        EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS, comparator=Comparator.EXISTS),
        EligibilityCriterion(concept="lactate", event_type=EventType.LAB, comparator=Comparator.GT,
                             value=2.0, window_hours=(0.0, 24.0))]})
    d = build_cohort(EVENTS, spec2, dataset="TEST").diagnostics
    assert any("post-t0" in w for w in d.leakage_warnings)


def test_landmark_guard_can_be_disabled():
    early = _frame([
        (9, _hr(-1), "diagn", "sepsis", "1"), (9, _hr(0), "lab", "lactate", "4.0"),
        (9, _hr(10), "outco", "death", "1")])
    c = build_cohort(early, SPEC, dataset="TEST", enforce_landmark=False)
    assert c.diagnostics.n_excluded_immortal == 0


def test_arm_assignment_normalizes_drug_prefix_and_substring():
    # trial intervention 'Drug: Thiamine'; MIMIC med 'Thiamine 100mg' -> should match
    spec = SPEC.model_copy(update={"arms": [
        Arm(name="thiamine", intervention_concepts=["Drug: Thiamine"]),
        Arm(name="control", is_control=True)]})
    ev = _frame([
        (1, _hr(-1), "diagn", "sepsis", "1"), (1, _hr(0), "lab", "lactate", "4"),
        (1, _hr(2), "medic", "Thiamine 100mg", "1"),
        (2, _hr(-1), "diagn", "sepsis", "1"), (2, _hr(0), "lab", "lactate", "4")])
    c = build_cohort(ev, spec, dataset="TEST")
    by = {a.name: a.trajectory_ids for a in c.arms}
    assert by.get("thiamine") == [1] and by.get("control") == [2]


def test_unmeasurable_eligibility_skipped_not_failing():
    # an age (DEMOGRAPHIC) criterion with NO demog events present -> skipped, not failing all
    spec = SPEC.model_copy(update={"eligibility": SPEC.eligibility + [
        EligibilityCriterion(concept="age", event_type=EventType.DEMOGRAPHIC,
                             comparator=Comparator.GE, value=18.0)]})
    c = build_cohort(EVENTS, spec, dataset="TEST")  # EVENTS has no DEMOGRAPHIC events
    assert c.n_total == 2  # 1,2 still enrolled (age criterion skipped, not failed)
    d = c.diagnostics
    assert d.n_skipped_unmeasurable == 1
    assert any("age" in s for s in d.skipped_eligibility)


def test_measurable_criteria_still_applied():
    # sanity: a measurable criterion (sepsis dx present) is still enforced
    c = build_cohort(EVENTS, SPEC, dataset="TEST")
    assert c.diagnostics.n_skipped_unmeasurable == 0  # sepsis + lactate both present
    assert c.n_total == 2


def test_skip_unmeasurable_can_be_disabled():
    spec = SPEC.model_copy(update={"eligibility": SPEC.eligibility + [
        EligibilityCriterion(concept="age", event_type=EventType.DEMOGRAPHIC,
                             comparator=Comparator.GE, value=18.0)]})
    c = build_cohort(EVENTS, spec, dataset="TEST", skip_unmeasurable=False)
    assert c.n_total == 0  # age criterion now fails everyone (no demog events)
