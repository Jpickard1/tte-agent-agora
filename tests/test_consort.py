"""#29 CONSORT attrition report tests. Builds a cohort (with #30 diagnostics)
and checks the flow reconciles, the diagram renders, and the corpus aggregate
sums. Skips without pandas."""

from datetime import datetime, timedelta, timezone

import pytest

pd = pytest.importorskip("pandas")

from tteEngine.cohort import (  # noqa: E402
    aggregate_diagnostics,
    build_cohort,
    consort_flow,
    format_consort,
)
from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType  # noqa: E402
from tteEngine.contracts.trial_spec import (  # noqa: E402
    Arm,
    Comparator,
    EligibilityCriterion,
    OutcomeSpec,
    TargetTrialSpec,
    TimeZeroRule,
)

T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _hr(h):
    return T0 + timedelta(hours=h)


def _frame(rows):
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    return df


SPEC = TargetTrialSpec(
    nct_id="NCT-CONSORT",
    eligibility=[EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS,
                                      comparator=Comparator.EXISTS)],
    arms=[Arm(name="steroid", intervention_concepts=["hydrocortisone"]),
          Arm(name="control", is_control=True)],
    outcomes=[OutcomeSpec(name="mortality", event_type=EventType.OUTCOME, concept="death",
                          horizon_hours=672)],
    time_zero=TimeZeroRule(anchor="lactate", grace_window_hours=24.0),
)

# 1 treated (kept), 1 control (kept), 1 ineligible (no sepsis), 1 immortal (dies @+5h)
EVENTS = _frame([
    (1, _hr(-1), "diagn", "sepsis", "1"), (1, _hr(0), "lab", "lactate", "4"),
    (1, _hr(2), "medic", "hydrocortisone", "50"),
    (2, _hr(-1), "diagn", "sepsis", "1"), (2, _hr(0), "lab", "lactate", "4"),
    (3, _hr(0), "lab", "lactate", "4"),  # no sepsis -> ineligible
    (4, _hr(-1), "diagn", "sepsis", "1"), (4, _hr(0), "lab", "lactate", "4"),
    (4, _hr(5), "outco", "death", "1"),  # dies before 24h landmark -> immortal-time excluded
])


def test_consort_flow_reconciles():
    d = build_cohort(EVENTS, SPEC, dataset="TEST").diagnostics
    f = consort_flow(d)
    assert f["screened"] == 4
    assert f["excluded_ineligible"] == 1     # traj 3
    assert f["eligible"] == 3                # 1,2,4
    assert f["excluded_immortal_time"] == 1  # traj 4
    assert f["enrolled"] == 2                # 1,2
    # the flow fully reconciles: nothing unaccounted for
    assert f["excluded_ineligible"] + f["excluded_immortal_time"] + f["enrolled"] == f["screened"]


def test_format_consort_renders():
    d = build_cohort(EVENTS, SPEC, dataset="TEST").diagnostics
    text = format_consort(d, title="Sepsis/steroid")
    assert "screened" in text and "immortal-time" in text and "steroid: 1" in text


def test_aggregate_diagnostics_sums_corpus():
    d1 = build_cohort(EVENTS, SPEC, dataset="A").diagnostics
    d2 = build_cohort(EVENTS, SPEC, dataset="B").diagnostics
    agg = aggregate_diagnostics([d1, d2])
    assert agg["n_cohorts"] == 2
    assert agg["screened"] == 8 and agg["enrolled"] == 4 and agg["excluded_immortal"] == 2
