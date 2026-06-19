"""Smoke tests for the seam contracts (#4) — the integration boundaries every
lane codes to. These must stay green so adapters/engine/cohort can rely on them.
"""

from datetime import datetime, timezone

from tteEngine.contracts import (
    CANONICAL_COLUMNS,
    ArmAssignment,
    CohortResult,
    ConceptRequest,
    EligibilityCriterion,
    Event,
    EventType,
    ExtractionPlan,
    OutcomeSpec,
    TargetTrialSpec,
)
from tteEngine.contracts.events import DICTIONARY_TRAJECTORY_ID


def test_canonical_columns():
    assert CANONICAL_COLUMNS == (
        "TRAJECTORY_ID",
        "TIMESTAMP",
        "EVENT_TYPE",
        "EVENT_NAME",
        "EVENT_VALUE",
    )
    assert DICTIONARY_TRAJECTORY_ID == 0


def test_event_row():
    e = Event(
        trajectory_id=1,
        timestamp=datetime(2020, 1, 1, 3, 15, tzinfo=timezone.utc),
        event_type=EventType.LAB,
        event_name="lactate",
        event_value="4.2",
    )
    assert e.event_type.value == "lab"


def test_extended_event_types():
    # broader controlled set actually emitted by EHR-DE extraction_v1.py
    for code in ("measu", "lab", "medic", "micro", "emar", "drg", "order"):
        assert EventType(code)


def test_trial_spec_roundtrip():
    ts = TargetTrialSpec(
        nct_id="NCT001",
        eligibility=[EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS)],
        outcomes=[OutcomeSpec(name="28d mortality", horizon_hours=672)],
    )
    assert TargetTrialSpec.model_validate_json(ts.model_dump_json()) == ts


def test_extraction_plan_seam():
    p = ExtractionPlan(
        nct_id="NCT001",
        concepts=[ConceptRequest(concept="lactate", event_type=EventType.LAB, role="eligibility", require_value=True)],
    )
    assert p.concepts[0].require_value is True


def test_cohort_result_seam():
    cr = CohortResult(
        nct_id="NCT001",
        dataset="MIMIC-IV",
        arms=[
            ArmAssignment(name="treated", trajectory_ids=[1, 2]),
            ArmAssignment(name="control", is_control=True, trajectory_ids=[3]),
        ],
    )
    assert cr.n_by_arm() == {"treated": 2, "control": 1}
