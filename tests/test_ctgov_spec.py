"""Tests for ctgov study -> TargetTrialSpec (#2, probe)."""

from tteEngine.contracts import Comparator, EventType
from tteEngine.ctgov.spec import (
    _age_to_years,
    _timeframe_to_hours,
    study_to_spec,
)


def _study() -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT001", "briefTitle": "Steroids in Sepsis"},
            "conditionsModule": {"conditions": ["Sepsis", "Septic Shock"]},
            "eligibilityModule": {
                "minimumAge": "18 Years",
                "maximumAge": "90 Years",
                "sex": "ALL",
            },
            "armsInterventionsModule": {
                "armGroups": [
                    {"label": "Hydrocortisone", "type": "EXPERIMENTAL",
                     "interventionNames": ["Drug: Hydrocortisone"]},
                    {"label": "Placebo", "type": "PLACEBO_COMPARATOR",
                     "interventionNames": ["Drug: Placebo"]},
                ]
            },
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "28-day mortality", "timeFrame": "28 days"}],
                "secondaryOutcomes": [{"measure": "ICU length of stay", "timeFrame": "Up to 90 days"}],
            },
        }
    }


def test_age_parser():
    assert _age_to_years("18 Years") == 18.0
    assert _age_to_years("6 Months") == 0.5
    assert _age_to_years("N/A") is None
    assert _age_to_years(None) is None


def test_timeframe_parser():
    assert _timeframe_to_hours("28 days") == 672.0
    assert _timeframe_to_hours("2 Weeks") == 336.0
    assert _timeframe_to_hours("unspecified") is None


def test_study_to_spec_basics():
    spec = study_to_spec(_study())
    assert spec.nct_id == "NCT001"
    assert spec.title == "Steroids in Sepsis"
    assert spec.condition == "Sepsis"


def test_arms_control_detection():
    spec = study_to_spec(_study())
    by_name = {a.name: a for a in spec.arms}
    assert by_name["Hydrocortisone"].is_control is False
    assert by_name["Placebo"].is_control is True  # PLACEBO_COMPARATOR + label
    assert "Drug: Hydrocortisone" in by_name["Hydrocortisone"].intervention_concepts


def test_outcomes_parsed_with_horizon():
    spec = study_to_spec(_study())
    prim = spec.outcomes[0]
    assert prim.name == "28-day mortality"
    assert prim.horizon_hours == 672.0
    assert prim.event_type == EventType.OUTCOME
    assert len(spec.outcomes) == 2


def test_eligibility_demographics():
    spec = study_to_spec(_study())
    ages = [c for c in spec.eligibility if c.concept == "age"]
    assert {c.comparator for c in ages} == {Comparator.GE, Comparator.LE}
    assert all(c.event_type == EventType.DEMOGRAPHIC for c in spec.eligibility)
    # sex == ALL -> no sex predicate
    assert not [c for c in spec.eligibility if c.concept == "sex"]


def test_roundtrip():
    spec = study_to_spec(_study())
    from tteEngine.contracts import TargetTrialSpec
    assert TargetTrialSpec.model_validate_json(spec.model_dump_json()) == spec
