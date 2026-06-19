"""Test the #28 enrichment wire into study_to_spec (#2, probe). Opt-in
enrich_eligibility upgrades demographics-only eligibility with free-text
inclusion/exclusion predicates. Pure (no analysis extra)."""

from tteEngine.contracts import TargetTrialSpec
from tteEngine.ctgov.spec import study_to_spec


def _study() -> dict:
    return {"protocolSection": {
        "identificationModule": {"nctId": "NCT9", "briefTitle": "t"},
        "conditionsModule": {"conditions": ["Sepsis"]},
        "eligibilityModule": {
            "minimumAge": "18 Years", "sex": "ALL",
            "eligibilityCriteria": ("Inclusion Criteria:\n* Adults with sepsis\n"
                                    "* Lactate >= 2 mmol/L\nExclusion Criteria:\n"
                                    "* Pregnancy\n* Age < 18"),
        },
        "armsInterventionsModule": {"armGroups": [
            {"label": "Drug", "interventionNames": ["Drug: X"]},
            {"label": "Placebo", "type": "PLACEBO_COMPARATOR"}]},
        "outcomesModule": {"primaryOutcomes": [{"measure": "28-day mortality", "timeFrame": "28 days"}]},
    }}


def test_base_is_demographics_only():
    base = study_to_spec(_study())
    assert isinstance(base, TargetTrialSpec)
    assert [c.concept for c in base.eligibility] == ["age"]   # opt-out: demographics only


def test_enrich_adds_clinical_predicates():
    enr = study_to_spec(_study(), enrich_eligibility=True)
    concepts = {c.concept for c in enr.eligibility}
    assert {"sepsis", "lactate"} <= concepts          # free-text inclusions parsed in
    assert any(c.concept == "pregnancy" and not c.include for c in enr.eligibility)  # exclusion
    assert len(enr.eligibility) > 1                   # strictly richer than the base


def test_enrich_is_idempotent_dedup():
    # enriching does not duplicate the demographic age>=18 already present
    enr = study_to_spec(_study(), enrich_eligibility=True)
    age_ge = [c for c in enr.eligibility if c.concept == "age" and c.comparator.value == "ge"]
    assert len(age_ge) == 1
