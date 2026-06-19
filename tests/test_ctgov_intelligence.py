"""Tests for the extraction-intelligence: TargetTrialSpec -> ExtractionPlan (#3)."""

from tteEngine.contracts import (
    Arm,
    Comparator,
    EligibilityCriterion,
    EventType,
    OutcomeSpec,
    TargetTrialSpec,
)
from tteEngine.ctgov.intelligence import spec_to_plan


def _spec() -> TargetTrialSpec:
    return TargetTrialSpec(
        nct_id="NCT001",
        condition="Sepsis",
        eligibility=[
            EligibilityCriterion(concept="age", event_type=EventType.DEMOGRAPHIC,
                                 comparator=Comparator.GE, value=18),
            EligibilityCriterion(concept="lactate", event_type=EventType.LAB,
                                 comparator=Comparator.GE, value=2.0, unit="mmol/L"),
            EligibilityCriterion(concept="septic_shock", event_type=EventType.DIAGNOSIS),
        ],
        arms=[
            Arm(name="Hydrocortisone", intervention_concepts=["hydrocortisone"]),
            Arm(name="Placebo", is_control=True, intervention_concepts=[]),
        ],
        outcomes=[OutcomeSpec(name="28d mortality", event_type=EventType.OUTCOME, kind="binary")],
        covariates=["sofa"],
    )


def test_roles_and_require_value():
    plan = spec_to_plan(_spec())
    roles = {c.role for c in plan.concepts}
    assert {"eligibility", "exposure", "outcome", "covariate"} <= roles
    lactate = next(c for c in plan.concepts if c.concept == "lactate")
    assert lactate.role == "eligibility" and lactate.require_value is True
    out = next(c for c in plan.concepts if c.role == "outcome")
    assert out.require_value is False  # binary outcome -> no value needed


def test_exposures_from_arms():
    plan = spec_to_plan(_spec())
    exp = [c for c in plan.concepts if c.role == "exposure"]
    assert any(c.concept == "hydrocortisone" and c.event_type == EventType.MEDICATION for c in exp)


def test_cohort_filter_has_condition_and_dx():
    plan = spec_to_plan(_spec())
    assert "Sepsis" in plan.cohort_filter_concepts
    assert "septic_shock" in plan.cohort_filter_concepts  # inclusion diagnosis


def test_default_covariates_added():
    covs = {c.concept for c in spec_to_plan(_spec()).concepts if c.role == "covariate"}
    assert {"age", "sex", "sofa"} <= covs


def test_vocab_resolver_applied():
    def vocab(concept, event_type):
        return {"hydrocortisone": "RxNorm:5492", "Sepsis": "ICD10:A41"}.get(concept, concept)

    plan = spec_to_plan(_spec(), vocab=vocab)
    assert any(c.concept == "RxNorm:5492" for c in plan.concepts)
    assert "ICD10:A41" in plan.cohort_filter_concepts


def test_dataset_tag_and_window():
    plan = spec_to_plan(_spec(), dataset="MIMIC-IV")
    assert plan.dataset == "MIMIC-IV"
    assert plan.window_hours == (-48.0, 24.0)
    assert "MIMIC-IV" in (plan.notes or "")


def test_exposure_dedup():
    spec = _spec()
    spec.arms[0].intervention_concepts.append("hydrocortisone")  # duplicate
    exp = [c for c in spec_to_plan(spec).concepts
           if c.role == "exposure" and c.concept == "hydrocortisone"]
    assert len(exp) == 1
