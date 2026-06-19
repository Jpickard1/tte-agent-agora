"""Tests for emulability triage (#35): per-dataset scoring + the sortable catalog
that NEVER silently drops trials (logs reasons) and prioritizes sepsis."""

from tteEngine import triage
from tteEngine.contracts.events import EventType
from tteEngine.contracts.trial_spec import (
    Arm, Comparator, EligibilityCriterion, OutcomeSpec, TargetTrialSpec,
)


def _sepsis_trial():
    return TargetTrialSpec(
        nct_id="NCT-SEPSIS", title="Corticosteroids in Sepsis", condition="Sepsis-3",
        eligibility=[EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS,
                                          comparator=Comparator.EXISTS)],
        arms=[Arm(name="steroid", intervention_concepts=["hydrocortisone"]),
              Arm(name="control", is_control=True)],
        outcomes=[OutcomeSpec(name="28d mortality", concept="death", horizon_hours=28 * 24)],
    )


def _unemulable_trial():
    # no identifiable intervention + a non-EHR outcome -> not emulable (but kept)
    return TargetTrialSpec(
        nct_id="NCT-QOL", title="Behavioral therapy, quality of life",
        condition="outpatient anxiety",
        eligibility=[EligibilityCriterion(concept="anxiety", event_type=EventType.DIAGNOSIS)],
        arms=[Arm(name="therapy"), Arm(name="control", is_control=True)],  # no intervention concepts
        outcomes=[OutcomeSpec(name="quality of life", concept="qol", horizon_hours=365 * 24 * 2)],
    )


def test_sepsis_trial_is_emulable_and_flagged():
    s = triage.score_spec(_sepsis_trial(), "MIMIC-IV")
    assert s.emulable is True and s.is_sepsis is True
    assert s.exposure_ok == 1.0 and s.outcome_ok == 1.0 and s.score >= 0.5


def test_unemulable_trial_kept_with_reasons():
    s = triage.score_spec(_unemulable_trial(), "eICU-CRD")
    assert s.emulable is False
    assert any("no identifiable intervention" in r for r in s.reasons)  # exposure gap logged
    assert any("not reliably EHR-measurable" in r for r in s.reasons)   # outcome gap logged


def test_long_horizon_outcome_scored_down_not_dropped():
    spec = _sepsis_trial()
    spec.outcomes = [OutcomeSpec(name="5y survival", concept="survival", horizon_hours=5 * 365 * 24)]
    s = triage.score_spec(spec, "MIMIC-IV")
    assert s.outcome_ok == 0.4 and any("horizon exceeds" in r for r in s.reasons)


def test_catalog_never_drops_and_prioritizes_sepsis():
    cat = triage.build_catalog([_unemulable_trial(), _sepsis_trial()])
    summ = cat["summary"]
    # 2 trials x 2 datasets = 4 rows, NONE dropped
    assert summ["n_rows"] == 4 and summ["n_trials"] == 2
    assert summ["n_emulable"] == 2 and summ["n_not_emulable"] == 2
    assert summ["not_emulable_reasons"]                       # drop reasons logged
    # sepsis rows sorted first
    assert cat["catalog"][0]["is_sepsis"] is True
    assert summ["n_emulable_sepsis"] == 2


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
